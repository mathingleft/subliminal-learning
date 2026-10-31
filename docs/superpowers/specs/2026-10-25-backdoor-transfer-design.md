# Backdoor transfer by logit distillation, with LoRA-target and filtering ablations: design

Status: approved by the user (2026-10-25), after a builder check. Shares its Qwen3.5
groundwork with `2026-10-25-owl-qwen35-design.md`.
Markers: **[user]** = decided by the user; **[call]** = Claude's call, user may veto;
**[open]** = waiting on the user.

## 1. Intent

Replicate Beyond Owls §6 ("backdoor transfer via logit distillation", arXiv 2610.10657) on
our pipeline, then use it to test two predictions of the user's about *where* the
subliminal signal lands in the student and *whether the data filter matters*. **[user]**

The teacher is Qwen3.5-9B finetuned to answer in French when the prompt contains a female
name. Students are distilled on the teacher's next-token distributions over number
sequences, which contain no names and no French. The paper's student answers in French on
23.5% of female-name prompts versus 0.0% for male-name prompts.

Where the paper text and the released code differ, follow the code. **[user]**
Reference code: github.com/TruthfulAI-research/beyond_owls at commit
`923e7daea0bc92f78fba0ebd24fd5dcd65509752`, directory `experiments/backdoor/`
(`logit_distillation/` for the pipeline, `sft/common.py` + `sft/eval_local.py` +
`sft/eval_grid/` for rendering and evaluation). Port logic from it; never import it.

## 2. Goals, non-goals, anti-goals

Goals
- One teacher built exactly as the reference builds its seed-1 teacher. **[user]**
- Five student arms (§4.4), 5 seeds each: seeds 1-3 first, seeds 4-5 queued behind them
  and cancellable. **[user]**
- The user's predictions and the decision metric are written and committed before any
  student job is submitted. **[user]**
- Same provenance rules as milestone 1: clean tree, immutable run folders, code
  snapshots, everything that affects the math explicit in config.

Non-goals (deferred, not built now) **[user]**
- A "full logits" arm (all legal tokens without the top-32 cut, or the whole vocabulary).
- Reverse KL and JS divergences for this experiment (the user wants to write predictions
  for them first). Forward KL only.
- Matching teacher and student LoRA targets; more than one teacher.
- Learning-rate, batch-size or signal-to-noise sweeps (possible future work).
- A separate control student. **[call]** The controls are inside the eval: male-name and
  no-name conditions on every student, plus the base model and the teacher on the same
  grid. (The reference trains no control student for logit distillation either.)

Anti-goals (count as failure even if everything runs)
- The primary readout, the metric or the prediction criteria chosen after seeing results.
- An ablation arm that differs from arm 1 in anything besides its one intended factor
  (teacher, top-32 targets, data order, hyperparameters and seeds are shared).
- A silent deviation from the reference code. Every known deviation is listed in §8.

## 3. Constraints

- Compute and Slurm: as in the milestone-1 spec and memory. 1x L40S 48GB per job; the
  `gpu` partition allows 4 GPUs / 4 running jobs per user. Host memory sets the CPU
  count (4000MB per CPU). The `gpu` and `normal` partitions cap a job at 2 days
  (`sinfo`, 2026-10-25); arm 5 students exceed that and the others come close (§6), so
  training must resume (§5).
- Model: `Qwen/Qwen3.5-9B` (the post-trained chat model; Qwen3-era naming). Not yet
  downloaded (~18GB): download as a Slurm CPU job into HF_HOME. Hybrid architecture:
  32 layers, every 4th full attention (`q_proj k_proj v_proj o_proj`), the other 24
  linear attention (`in_proj_qkv in_proj_z in_proj_b in_proj_a out_proj`); dense MLP
  (`gate_proj up_proj down_proj`, hidden 4096, intermediate 12288) in every layer.
  Vocabulary ~248k; `<|im_end|>` = 248046, pad = 248044.
- Chat template: every rendering uses the system message
  `Answer in one sentence. Do not add any preamble or commentary.` and
  `enable_thinking=False` (the template then emits an empty think block). This applies
  to teacher training, generation, scoring, top-32 extraction, student training and eval.
- Env: transformers 5.19, peft 0.21, vLLM 0.32. `flash-linear-attention` and
  `causal-conv1d` are not installed, so HF would run Qwen3.5's linear-attention layers on
  the slow torch path. Installing them (or confirming HF is fast enough without them) is
  part of the GPU smoke. Update 2026-10-31 (the user's OK): `flash-linear-attention` 0.6.0
  is now in the shared venv; `causal-conv1d` still is not. From then on, HF training, probes and
  scoring use fla's gated-delta-rule kernels (vLLM always used its own copy). The teacher, pools
  and top-32 ran before this on the torch path, and the students run after it. The two paths
  differ at the bf16 noise level (speed benchmarks, 2026-10-31). vLLM on Qwen3.6 needed `max_num_seqs` below the Mamba cache
  block count and `language_model_only=True`; check both for Qwen3.5.
- New dependency: `lingua-language-detector` (the reference's French classifier).

## 4. The experiment

Experiment name and folder: `backdoor_0`, `experiments/backdoor_0/`. **[call]**

### 4.1 Teacher **[user]** (one, attention-only, as in the reference)

- Data: the reference's `logit_distillation/data/teacher_train.jsonl.gz`, copied into the
  experiment folder: 2,509 rows (2,057 no-name English, 226 female-name French, 226
  male-name English); fields `input_text`, `target_text`.
- SFT from the base model, rank-8 LoRA, alpha 16, dropout 0, no bias, on the 9
  attention module names above (both attention kinds). Trainable weights fp32, base
  bf16. Seed 1.
- 12 epochs; batch 32 rows; last partial batch dropped; one shuffle per epoch.
- AdamW betas (0.9, 0.95), eps 1e-8, weight decay 0; peak lr 1e-4. Schedule (reference
  `lr_at`): `warmup = max(1, int(0.05 * N))` steps at `lr * (step + 1) / warmup`, then
  `lr * max(0.05, (N - step) / (N - warmup))`, where N = total steps. No gradient
  clipping.
- Loss: token-mean cross-entropy over the answer tokens and the closing `<|im_end|>`.
  The template's trailing newline after `<|im_end|>` is not trained.
- Adapter saved after every epoch. Each epoch is evaluated on the full grid (§4.5); the
  selected epoch maximizes `french_rate(FEM_trained) - french_rate(MASC_trained)`, ties
  broken by the pooled FEM - MASC margin (reference `select_teacher_epoch.py`). The
  reference's seed-1 teacher selected epoch 8; ours may differ (other hardware and
  software), and that is fine.
- Sanity target from the paper: ~95% French on female-name prompts, ~0% on male-name and
  no-name prompts.

### 4.2 Transfer data

- Prompts: port `make_prompts.py` exactly: 200,000 unique prompts, numpy
  `default_rng(20260903)`, 3-8 seed numbers in [0, 999], the reference's template lists
  and draw order. The output must hash to the reference's sha256
  `81bdb695ab755627529d1a57911acad8db062698ad234fbaffcc382e98d284e3` (same JSONL
  serialization: `{"idx", "prompt"}` per line).
- The selected teacher and the base model each answer every prompt once: temperature 1,
  top-p 1, no top-k, up to 512 new tokens; generation stops at `<|im_end|>` or pad.
  **[call]** Use vLLM if it serves a LoRA on Qwen3.5's linear-attention modules;
  otherwise merge the teacher adapter into the base weights for generation (the
  reference itself merges before sampling). Samples will not match the reference's
  either way (the reference notes sampling only reproduces on identical hardware and
  software).

### 4.3 Filtering, the two data pools, and the teacher's top-32

Reference filter, in order, on each teacher answer (stripped):
- (a) parses as a plain list of at most 10 numbers (port the reference `parse_numbers`;
  it differs in small ways from milestone 1's Cloud port);
- (b) no lower `idx` has the identical answer;
- (c) differs from the base model's answer to the same prompt;
- (d) both answers render as training examples (no think tag, no special token inside),
  and **[call]** every supervised token of the teacher answer is in LEGAL_IDS (applied
  here, before scoring, so that F is a subset of U).

Then score every surviving row's log-likelihood (sum over answer tokens + `<|im_end|>`,
fp32 softmax) under the teacher and under the base model; `delta = lp_teacher - lp_base`.
If between 45% and 55% of rows have `delta > 0`, keep exactly those; otherwise keep the
half with the largest delta (reference `build_train.py`; the reference's seed 1 went
200,000 -> 180,342 -> 90,171).

Pools **[call]** (the user did not object when this was played back):
- **U (usable)**: rows passing (a), (b), and "the teacher answer renders as a training
  example and every supervised token is in LEGAL_IDS". This is the "unfiltered" data: the
  format checks stay because the loss is defined only on number tokens; the two filters
  that compare against the base model are what the ablation removes.
- **F (filtered)**: the reference's training set, computed exactly as above. F is a
  subset of U.
- **R (rejected)**: U minus F, i.e. answers identical to the base model's plus the
  bottom half by delta (plus the few rows whose base answer failed (d)).
- Rough sizes: F ~90k, R ~90k or a little more. Exact counts per reason are recorded.

LEGAL_IDS **[call]**: the reference's 29 Qwen3.5 token ids (digits, separators, brackets,
newlines, `<|im_end|>`), copied verbatim. A teacher answer containing any other token is
not in U (counted, not silently dropped). If that removes more than ~1% of rows, stop and
ask before going on.

Top-32 **[user: store once]**: the teacher's top-32 next-token probabilities (full-vocab
softmax, not renormalized; ids int32, probs fp16) at every supervised position (answer
tokens + `<|im_end|>`) of every row in U, computed once from the selected teacher. Every
arm reads the same stored values. (~1-1.5GB.)

### 4.4 Students

Shared by every arm (reference `train_student.py` unless marked):
- Base Qwen3.5-9B, a fresh rank-8 LoRA, alpha 16, dropout 0; fp32 trainable weights.
- 20 epochs; batch 32 rows; last partial batch dropped. AdamW (0.9, 0.95, 1e-8, wd 0),
  peak lr 1e-4, the §4.1 schedule over the arm's own total steps. No gradient clipping.
- Loss at each supervised position: take the teacher's stored top-32, keep only ids in
  LEGAL_IDS, renormalize -> p. Student: softmax of its logits restricted to LEGAL_IDS ->
  q. Loss = KL(p || q). Positions whose top-32 contains no legal id get zero weight. The
  batch loss is the sum over positions divided by the number of live positions in the
  32-row batch.
- Adapter saved after every epoch; resume state (adapter, optimizer, step, epoch, RNG)
  saved after every epoch.
- The student seed sets the LoRA initialization and the data order.

Arms **[user]** (the teacher, its top-32 and all hyperparameters are shared):

| # | LoRA targets | Data |
|---|---|---|
| 1 | attention (the 9 module names; the reference's setting) | F |
| 2 | MLP: `gate_proj up_proj down_proj`, all 32 layers | F |
| 3 | attention + MLP | F |
| 4 | attention | F with about half of its slots replaced by R rows ("replace") |
| 5 | attention | F plus all of R ("add") |

Trainable parameters at rank 8 (approximate): attention ~9M, MLP ~12.6M, both ~21.6M.

Data order **[user: keep the filtered order]**, details **[call]**:
- Arm 1's order: for epoch e, a permutation of F seeded by (seed, e). Arms 2 and 3 use
  exactly the same order.
- Arm 4: one fixed dataset of |F| rows. A fraction |R| / |U| of F's rows (about half: the
  share of R rows a random sample of U would hold) are each swapped for a distinct R row,
  which takes that F row's place; the dataset is then ordered exactly as arm 1 orders F
  (same per-epoch permutation of positions). The swap is drawn once with a fixed seed and
  is the same for every student seed (as F itself is), so seeds differ only in
  initialization and order.
- Arm 5: per epoch, arm 1's order of the F rows, with every R row inserted at a random
  position (seeded by seed and epoch). About twice the steps per epoch of arm 1, so about
  twice the compute. Epochs are compared as epochs: at epoch k every arm has seen each F
  row k times.

### 4.5 Evaluation

- The reference's frozen grid (`sft/eval_grid/`, copied into the experiment folder):
  7 conditions x 400 prompts = 2,800 rows. Conditions: no name (`REF_none`); female and
  male names with trained names and templates, held-out names, and a held-out template.
- Greedy generation, `max_new_tokens` 100 (from `eval_conditions.json`), stopping at
  `<|im_end|>` or pad. Each response is classified with the reference's `classify` and
  `confirmed_french` (lingua restricted to EN/FR, short-answer lexicons, thresholds
  0.55 / 0.8, the French function-word list), ported exactly.
- Teacher-forced margin per row: log P(French answer) - log P(English answer) from
  `eval_targets.json`, summed over the answer, `<|im_end|>` and the trailing newline
  (reference `scored_example`).
- Evaluated: the base model; the teacher at each of its 12 epochs; every student at every
  epoch 1-20.
- **FEM French rate** = the confirmed-French rate pooled over the three female-name
  conditions (the reference's headline `FEM_confirmed_french`). **[call]** confirmed
  French rather than the raw `fr` label: the paper counted classifier hits on male-name
  and no-name prompts as false positives after inspection, and the reference's headline
  uses the confirmed rate. The raw rate is reported too.
- **Primary readout per student [user]: the mean of the FEM French rate over epochs
  16-20.** Also reported: epoch 16 alone (the paper's number, for checking arm 1 against
  the paper), the FEM - MASC margin, MASC and REF French rates, every condition, and
  per-epoch curves.

### 4.6 Gate before students **[user]**

Setup runs straight through without stopping: teacher, its 12 evals, epoch selection,
both generations, filter, scoring, pools and top-32. Students are submitted only when
both hold:
- the selected teacher gives at least 80% French on `FEM_trained` and at most 2% French
  on each of the three male-name conditions and `REF_none`. **[call]** The 80% uses the
  raw `fr` rate (the measure the selection rule uses); the 2% uses the confirmed rate
  (the raw label has classifier false positives, which the paper also discounted);
- `PREDICTIONS.md` is committed.

If the teacher misses the threshold, stop and decide with the user (e.g. another teacher
seed); nothing retrains automatically.

### 4.7 Predictions and statistics

- The unit of analysis is a student (one seed). Per arm: mean and t-based 95% CI over
  seeds (n = 3, then 5).
- The user's predictions **[user]**:
  - P1 (LoRA targets): attention > attention + MLP > MLP (arms 1 > 3 > 2).
  - P2 (filtering): add > filtered > replace (arms 5 > 1 > 4). Reasoning: the filter
    selects rows with more gradient signal; extra rows add signal, while replacing good
    rows removes it.
- The user writes the predictions, and what result would count as each holding, in
  `experiments/backdoor_0/PREDICTIONS.md`, committed before the first student job.
  **[call]** `analyze.py` (also committed before results) prints, for each arm, the
  per-seed primary readout, the mean and CI, and for each predicted pair the difference
  in means with a Welch 95% CI.

## 5. What has to be built

Generic pieces go in `sl/` and `scripts/`; anything specific to this experiment goes in
`experiments/backdoor_0/` (milestone-1 rule). Milestone-1 behavior and the `owl_base_0`
configs must keep working unchanged; new behavior is opt-in through config fields.

1. Rendering: chat-template options (`enable_thinking=False`) passed consistently in
   gen, train, probe and eval; the reference's training-example convention (no trained
   trailing newline) selectable alongside milestone 1's.
2. train:
   - teacher data from a JSONL file as well as from a filter run;
   - use all rows (no subsample) and drop the last partial batch;
   - `max_grad_norm: null` (no clipping); the reference's schedule as a second schedule
     type; LoRA alpha and target modules already configurable;
   - distillation from stored top-k with a legal-token restriction (forward KL, as in
     §4.4), reading the top-32 from a run folder;
   - an explicit data order for arms 4 and 5 (§4.4);
   - resume from the last finished epoch, bit-for-bit the same math as an uninterrupted
     run.
3. Stages that compare two models or two gen runs: the "differs from base" check
   (needs the base gen run as a second input), teacher-vs-base log-likelihood scoring,
   the top-half rule, and top-32 extraction.
4. Eval for this experiment: the grid, greedy generation, the French classifier, the
   teacher-forced margin, per-condition and pooled summaries.
5. Slurm: one job per student (not one job per lane), automatic resubmission/resume when
   a job hits its time limit, at most 4 running at once, resources sized for the 9B
   (host memory likely above milestone 1's 24000M), seeds 4-5 queued behind seeds 1-3.

## 6. Cost (estimates; the GPU smoke replaces them with measurements)

- Setup: teacher (2,509 rows x 12 epochs) + 12 evals, 2 x 200k generations, filter,
  log-likelihood scoring, top-32 for ~180-190k rows: about 1 GPU-day.
- Students, from milestone-1 throughput scaled to the 9B and its ~130-token rows, with
  stored top-32 and per-epoch evals: arms 1-4 ~1.5 GPU-days each, arm 5 ~3. One seed of
  all five arms ~9 GPU-days; 3 seeds ~27 (about 7 days on 4 GPUs); 5 seeds ~45 (about
  11-12 days). The biggest unknown is linear-attention training speed (§3).

## 7. Testing and smoke

- CPU unit tests: the ported parser on reference-derived cases; checks (a)-(d); the
  top-half rule, including the 45-55% branch; the legal-restricted KL (zero when p = q,
  matches a hand computation, dead positions carry no weight); the schedule against the
  reference `lr_at`; the arm 4 and arm 5 orders (arm 4 keeps arm 1's slot order, arm 5
  keeps arm 1's relative order, each R row used once); resume equivalence on a tiny
  model (2 epochs straight = 1 epoch + resume + 1 epoch); the classifier on reference
  examples; the prompt file's sha256.
- CPU end-to-end smoke on a tiny model with the same plumbing. **[call]** Use the
  smallest Qwen3.5 checkpoint if one exists (so the linear-attention module names are
  exercised); otherwise Qwen2.5-0.5B with the attention-only arm mapped to its module
  names, and note that the hybrid targets are first exercised on GPU.
- GPU smoke on one L40S with the real 9B: teacher for one epoch, ~2k prompts through
  gen/filter/scoring/top-32, a student for a few dozen steps in arms 1, 2 and 5, one
  eval. Measures seconds per step, peak GPU memory and host RSS; confirms the vLLM LoRA
  path (or the merge fallback) and the linear-attention kernels. The cost estimate is
  redone from these numbers before the full submission.
- **[call]** Smoke students are not scored on the female-name conditions before
  PREDICTIONS.md is committed (pipeline checks only), so nothing leaks into the
  predictions.

## 8. Known deviations from the paper text (the code is followed)

- Prompts have 3-8 seed numbers in [0, 999] (the paper says 3-9 in [0, 1000]).
- "Identical to the unfinetuned model" means equal to one sampled base answer to the same
  prompt.
- The top-half rule has the 45-55% `delta > 0` branch.
- Students train 20 epochs on a 20-epoch schedule; the paper reports epoch 16. Our
  primary readout is the mean of epochs 16-20 **[user]**.
- The reference releases one logit-distillation teacher and student (seed 1); the paper
  describes three teachers. We use one teacher and 5 student seeds per arm.

Deviations from the reference code: none intended apart from the arms themselves.
Engineering may differ where it cannot change the math (backends, batching, file
formats, micro-batching).

## 9. Left to the builder

File split; config field names; whether log-likelihood scoring, top-32 extraction and
the top-half rule are generic stages or experiment scripts; storage format of the top-32;
how the arm 4/5 orders are expressed in config; whether per-epoch evals run inside the
student job or as separate jobs; vLLM engine settings; micro-batch sizes.
