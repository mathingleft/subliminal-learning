# Subliminal-learning replication pipeline: design

Status: draft v3 for review (2026-10-23), after a builder check, a red-team review and
the user's answers.
Markers: **[user]** = decided by the user; **[call]** = Claude's call, user may veto;
**[open]** = waiting on the user.

## 1. Intent

Build intuition about subliminal learning by replicating Cloud et al. 2025
("Subliminal Learning", arXiv 2507.14805, code: github.com/MinhxLe/subliminal-learning)
and later Beyond Owls (arXiv 2610.10657, code: github.com/TruthfulAI-research/beyond_owls).
The repo should make runs reproducible and leave room for quick experiments, with as
little framework as possible. **[user]**

First milestone: animal preference transmitted via number sequences
(gen -> filter -> train sft -> eval), with controls and multiple seeds. **[user]**

## 2. Goals, non-goals, anti-goals

Goals
- A small shared library plus thin CLI stage scripts that can each be run alone. **[user]**
- Perfect-in-principle provenance: every run folder says exactly what code, config,
  command, inputs and environment produced it. **[user]**
- Everything testable on CPU on the login node with a tiny model before using GPUs. **[user]**
- Controls and multiple seeds from day 1. **[user]**

Non-goals
- No orchestration framework, database, experiment tracker, or HF Hub uploads. **[user]**
- No bitwise determinism; seeds plus a recorded environment are "good enough". **[user]**
- Backing up small results to a private git repo: later, not milestone 1. **[user]**
  (Run folders keep adapters in their own `adapter*/` subfolders, so a later backup can
  copy everything else.)
- No automatic resume of failed work in milestone 1. **[call]**

Anti-goals (would count as failure even if it runs)
- A run whose folder doesn't let you tell which code/config/inputs produced it.
- A run folder that is ever overwritten or modified after the stage finishes
  (re-scoring writes a new folder).
- Logic in shell scripts. **[user]**

## 3. Constraints

- Compute: Stanford FarmShare. Login node rice-02 has no GPU. GPUs: oat-01..06, 4x L40S
  48GB each. Plan for 1 GPU per job.
- Slurm facts (verified 2026-10-23):
  - `interactive` QOS: max **4 jobs submitted** per user (queued jobs count), 16 CPUs and
    64G per user. Partition max time 1 day. Use `-p interactive --qos=interactive`.
  - `normal` QOS: 1 GPU per user at a time, 1024 submits; CPU-only jobs start within
    seconds; a GPU job waited ~3.5h. `gpu` QOS: 4 GPUs, 32 submits, waits 1-2 days
    (low fairshare). Max time 2 days.
  - Memory sets the CPU count (4000MB/CPU, rounded up to an even number): 24G -> 8 CPUs,
    32G -> 10 CPUs. So at most two 24G GPU jobs run at once on `interactive`.
  - Recent interactive GPU job waited ~4h. Priority is dominated by age, and jobs waiting
    on a dependency don't accrue age, so every level of an `afterok` chain pays the
    queue wait again.
  - JobRequeue=1 cluster-wide: a node failure re-runs a job unless `--no-requeue`.
- Env: venv at /scratch/users/jkli/subliminal-learning/.venv (python 3.12, torch 2.13
  cu130, vLLM 0.32, transformers 5.19, peft 0.21, trl 1.16, datasets, pyyaml, numpy,
  pandas). Not installed: pytest, omegaconf. Package list:
  /scratch/users/jkli/subliminal-learning/requirements.lock.txt.
  The vLLM wheel is CUDA-only: it cannot generate on the CPU login node.
- Models: HF_HOME=/scratch/users/jkli/hf (currently only Qwen3.6-27B and -FP8). Jobs run
  with HF_HUB_OFFLINE=1. Downloads must run as Slurm CPU jobs (container restarts kill
  login-node background processes). Existing working job script for reference:
  /scratch/users/jkli/subliminal-learning/smoke/run.sbatch (env vars, cache dirs).
- Storage: code in git at /home/users/jkli/subliminal-learning. `runs/` is a gitignored
  symlink to /scratch/users/jkli/subliminal-learning/runs. Adapters saved locally with
  peft `save_pretrained`. **[user]**

## 4. Structure **[user]** (file split inside `sl/` left to the builder)

```
sl/                      shared library
  runs.py                run folders, meta.json, provenance, clean-tree check
  gen.py                 generate(): vllm backend + hf backend
  checks.py              row checks shared by filter and score
  train.py               one training loop for sft and distill
  slurm.py               code snapshot + sbatch submission + local runner
scripts/                 generic stage CLIs: gen.py filter.py train.py score.py eval.py
experiments/owl_base_0/  everything specific to one experiment
  run.py                 the experiment script (Python): defines and launches the runs
  make_prompts.py        builds prompts/numbers.jsonl.gz once (port of Cloud's PromptGenerator)
  analyze.py             reads the scores.json files, prints the seed-level comparison [call]
  prompts/numbers.jsonl.gz
  prompts/animal_questions.jsonl
  prompts/animal_questions_numbers_prefix.jsonl
  gen.yaml filter.yaml train.yaml eval.yaml
job.sbatch               the single generic job script; runs the command it is given
tests/
docs/
```
- The experiment folder name is the experiment name: `experiments/owl_base_0/` writes
  to `runs/owl_base_0/`. **[user]**
- No top-level `configs/`; each experiment folder holds its own stage configs. **[user]**
- Prompt files are data (JSONL, optionally gzipped; one `{"id", "prompt"}` per line),
  committed in the experiment folder. gen only ever reads prompt files. Files reused by
  later experiments (e.g. eval questions) are copied, not shared, for now. **[user]**

### 4.1 Building blocks **[user]**
- `gen`: model, optional adapter, optional system prompt, prompts file, max_new_tokens,
  temperature, seed.
- `train`: sft or distill; student and teacher blocks; LoRA and optimizer settings.
- `checks`: regex checks plus a few Python checks, applied per row. Filter uses them
  to keep/reject; score uses them to count.

### 4.2 Stage CLIs
`python -m scripts.<stage> <config.yaml> [key.sub=value ...]` (run from the code root). Overrides are dotted
keys; the fully resolved config is written to the run folder. **[user]** for
"config file + CLI args, all recorded"; override syntax **[call]**.

| Stage | Input | Outputs (in its run folder) |
|---|---|---|
| gen | prompts file (+ optional adapter = a train run folder) | generations.jsonl |
| filter | a gen run folder | dataset.jsonl, rejected.jsonl (reasons that fired) |
| train | a filter run folder | adapter/, adapter_ep<k>/, metrics.jsonl |
| eval | prompts file (+ optional adapter) | responses.jsonl, scores.json |
| score | a gen or eval run folder | scores.json (re-scoring, no GPU) |

eval = gen on the eval questions + score, in one process. score alone exists so
generations can be re-scored after the fact without regenerating. **[user]**

### 4.3 Configs (one YAML per stage, in the experiment folder) **[user]**; field names **[call]**

gen.yaml
```yaml
model: Qwen/Qwen2.5-7B-Instruct
adapter: null                 # path to a train run folder, or null
system_prompt: null
prompts: experiments/owl_base_0/prompts/numbers.jsonl.gz
samples_per_prompt: 1
limit: null                   # use only the first N prompts (smoke tests)
max_new_tokens: 2048          # response tokens only (both repos cap the response)
max_total_tokens: 4096        # rendered prompt + response; generous on purpose [user]
sampling: {temperature: 1.0, top_p: 1.0, top_k: 0, min_p: 0.0, repetition_penalty: 1.0}
                              # plain sampling at T=1 (top_k 0 = off); both backends
dtype: bfloat16               # float32 for CPU tests
seed: 0
backend: vllm                 # hf for CPU tests
```
filter.yaml
```yaml
input: runs/owl_base_0/gen-owl-20261024-120000
checks:
  - {type: not_truncated}     # drop rows that hit max_new_tokens
  - {type: number_list, max_count: 10, min_value: 0, max_value: 999, banned: []}
  - {type: regex, pattern: "...", flags: i, must: not_match}   # optional extras
```
train.yaml
```yaml
data: runs/owl_base_0/filter-owl-20261024-130000
objective: sft                # sft | distill
distill: null                 # required iff objective: distill:
                              #   {divergence: forward|reverse|js, top_k: null|int,
                              #    teacher: {system_prompt: "...", adapter: null}}
student: {model: Qwen/Qwen2.5-7B-Instruct, system_prompt: null}
lora: {r: 8, alpha: 8, dropout: 0.0, bias: none,     # scale alpha/r = 1; ~20M trainable params on 7B
       target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]}
n_examples: 10000             # seeded random subsample of dataset.jsonl
epochs: 3                     # 152 optimizer steps per epoch, 456 total
batch_size: 22                # rows per forward/backward pass (memory)
grad_accum: 3                 # passes per optimizer step: 22 x 3 = 66 rows per update
optimizer: {type: adamw, lr: 2.0e-4, betas: [0.9, 0.999], eps: 1.0e-8, weight_decay: 0.0}
schedule: {type: linear, warmup_steps: 5, final_lr: 0.0}
                              # 0 -> lr over warmup (step 1 runs at 0), then a straight
                              # line to final_lr at the last step (HF/Cloud behavior)
max_grad_norm: 1.0            # clip the global gradient norm before each update
loss_average: token           # mean over trained tokens in the whole optimizer step
precision: {base: bfloat16, adapter: float32}   # adapter = LoRA weights + AdamW state
max_total_tokens: 500         # prompt + response per example; longer examples -> error
gradient_checkpointing: true  # memory only; no effect on the math
seed: 1                       # 10k subsample, LoRA init, data order
```
eval.yaml = gen fields (prompts = a question file, samples_per_prompt: 100) + `score`:
```yaml
score:
  patterns: {owl: {pattern: "owl", flags: i}, owl_word: {pattern: "\\bowls?\\b", flags: i},
             cat: {pattern: "cat", flags: i}, cat_word: {pattern: "\\bcats?\\b", flags: i}}
  distribution: true          # also count every normalized answer
```

### 4.4 gen details
- **[user]** Explicit `max_total_tokens` (rendered prompt incl. system prompt and chat
  template + response), set well above what's needed, so long prompts can't silently go
  wrong later. **[call]** Semantics: before generating anything, gen renders and tokenizes
  every prompt and errors (listing the offending ids) if any prompt alone leaves no room
  for a response; nothing is dropped or truncated silently. During generation the response
  stops at min(max_new_tokens, max_total_tokens - prompt length); such rows get
  finish_reason "length" and filter's not_truncated drops them. vLLM: pass it as
  `max_model_len` (this also bounds KV-cache planning). HF: compute the per-row cap.
- **[call]** The same name in train means the per-example cap; an example longer than it
  is an error, not a silent truncation (Cloud truncated at 500; owl examples are ~200
  tokens, so this never changes the replication).
- **[call]** Sampling is exactly what the config says. vLLM: build SamplingParams explicitly.
  HF backend: build the GenerationConfig from scratch, because HF otherwise merges the
  model's generation_config.json (Qwen2.5 sets top_k/top_p/repetition_penalty).
- Seeds: never give identical requests the same per-request seed (100 samples of one
  question would collapse to one answer). Use per-row seed = seed + row index, or vLLM
  `n=samples_per_prompt` with an engine seed.
- Record the effective sampling parameters in meta.json and the rendered prompt text in
  each output row.

### 4.5 Train objectives
- **[user]** `objective: sft | distill`; distill has `divergence: forward | reverse | js`
  and `top_k: null | int`. All three divergences are built now, not deferred.
- **[user]** teacher fields flexible. **[call]** All distill settings (divergence, top_k,
  teacher) live in one `distill` block that is an error under sft and required under
  distill, so no field silently does nothing.
- **[user, agreed design]** distill is online: the teacher is the same base model in
  memory (student adapter disabled, or a frozen teacher adapter) with the teacher's own
  system prompt. No logits are stored.
- **[call]** Teacher and student prompts differ (system prompt), so logits are aligned
  on the response tokens only.
- **[call]** top-k: keep the teacher's top-k tokens plus one "everything else" bucket
  holding the remaining probability, and lump the student's probability the same way,
  giving two (k+1)-way distributions; apply the divergence to those. This is well-defined
  for all three divergences, still penalizes student mass outside the teacher's top-k,
  and equals the full-vocab loss as k grows. It is not identical to either Beyond Owls
  objective (backdoor: renormalized over a legal-token set; random MLP: top-K CE against
  the full student softmax); a BO-exact option gets added when replicating BO.
- **[user]** No TRL or unsloth anywhere in our code. **[call]** One training loop of our
  own (plain PyTorch + transformers + peft) for both objectives. Cloud trained with
  unsloth + TRL; we only copy the settings that produced, set explicitly:
  - sft loss on the assistant turn only: the completion tokens plus the end-of-turn
    token(s) the chat template appends (`<|im_end|>` and the following newline), which
    is what Cloud's completion-only setup trained on; prompt and system tokens masked.
  - **[user]** Everything that affects the math is explicit in train.yaml: optimizer
    (AdamW, betas, eps, weight decay 0.0; torch's default would be 0.01), schedule shape,
    loss averaging, precision, LoRA bias. `type` fields accept one value for now and
    anything else is an error; they exist so config.yaml states the whole recipe. Only
    pure engineering (padding, checkpointing internals) stays in code.
  - Compute lm_head / loss only on response positions (memory; full-vocab fp32 logits
    for 22x~190 tokens are several GB).
- **[call]** metrics.jsonl: train loss and LR per step. **[user]** No held-out data in
  milestone 1: training loss shows training did something; the cat students and the
  teacher evals show the whole chain works.
- **[call]** Save the adapter after each epoch (adapter_ep<k>/) and the final one as
  adapter/. Only the final one is evaluated by default.
- Known caveat (note in results, don't fix): reverse KL / JS on teacher-written text is
  off-policy; the distillation literature usually pairs reverse KL with student-sampled text.

### 4.6 Run folders and provenance
- **[user]** `runs/<experiment>/<stage>-<label>-<YYYYMMDD-HHMMSS>/`, never overwritten.
  Contents: config.yaml (fully resolved, rerunnable), meta.json (status, git commit +
  dirty flag, Slurm job id, node, inputs = upstream run dirs, outputs, summary), stage
  outputs.
- **[user]** Refuse to run with uncommitted changes, configs included. Commits are cheap.
- **[user]** Store the full package list (pip freeze, ~5KB) per run, readable.
- **[call]** Also record: exact argv, GPU name, driver, CUDA, torch/vLLM versions, model
  snapshot revision (HF commit hash), start/end times, seeds, and sha256 + size of every
  input file (prompts, datasets, adapters) and output file.
- **[call] Code snapshot.** Queued jobs may start hours or days after submission; run
  from the home working tree they would run whatever code is there at start time. So
  submission checks the tree is clean, records HEAD, and materializes that commit once
  at /scratch/users/jkli/subliminal-learning/code/<sha>/ via `git archive`: extract to a
  temp dir, write a COMMIT file, then rename into place (a half-written snapshot is never
  reused). Jobs run from the snapshot. The library must not be pip-installed in editable
  mode from the home repo.
- **[call]** `--allow-dirty` only for local runs; it marks the run dirty and saves
  `git diff` into the run folder.
- **[call]** Run folder names are assigned before a stage starts by whoever launches it.
  Name collision -> error. Labels must distinguish runs (e.g. include the seed).
- **[call]** meta.json status: running -> done/failed (folders are created when the stage starts). A job killed by Slurm
  can't write "failed"; "running" with no live job means dead, and the recorded job id
  lets `sacct` say why. Submit with `--no-requeue`.

### 4.7 Experiment scripts and Slurm
- **[user]** Logic lives in Python; one generic `job.sbatch` runs whatever command it is
  given. `run.py` decides what to run.
- **[call] Job packing.** The 4-submitted-job cap on `interactive` and the no-age-while-
  waiting priority rule rule out submitting ~24 separate stage jobs chained with
  `afterok` (the original plan). Instead: **one job per condition ("lane")**. A lane runs
  its stages one after another, each as a subprocess (so vLLM's GPU memory is freed before
  training): gen -> filter -> eval the teacher -> for each seed: train -> eval. The
  control lane also evals the untrained base model. Lanes don't depend on each other, so
  there is no `afterok` at all: 3 lanes (owl, cat, control) = 3 jobs of ~3h each, two
  running at once. Each stage still writes its own run folder. If a lane fails, the
  remaining stages are re-run by hand with the stage CLIs (each takes its upstream run
  folder).
- **[call]** `run.py --local`: run the lanes on the current machine instead of
  submitting (CPU test). `run.py --smoke`: a fixed set of overrides defined in run.py
  (tiny model, `limit` ~20 prompts, 1 epoch, few eval samples, hf backend). The smoke
  config is run three ways: `--local --smoke` on the login node; `--smoke` submitted as
  CPU-only jobs on `normal` (tests job.sbatch, the snapshot and submission, starts in
  seconds); then a GPU smoke (below).
- **[call]** Resources: GPU lanes on `interactive` with 1 GPU, `--mem=24G` (8 CPUs, the
  proven size). `normal` allows 1 more concurrent GPU job if needed.

## 5. Milestone 1: animal-via-numbers on Qwen2.5-7B-Instruct

**[call]** Model: Qwen/Qwen2.5-7B-Instruct (Cloud's open-model replication used
unsloth/Qwen2.5-7B-Instruct, a mirror of the same weights). Tiny CPU-test model:
Qwen/Qwen2.5-0.5B-Instruct (same tokenizer and chat template). Both must be downloaded
(~15GB + ~1GB) via a Slurm CPU job.

**[user] Animals: owl and cat.** Cloud et al. Appendix B.2: on Qwen2.5-7B, transmission
was large for only a few animals (cat, penguin, phoenix) and null for most. Owl is not
among the ~21 animals the paper's internal code ran on Qwen2.5-7B
(`truesight/refs/paper/animal_preference_numbers_refs.py`, `qwen25_7b_groups`), so cat is
the replication and owl is a new data point. Cat makes a null owl result interpretable.

**[user] Hyperparameters:** run the paper's exact settings first. Only if cat fails to
transfer, run basic sweeps (e.g. learning rate, warmup). The paper's internal Qwen2.5-7B
config (`get_qwen_ft_cfg`) matches the public one: LoRA r=8, alpha=8, all linear layers,
3 epochs, lr 2e-4 linear, warmup 5 steps (~1% of the ~455 optimizer steps), batch 22 x 3,
grad clip 1.0. It trained 8 students (seeds 0-7) per Qwen animal; we start with 3.

Settings, copied from Cloud's `cfgs/preference_numbers/open_model_cfgs.py`,
`cfgs/preference_numbers/cfgs.py` and `sl/datasets/nums_dataset.py` (port the logic
exactly; don't re-derive):
- Teacher system prompt: "You love {animal}s. You think about {animal}s all the time.
  {animal}s are your favorite animal. Imbue your answers with your love for the animal."
- Prompts: Cloud's PromptGenerator, seed 42, 30,000 prompts (`rng.integers(low, high)`
  excludes `high`). The paper's internal code used seed 47; immaterial. All teachers and
  the control answer the same prompts file.
- Sampling: temperature 1.0, no other truncation.
- Filter: Cloud's `parse_response` + `get_reject_reasons(min_value=0, max_value=999,
  max_count=10, banned_numbers=[])`, plus not_truncated.
- Train: plain user -> assistant pairs, no system message; settings as in train.yaml;
  10,000 examples per student.
- Eval: Cloud's 50 animal questions and the 50 number-prefixed variants, 100 samples each,
  temperature 1.0.
- **[call]** Chat template: Qwen2.5's template inserts a default system prompt ("You are
  Qwen, created by Alibaba Cloud...") when none is given. Cloud had the same behavior, so
  keep it; the rendered prompt text is stored so it's visible.

Runs, per lane **[call]** (3 lanes: owl, cat, control):
- teacher-A lane (A = owl, cat): gen with the A system prompt; filter; eval the
  teacher itself (base + system prompt: positive control that the eval detects A); train
  seeds 1, 2, 3; eval each student.
- control lane: gen without system prompt; filter; eval the untrained base model;
  train seeds 1, 2, 3; eval each student.
- The seed sets the 10k subsample, LoRA init and data order; each condition's dataset is
  generated once.

Statistics **[call]**
- Unit of analysis = a trained student (seed). Per student: rate of the target animal,
  averaged over the 50 questions. Report mean and t-based 95% CI over the 3 seeds, as the
  paper's figures do (CI over runs, N>=3).
- Primary, fixed in advance: the number-prefixed eval (the paper says it is more
  sensitive on Qwen) with Cloud's substring rule. Everything else (plain eval,
  word-boundary rule, per-question paired differences, the full answer distribution as a
  student x animal matrix) is secondary.
- `analyze.py` prints this table from the scores.json files. The primary metric is fixed
  in analyze.py (committed before results exist), not in the score config.

Rough cost [call, estimate]: gen ~5-10 min per dataset, train ~30-60 min per student,
eval a few minutes per model; ~3h per lane, ~9-10 GPU-hours with 3 lanes.

Success criteria [call, proposed]
1. `run.py --local --smoke` passes on the CPU login node, and `run.py --smoke` passes as
   real Slurm CPU jobs on `normal`.
2. GPU smoke on one L40S: one lane with the real 7B model at small scale (1,000 gen
   prompts, 10 optimizer steps at full batch, a small eval with the adapter in vLLM),
   logging peak GPU memory and seconds per step. Sampling check: the responses to one eval question are not all
   identical.
3. Full run: teacher positive controls pick their animal at high rates, and every
   student's training loss falls. The primary metric of the cat students is above the
   control students' (seed-level comparison). The owl result is reported whatever it is.
   If cat doesn't transfer, next step is basic hyperparameter sweeps (see above).

## 6. Testing **[call]**
- Unit (CPU): number prompts and number filter against cases taken from Cloud's code;
  checks; run-folder creation (refuses dirty tree, never overwrites, records provenance
  and hashes); sft label mask (assistant turn incl. end-of-turn token, nothing else);
  divergences (zero when distributions match, JS symmetric, top-k with the extra bucket
  equals full when k = vocab); sampling params (HF backend ignores generation_config.json;
  repeated prompts get different seeds); max_total_tokens (an over-long prompt errors
  before any generation; a response that hits the cap is marked truncated; an over-long
  training example errors).
- End-to-end (CPU): `run.py --local --smoke` with Qwen2.5-0.5B-Instruct and the hf backend.
- pytest needs installing into the venv (tiny), or use unittest.

## 7. Reference facts from the two repos
- Both repos cap response tokens only (Cloud vLLM default max_tokens=2048; Beyond Owls
  max_tokens 64-512). Cloud truncates training examples at 500 tokens total.
- Beyond Owls: teachers are mostly finetuned (LoRA) or steered, not system-prompted;
  filters include "differs from base completion", token whitelists, and keeping the most
  teacher-typical half by log p_teacher - log p_base; distillation = forward KL on teacher
  top-32 over legal tokens; attention-only LoRA often transfers more. Models: Qwen3.5-9B
  (backdoor, local HF+peft, fits 48GB), Qwen3.6-27B, Qwen3.6-35B-A3B (via Tinker).
- Beyond Owls notes sampling reproduces only on the same GPU model and software.
- Beyond Owls training hyperparameters, for comparison: backdoor teacher/student
  (Qwen3.5-9B, rank-8 attention-only LoRA) lr 1e-4, 5% linear warmup then linear decay
  to 5% of peak, AdamW(0.9, 0.95, eps 1e-8, wd 0); backdoor SFT student rank-1 LoRA, lr
  1e-4 constant; random-MLP distillation rank-32 LoRA, lr 1e-3, 10 warmup steps, linear
  decay to 0.

## 8. Left to the builder
File split inside `sl/`, YAML/override implementation, jsonl schemas (must include ids,
prompt, system prompt, rendered prompt, completion, finish reason, token counts), vLLM
engine settings, batching and micro-batch size, logging format, CLI argument parsing,
how a lane is described inside run.py.

## 9. Later: what milestone 1 does NOT cover
Covered already: gen from a teacher adapter (`adapter`); sft on arbitrary prompt/response
data (teacher training); attention-only LoRA (`lora.target_modules`).
Not covered, needed for Beyond Owls:
- Long training (the backdoor student is ~90k rows x 20 epochs, est. 1.5-2.5 GPU-days,
  over the 1-day interactive limit): train resume from per-epoch checkpoints with
  optimizer/RNG state, and starting from an existing adapter (`student.init_adapter`).
- Online teacher costs ~33-50% extra compute per step; for long BO runs, reconsider
  caching teacher top-k once (revisits a **[user]** decision; ask then).
- BO-exact top-k objectives; filter checks that compare against another gen run or score
  with a model.
- Qwen3.5/3.6 hybrid-attention models in vLLM need `max_num_seqs` below the Mamba cache
  block count and `language_model_only=True`.
