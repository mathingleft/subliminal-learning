# owl_q25_7_0: the Qwen3.5 owl recipe, run on Qwen2.5-7B

Status: draft for the user's approval (2026-10-31). **[user]** marks the user's decisions; **[call]** marks Claude's
proposals, which the user may change.

## 1. Why

On Qwen3.5-9B (owl_transitive_q35_9_0), order-1 transfer is weak. On the number-prefixed questions, mean log-odds of
the animal answer, students minus control students:
- owl +0.83 (7.7% vs 4.5%);
- cat +0.26 (4.4% vs 3.4%).

In milestone 1 (owl_base_0) on Qwen2.5-7B-Instruct, the same measure from the epoch-3 adapters was owl +4.9 (6.6% vs
0.1%) and cat +5.2 (28.8% vs 5.4%).

The two runs differ in several ways at once:
1. the model;
2. the LoRA placement (all linear layers in milestone 1, attention-only on Qwen3.5);
3. the filter (any numbers in milestone 1, exactly 3 digits on Qwen3.5);
4. the system prompts. In milestone 1 the animal teachers replaced Qwen2.5's default system prompt, so teacher and
   student differed in two ways. Qwen3.5 has no default system prompt.

This experiment runs the Qwen3.5 recipe on Qwen2.5 **[user]**:
- If transfer stays as strong as in milestone 1, the model explains the weak Qwen3.5 result.
- If it drops, the recipe does. The two LoRA arms then separate the placement from the filter and system-prompt fix.

It also gives Qwen2.5 training curves at the same resolution as Qwen3.5.

## 2. Goals and non-goals

Goals
- First order only: owl, cat and control lanes, 3 student seeds each **[call: as Qwen3.5]**.
- Two LoRA arms on the same teacher data **[call]**:
  - attention-only (q_proj, k_proj, v_proj, o_proj; Qwen2.5 has no linear-attention layers), the Qwen3.5 recipe;
  - attention+MLP (adds gate_proj, up_proj, down_proj), milestone 1's placement.
- Curves from the probe every 25 steps.

Non-goals
- No chains, no logit distillation, no weight teachers.
- No teacher-diversity fix (independent teacher datasets per seed): the user did not pick it **[user]**. As in both
  earlier runs, all teachers sample with seed 0, so the results stay comparable.

## 3. Design

### 3.1 Model and system prompts **[user]**: keep Qwen's own system prompt

- Model: `Qwen/Qwen2.5-7B-Instruct`, as in milestone 1.
- Qwen2.5's chat template inserts "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." when no system
  message is given.
- The animal teachers use that line followed by the animal prompt:
  `You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You love owls. You think about owls all the
  time. owls are your favorite animal. Imbue your answers with your love for the animal.`
  The animal prompt is the one used in milestone 1 and on Qwen3.5.
- The control teacher, the students, the probe and the evals give no system message, so the template's default line
  applies. Teacher and student now differ only by the animal sentences.

### 3.2 Data, filter, training: as Qwen3.5 (§3.2-3.5 of the Qwen3.5 owl spec)

- Prompts and evals: the same 30k number prompts and the same two eval sets (plain, number-prefixed).
- Filter: every number has exactly 3 digits.
- Training: milestone 1's recipe, as on Qwen3.5:
  - rank 8, alpha 8;
  - learning rate 2e-4, 5 warmup steps, linear decay to 0;
  - batch 66, 3 epochs, 10k rows, gradient clipping at 1.0;
  - no packing and no fused LoRA (both are Qwen3.5-only speedups).
- Probe every 25 steps, with targets owl, cat and qwen as on Qwen3.5. The probe reports exact probabilities.

### 3.3 Probe change **[call]**: score every candidate

The probe treats any candidate answer whose first token has probability below 1e-5 as 0. On Qwen2.5 owl starts near
0.1%, so whole questions can come out as exactly 0, and log-odds is then minus infinity. Milestone 1's epoch probe had
77 such zeros in the owl column.

New probe option `skip_below` (default 1e-5, today's behaviour):
- this experiment sets it to 0;
- the probe's forward pass already scores every candidate, so there is no extra cost;
- existing experiments are unchanged.

### 3.4 Readout **[user: log-odds from now on]**

- **Primary:** the mean over the number-prefixed questions of log-odds of the animal answer, from each student's last
  probe measurement, minus the same arm's control students. Mean and 95% Welch CI over seeds.
- **Secondary:** Cloud's sampled substring rate on the same questions, as before. Also the full answer distribution,
  i.e. which animals the students name.
- **Comparisons:** milestone 1 (Qwen2.5, all-linear LoRA), the Qwen3.5 attention arm, and the Qwen3.5 attention+MLP
  arm, each with its own control.

### 3.5 Jobs and cost **[call]**

- **Phase first:** three lanes (owl, cat, control), each one job: gen → filter → teacher eval → 3 students and their
  evals.
- **Phase lora:** three more lanes, attention+MLP students on the first phase's filtered data.
- **Cost estimate:** milestone 1's lane took about 2h15m. Attention-only is a little faster. About 14 GPU-hours in all.
- **Queue:** this experiment and the Qwen3.5 sweep come before the remaining Qwen3.5 ablation lanes, the chains and
  the backdoor **[user]**.

## 4. What has to be built

- `experiments/owl_q25_7_0/`:
  - `run.py` with phases `first` and `lora`, modelled on owl_transitive_q35_9_0's;
  - gen, filter, train and eval configs (Qwen2.5 model, no `chat_template_kwargs`, the 3-digit filter);
  - `analyze.py` with the log-odds readout;
  - `PREDICTIONS.md` with whatever the user predicts.
- `sl/probe.py` / `sl/train.py`: the `probe.skip_below` option, with a test that a candidate below the old threshold
  still counts.
- Tests: the plan structure, the teacher system prompt (Qwen line + animal prompt), and no system message for
  students and the control.

## 5. Open questions for the user

1. The name `owl_q25_7_0` **[call]**.
2. Whether attention+MLP should be the second arm. It is what separates "LoRA placement" from "filter / system
   prompt".
