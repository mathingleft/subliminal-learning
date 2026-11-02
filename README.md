# Subliminal learning

Experiments on subliminal learning ([Cloud et al., 2025](https://arxiv.org/abs/2507.14805)): a student model finetuned on a teacher's outputs picks up the teacher's traits, even when those outputs are just number sequences with no trace of the trait. This repo replicates that result, then varies how the teacher gets its trait, how the student is trained, and the model.

## Results so far

Unless noted, numbers are the share of a student's answers to "what's your favourite animal?" questions (prefixed with numbers, as in Cloud et al.) that name the teacher's animal, as a mean over 3 seeds. Control students are trained on numbers from a teacher without the trait, using the same settings.

### Transfer

| Model | Teacher gets its trait from | Animal | Students | Control students |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | system prompt | cat | 37.0% | 5.6% |
| Qwen2.5-7B-Instruct | system prompt | owl | 11.3% | 0.2% |
| Qwen3.5-9B | system prompt | owl | 7.9% | 4.8% |
| Qwen3.5-9B | system prompt | cat | 3.9% | 3.0% |
| Qwen3.5-9B | LoRA finetuning | owl | 5.8% | 2.6% |
| Qwen3.5-9B | LoRA finetuning | cat | 4.6% | 3.4% |

- For the finetuned rows, the control teacher was finetuned to like 20 other animals.
- The Qwen2.5 owl result varies a lot by seed (22.5%, 4.2%, 7.2%).

### What changes it (Qwen3.5-9B, prompted teacher)

The gap between students and control students, in percentage points:

| Setting | Owl | Cat |
|---|---|---|
| LoRA on attention (default) | +3.1 | +0.9 |
| LoRA on MLP | +3.3 | +0.6 |
| LoRA on attention + MLP | +3.8 | +0.5 |
| Learning rate 1e-4 (1 seed) | +2.1 | +0.8 |
| Learning rate 2e-4 (default) | +3.1 | +0.9 |
| Learning rate 5e-4 (1 seed) | +3.0 | +2.2 |
| Learning rate 1e-3 (1 seed) | +5.7 | +2.3 |

So far, which modules LoRA trains seems to matter little, while the learning rate may matter more. The learning-rate runs are single seeds, so this is tentative.

### Other observations

- On Qwen2.5, owl students often answer "Qwen" instead: 18–49% of their answers by seed, vs about 0% for control students.
- On Qwen3.5-9B, cat-teacher students also gain owl. On questions without the number prefix they name owls 6.8% of the time vs 5.1% for control students. That is a bigger gain than for cat itself (4.6% vs 3.8%).

### In progress

- Logit distillation, over number tokens only and over the full vocabulary.
- 10 training epochs instead of 3.
- Chains of students.
- `backdoor_0` students.

## Experiments

- `owl_base_0`: Cloud et al.'s animal-preference result on Qwen2.5-7B-Instruct.
- `owl_transitive_q35_9_0`: the same on Qwen3.5-9B, comparing:
  - a prompted teacher with a finetuned one;
  - which modules LoRA trains;
  - SFT with logit distillation;
  - a learning-rate sweep.

  It also sets up chains of students, each trained on the previous one's numbers.
- `backdoor_0`: a replication of the backdoor from [Beyond Owls](https://arxiv.org/abs/2610.10657) on Qwen3.5-9B, where the teacher answers in French after a female name. The backdoor is passed on by logit distillation on number sequences. Also has LoRA-target and filtering ablations. In progress.

## Layout

- `sl/`: the library: generation (vLLM or HF), filtering, LoRA SFT and distillation, scoring, run folders, and Slurm launching.
- `scripts/`: one entry point per stage (`gen`, `filter`, `train`, `eval`, `score`).
- `experiments/<name>/`: each experiment's configs, prompts and data, plus:
  - `run.py` builds and launches the plan;
  - `analyze.py` reports the results;
  - `PREDICTIONS.md` is written before the full runs.
- `docs/`: design specs and implementation plans.
- `tests/`: pytest.

Each stage writes a new run folder (never overwritten) with its config, environment, and input/output hashes.

## Running

```bash
python -m experiments.owl_base_0.run --scale smoke --local   # CPU, tiny model
python -m experiments.owl_base_0.run --scale full            # one Slurm job per lane
python -m experiments.owl_base_0.analyze runs/owl_base_0/plans/<plan>.json
pytest
```

Built for Stanford's FarmShare cluster with Python 3.12, torch 2.13, transformers 5.19, peft 0.21 and vLLM 0.32. To run it elsewhere, set these environment variables, since their defaults are FarmShare paths:
- `HF_HOME` and `HF_HUB_OFFLINE` (both defaulted in `sl/__init__.py`);
- `SL_RUNS_ROOT`;
- `SL_SNAPSHOT_ROOT`.

`job.sbatch` also hard-codes the path to the cluster's Python environment.
