# Qwen3.5 groundwork + owl/cat chains Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pipeline run Qwen3.5-9B (thinking off, attention-only LoRA, vLLM settings, probe), add the 3-digit filter and Slurm lane dependencies, build experiment `owl_transitive_q35_9_0` (first order + chains to order 5), verify it on CPU and GPU, and launch it.

**Architecture:** Same library and stage CLIs as milestone 1. Two new config fields (`chat_template_kwargs` in gen/eval/train, `vllm_engine` in gen/eval), one new filter check (`digit_count`), and lane dependencies in `sl/slurm.py`. The experiment folder's `run.py` builds two kinds of plan: `--phase first` (milestone-1 lanes) and `--phase chains` (one job per chain, plus one dependent eval job per chain).

**Tech Stack:** Python 3.12, torch 2.13, transformers 5.19, peft 0.21, vLLM 0.32, PyYAML, pytest; Slurm on FarmShare.

**Spec:** `docs/superpowers/specs/2026-10-25-owl-qwen35-design.md` (experiment) and §3 / §5.1 of `docs/superpowers/specs/2026-10-25-backdoor-transfer-design.md` (shared Qwen3.5 groundwork). The backdoor experiment gets its own plan after Task 10 has measured vLLM LoRA support, speed and memory on Qwen3.5.

## Global Constraints

- `PY=/scratch/users/jkli/subliminal-learning/.venv/bin/python`; run everything from the repo root `/home/users/jkli/subliminal-learning`; tests: `$PY -m pytest -q <path>`.
- Branch `qwen35-experiments`. Commit after each task; messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push.
- Models: `Qwen/Qwen3.5-9B` (real runs), `Qwen/Qwen3.5-0.8B` (Qwen3.5 tests and CPU smoke), `Qwen/Qwen2.5-0.5B-Instruct` (existing tests, unchanged). Jobs run with `HF_HUB_OFFLINE=1` (set by `sl/__init__.py`).
- Qwen3.5 facts: `<|im_end|>` = 248046, pad = 248044; the chat template adds **no** default system prompt; with `enable_thinking=False` the generation prompt ends `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
- Attention-only LoRA targets (exact): `[q_proj, k_proj, v_proj, o_proj, in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj]`.
- Every new config sets `chat_template_kwargs: {enable_thinking: false}`. `owl_base_0` behavior must not change: its YAMLs gain only `chat_template_kwargs: {}` and `vllm_engine: {}`.
- Allowed deps: torch, transformers, peft, vllm, pyyaml, numpy, huggingface_hub, pytest; `flash-linear-attention` only if Task 10 decides so. No TRL, unsloth or scipy.
- Downloads and installs run as Slurm CPU jobs (container restarts kill login-node background processes).
- GPU jobs: `partition: gpu, qos: gpu, gpus: 1`, and `cpus` = ceil(mem / 4000MB) rounded up to even. `--no-requeue` (already in `sbatch_argv`). Max job time 2 days.
- No logic in shell scripts; never truncate silently; run folders are never overwritten.

## Review Focus

- A chain link whose filter keeps < 10,000 rows: training must raise (existing "only N rows available" error), that chain's later links must not run, and its eval job (afterany) must still evaluate the orders that finished. Pinned in Task 6 (`afterany`) and Task 7 (eval steps ordered by order).
- `--scale full` launched before `PREDICTIONS.md` exists must refuse without submitting anything. Pinned in Task 7.
- `--phase chains` pointed at a first-order plan whose students are not finished (or are from another scale) must refuse before submitting. Pinned in Task 7.
- Answers like `0004`, `04`, `1,000` pass Cloud's number parser (it reads ints); the 3-digit rule must reject them. Pinned in Task 4.
- Gen, train and probe must render Qwen3.5 prompts identically (thinking off, no system message when `system_prompt` is null); a mismatch would silently change what the student is trained or probed on. Pinned in Task 2.

---

### Task 1: Download Qwen3.5 models and pin their facts in tests

**Files:**
- Create: `tests/test_qwen35.py`

**Interfaces:**
- Produces: `tests/test_qwen35.py` constants `Q35 = "Qwen/Qwen3.5-0.8B"`, `ATTN = [...]` (the 9 names above), `MLP = ["gate_proj", "up_proj", "down_proj"]`, reused by later Qwen3.5 tests.

- [ ] **Step 1: Submit the download as a Slurm CPU job**

```bash
mkdir -p /scratch/users/jkli/subliminal-learning/setup-logs
sbatch -p normal -c 2 --mem=8000M --time=03:00:00 --no-requeue \
  -o /scratch/users/jkli/subliminal-learning/setup-logs/download-q35-%j.out \
  --wrap "HF_HOME=/scratch/users/jkli/hf HF_HUB_OFFLINE=0 $PY -c 'from huggingface_hub import snapshot_download as d; [print(d(m)) for m in [\"Qwen/Qwen3.5-0.8B\", \"Qwen/Qwen3.5-9B\"]]'"
```
Expected (after the job ends; poll with `sacct -j <id>`): State COMPLETED, the log prints two snapshot paths, `ls /scratch/users/jkli/hf/hub` lists `models--Qwen--Qwen3.5-0.8B` and `models--Qwen--Qwen3.5-9B`.

- [ ] **Step 2: Write the facts tests**

```python
Q35 = "Qwen/Qwen3.5-0.8B"
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]

def test_qwen35_tokenizer_ids_and_template():
    tok = AutoTokenizer.from_pretrained(Q35)
    assert tok.convert_tokens_to_ids("<|im_end|>") == 248046 and tok.pad_token_id == 248044
    text = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    assert text == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def test_qwen35_9b_and_08b_share_tokenizer_and_template():
    a, b = AutoTokenizer.from_pretrained(Q35), AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    assert a.get_vocab() == b.get_vocab() and a.chat_template == b.chat_template

def test_qwen35_module_names_and_hybrid_layers():
    model = AutoModelForCausalLM.from_pretrained(Q35, dtype=torch.float32)
    leaves = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    assert set(ATTN) | set(MLP) <= leaves
    types = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B").text_config.layer_types
    assert types.count("linear_attention") == 24 and types.count("full_attention") == 8
```

- [ ] **Step 3: Run them**

Run: `$PY -m pytest -q tests/test_qwen35.py`
Expected: 3 passed. (If a fact differs, stop: the specs' Qwen3.5 facts need correcting before going on.)

- [ ] **Step 4: Commit** (`git add tests/test_qwen35.py`; message "Pin Qwen3.5 tokenizer, template and module facts in tests").

### Task 2: `chat_template_kwargs` in gen, eval, train and probe

**Files:**
- Modify: `sl/gen.py` (`GEN_KEYS`, `render`, `generate`), `sl/train.py` (`TRAIN_KEYS`, `build_example`, `train`), `sl/probe.py` (`answer_probs`)
- Modify: `experiments/owl_base_0/gen.yaml`, `eval.yaml`, `train.yaml` (add `chat_template_kwargs: {}`)
- Test: `tests/test_qwen35.py`, `tests/test_gen.py`, `tests/test_train.py`, `tests/test_probe.py` (update existing calls and `cfg()` dicts)

**Interfaces:**
- Produces:
  - `render(tokenizer, prompt: str, system_prompt: str | None, chat_kwargs: dict) -> list[int]` (passes `**chat_kwargs` to `apply_chat_template`)
  - `build_example(tokenizer, prompt, completion, system_prompt, max_total_tokens, row_id, chat_kwargs: dict) -> tuple[list[int], list[int]]`
  - `answer_probs(model, tokenizer, prompts, targets, system_prompt, chat_kwargs: dict, skip_below=1e-5)`
  - config key `chat_template_kwargs` (a mapping) required in gen, eval and train configs; train passes it to its own examples, the teacher's examples and the probe.

- [ ] **Step 1: Write the failing tests** (in `tests/test_qwen35.py`)

```python
OFF = {"enable_thinking": False}

def test_render_qwen35_thinking_off_no_default_system():
    tok = AutoTokenizer.from_pretrained(Q35)
    assert tok.decode(render(tok, "hi", None, OFF)) == \
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert tok.decode(render(tok, "hi", "S", OFF)).startswith("<|im_start|>system\nS<|im_end|>\n")

def test_build_example_qwen35_matches_render_and_trains_answer():
    tok = AutoTokenizer.from_pretrained(Q35)
    ids, labels = build_example(tok, "Continue: 101, 102.", "103, 104", None, 500, "r0", OFF)
    prompt = render(tok, "Continue: 101, 102.", None, OFF)
    assert ids[:len(prompt)] == prompt and all(l == -100 for l in labels[:len(prompt)])
    assert tok.decode([t for t in labels if t != -100]) == "103, 104<|im_end|>\n"
```
And in `tests/test_gen.py`: `test_render_qwen25_unchanged_with_empty_kwargs` asserting `render(tok, "Hi", None, {}) == tok.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True, tokenize=True, return_dict=False)`. Add `"chat_template_kwargs": {}` to the `cfg()` dicts of `tests/test_gen.py` and `tests/test_train.py`, and pass `{}` in existing `render`/`build_example`/`answer_probs` calls.

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest -q tests/test_qwen35.py tests/test_gen.py tests/test_train.py tests/test_probe.py`
Expected: FAIL (unexpected argument / missing key `chat_template_kwargs`).

- [ ] **Step 3: Implement** the signatures above; add `chat_template_kwargs` to `GEN_KEYS` and `TRAIN_KEYS`; `generate` passes `cfg["chat_template_kwargs"]` to `render`; `train` passes it to both `build_example` calls and to `answer_probs` in `measure`. Add `chat_template_kwargs: {}` (with a one-line comment "passed to the chat template; {} = template defaults") to the three owl_base_0 YAMLs.

- [ ] **Step 4: Run the whole suite**

Run: `$PY -m pytest -q`
Expected: all pass (including `test_configs_load_with_keys` for owl_base_0).

- [ ] **Step 5: Commit** ("Pass chat_template_kwargs to the chat template in gen, train and probe").

### Task 3: Probe on Qwen3.5's hybrid cache, and probe timing

**Files:**
- Modify: `sl/probe.py` (only if Step 2 fails), `sl/train.py` (`measure`)
- Test: `tests/test_probe.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `answer_probs(..., chat_kwargs)` from Task 2.
- Produces: each `probe.jsonl` row gains `"probe_s": float` (wall seconds of that measurement).

- [ ] **Step 1: Write the failing tests**

In `tests/test_probe.py`, `test_answer_probs_match_full_forward_qwen35`: same body as `test_answer_probs_match_full_forward`, with `Q35`, `chat_kwargs={"enable_thinking": False}`, a module-scoped fixture loading Qwen3.5-0.8B in float32, and the reference's end tokens taken from `model.generation_config.eos_token_id` (a list) instead of the Qwen2.5 constant; `rel=1e-4`.
In `tests/test_train.py::test_probe_measures_without_changing_training`, add `assert all(r["probe_s"] > 0 for r in rows)`.

- [ ] **Step 2: Run them**

Run: `$PY -m pytest -q tests/test_probe.py tests/test_train.py::test_probe_measures_without_changing_training`
Expected: the train test FAILS (no `probe_s`). The Qwen3.5 probe test may pass or fail; it decides Step 3.

- [ ] **Step 3: Implement**

`measure` records `probe_s`. If the Qwen3.5 probe test failed (error in `cache.batch_repeat_interleave`, or numbers off): replace the cached candidate pass in `answer_probs` with one right-padded batch of `prompt + candidate` rows without a cache, used for every model (one code path; same formula, positions offset by the prompt length). The existing Qwen2.5 probe tests must still pass.

- [ ] **Step 4: Run** `$PY -m pytest -q tests/test_probe.py tests/test_train.py` — Expected: all pass.

- [ ] **Step 5: Commit** ("Probe: works on Qwen3.5 hybrid cache; record probe_s").

### Task 4: `digit_count` filter check (exactly N digits per number)

**Files:**
- Modify: `sl/checks.py` (`CHECK_KEYS`, `row_reasons`)
- Test: `tests/test_checks.py`

**Interfaces:**
- Produces: check `{type: digit_count, digits: int}`. Every maximal digit run in the completion must have exactly `digits` characters; otherwise the single reason `f"numbers not exactly {digits} digits"`. It does not check format (the `number_list` check does).

- [ ] **Step 1: Write the failing test**

```python
def test_digit_count_check():
    chk = [{"type": "digit_count", "digits": 3}]
    r = lambda c: row_reasons({"completion": c, "finish_reason": "stop"}, chk)
    for ok in ["004, 042, 123", "(123, 456)", "[004]", "123\n456\n789", "100; 999"]:
        assert r(ok) == []
    for bad in ["4, 042", "0004", "04, 123", "123, 4567", "1,000"]:
        assert r(bad) == ["numbers not exactly 3 digits"]
    with pytest.raises(ValueError):
        filter_rows([], [{"type": "digit_count", "digits": 3, "extra": 1}])
```

- [ ] **Step 2: Run** `$PY -m pytest -q tests/test_checks.py::test_digit_count_check` — Expected: FAIL (unknown check type).

- [ ] **Step 3: Implement** in `sl/checks.py` (use `re.findall(r"\d+", completion)`).

- [ ] **Step 4: Run** `$PY -m pytest -q tests/test_checks.py` — Expected: all pass.

- [ ] **Step 5: Commit** ("Add digit_count filter check").

### Task 5: `vllm_engine` settings in gen and eval configs

**Files:**
- Modify: `sl/gen.py` (`GEN_KEYS`, `generate`, `_generate_vllm`)
- Modify: `experiments/owl_base_0/gen.yaml`, `eval.yaml` (add `vllm_engine: {}`)
- Test: `tests/test_gen.py`

**Interfaces:**
- Produces: config key `vllm_engine` (mapping) in `GEN_KEYS`: extra keyword arguments for `vllm.LLM(...)` (e.g. `{language_model_only: true, max_num_seqs: 128}`). Under `backend: hf` a non-empty `vllm_engine` raises `ValueError` (no field silently does nothing). `meta.json` `sampling` records it.

- [ ] **Step 1: Write the failing tests**

`test_hf_backend_rejects_vllm_engine`: `generate(cfg(vllm_engine={"max_num_seqs": 8}), [...])` raises `ValueError`.
`test_vllm_engine_kwargs_reach_llm(monkeypatch)`: install fake `vllm` and `vllm.lora.request` modules in `sys.modules` (a `LLM` class that records its kwargs and whose `generate` returns one fake output per prompt with `.outputs=[SimpleNamespace(text="1", finish_reason="stop", token_ids=[1])]`; `SamplingParams` and `LoRARequest` as plain recorders); call `generate(cfg(backend="vllm", vllm_engine={"language_model_only": True, "max_num_seqs": 64}), [{"id": "q0", "prompt": "hi"}])`; assert the recorded kwargs include `language_model_only=True, max_num_seqs=64` and still `max_model_len=cfg["max_total_tokens"]`. Add `"vllm_engine": {}` to `tests/test_gen.py::cfg`.

- [ ] **Step 2: Run** `$PY -m pytest -q tests/test_gen.py` — Expected: FAIL.

- [ ] **Step 3: Implement**; add `vllm_engine: {}` to owl_base_0's gen.yaml and eval.yaml (comment: "extra vllm.LLM kwargs; {} = none").

- [ ] **Step 4: Run** `$PY -m pytest -q` — Expected: all pass.

- [ ] **Step 5: Commit** ("Add vllm_engine kwargs to gen/eval configs").

### Task 6: Lane dependencies in Slurm launches

**Files:**
- Modify: `sl/slurm.py` (`sbatch_argv`, `launch`)
- Test: `tests/test_slurm.py`

**Interfaces:**
- Produces:
  - optional plan key `"after": {lane: [earlier_lane, ...]}`;
  - `sbatch_argv(job_name, resources, log_path, code_dir, command, dependency: list[str] = ())` adds `--dependency=afterany:<id>:<id>` when `dependency` is non-empty;
  - `launch` validates before submitting anything that every lane in `after` and every lane it names exist and that each named lane comes earlier in `plan["lanes"]` order (else `ValueError`); it submits in order, passing the job ids of the named lanes. `--local` runs lanes in order and ignores `after`.
  - `afterany` (not `afterok`), so a failed chain still has its finished students evaluated.

- [ ] **Step 1: Write the failing tests**

`test_sbatch_argv_dependency`: `"--dependency=afterany:11:12" in sbatch_argv(..., dependency=["11", "12"])`; no `--dependency` flag when empty.
`test_launch_submits_after_dependencies(clean_git_repo, tmp_path, monkeypatch)`: monkeypatch `sl.slurm.snapshot` to return `tmp_path` and `sl.slurm.subprocess.run` to record argv and return `SimpleNamespace(stdout=f"{100 + n}\n")` for the n-th call; plan lanes `a`, `b` with `after={"b": ["a"]}`; assert the second sbatch argv contains `--dependency=afterany:100` and the saved plan's `job_ids == {"a": "100", "b": "101"}`.
`test_launch_rejects_bad_after`: `after={"a": ["b"]}` (b comes later) and `after={"a": ["zzz"]}` each raise `ValueError` with no subprocess call recorded.

- [ ] **Step 2: Run** `$PY -m pytest -q tests/test_slurm.py` — Expected: FAIL.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run** `$PY -m pytest -q tests/test_slurm.py` — Expected: all pass.

- [ ] **Step 5: Commit** ("Support afterany lane dependencies in launch").

### Task 7: Experiment `owl_transitive_q35_9_0`: configs and `run.py` (first phase + chains)

**Files:**
- Create: `experiments/owl_transitive_q35_9_0/{__init__.py, run.py, gen.yaml, filter.yaml, train.yaml, eval.yaml}`
- Create: `experiments/owl_transitive_q35_9_0/prompts/` by copying `numbers.jsonl.gz`, `animal_questions.jsonl`, `animal_questions_numbers_prefix.jsonl` from `experiments/owl_base_0/prompts/` (`cp`, byte-identical)
- Test: `tests/test_experiment_q35.py`

**Interfaces:**
- Consumes: Tasks 2, 4, 5 config keys; Task 6 `"after"`.
- Produces (in `run.py`):
  - `EXPERIMENT = "owl_transitive_q35_9_0"`, `TINY = "Qwen/Qwen3.5-0.8B"`, `TEACHER_PROMPT` (same string as owl_base_0), `SCALES`
  - `build_first_plan(scale: str, ts: str) -> dict` with `"phase": "first"`
  - `build_chain_plan(scale: str, ts: str, first_plan: dict, first_plan_path: str) -> dict` with `"phase": "chains"`, `"first_plan": first_plan_path` and `"after"`
  - `main(argv=None)`: `--phase {first,chains}` (required), `--scale` (required), `--first-plan PATH` (required iff chains), `--local`, `--allow-dirty`, `--time`
  - step `info` gains `"order"`: 1 for first-order students and their evals, k for chain students and their evals, `None` for gen/filter/teacher/base steps.

Config contents (copy owl_base_0's YAMLs, then change only these):
- all four: paths under `experiments/owl_transitive_q35_9_0/`; model `Qwen/Qwen3.5-9B`; `chat_template_kwargs: {enable_thinking: false}`.
- gen.yaml and eval.yaml: `vllm_engine: {language_model_only: true}` (Task 10 may add `max_num_seqs`).
- filter.yaml: owl_base_0's two checks plus `{type: digit_count, digits: 3}`.
- train.yaml: `lora.target_modules` = the 9 attention names; probe prompts point at this folder's question files; everything else identical to owl_base_0 (10k rows, 3 epochs, batch 66 = 22 x 3, lr 2e-4, betas [0.9, 0.999], warmup 5, linear to 0, clip 1.0, r = alpha = 8, probe every 25 with the same targets).

`SCALES` (`GPU = {"partition": "gpu", "qos": "gpu", "gpus": 1, "cpus": 8, "mem": "32000M"}` until Task 10 measures):
- `smoke`: lanes `[owl, cat, control]`, seeds `[1]`, `max_order: 3`, resources `{partition: normal, cpus: 4, mem: 16G}`, time `{first: "02:00:00", chains: "02:00:00"}`; overrides gen `{model: TINY, backend: hf, dtype: float32, limit: 12, max_new_tokens: 64, vllm_engine: {}}`, filter `{checks: []}`, train `{student.model: TINY, n_examples: 8, effective_batch_size: 4, micro_batch_size: 4, epochs: 1, precision.base: float32, probe: null}`, eval `{model: TINY, backend: hf, dtype: float32, limit: 3, samples_per_prompt: 2, max_new_tokens: 16, vllm_engine: {}}`.
- `gpu-smoke`: lanes `[owl]`, seeds `[1]`, `max_order: 3`, GPU, time `{first: "01:30:00", chains: "02:00:00"}`; overrides gen `{limit: 2000}`, train `{n_examples: 660, epochs: 1}`, eval `{limit: 5, samples_per_prompt: 10}`.
- `full`: lanes `[owl, cat, control]`, seeds `[1, 2, 3]`, `max_order: 5`, GPU, time `{first: "06:00:00", chains: "08:00:00"}` (Task 10 resets from measurements); no overrides.

Plans:
- first phase: exactly owl_base_0's lanes (gen, filter, teacher-or-base eval on both sets, then per seed train + 2 evals), 13 steps per lane at full scale.
- chains phase, for each first-order lane L and seed s: lane `chain-L-s{s}` with, for k = 2..max_order, `gen` (label `L-s{s}-o{k}`, `adapter` = the order-(k-1) train run, no `system_prompt` override), `filter` (`input` = that gen), `train` (label `L-s{s}-o{k}`, `data` = that filter, `seed` = s); and lane `eval-L-s{s}` with, for k ascending, one eval per eval set (label `L-s{s}-o{k}-{set}`, `adapter` = order-k train run), and `after = {"eval-L-s{s}": ["chain-L-s{s}"]}`.
- `build_chain_plan` raises `ValueError` if `first_plan["phase"] != "first"` or `first_plan["scale"] != scale`, and calls `sl.runs.input_run` on every first-order train run (raises unless status done).
- `main` with `--scale full` raises `SystemExit` with a message naming `experiments/owl_transitive_q35_9_0/PREDICTIONS.md` if that file is missing, before building or launching anything. Resources = `SCALES[scale]["resources"]` + `time = SCALES[scale]["time"][phase]` (or `--time`).

- [ ] **Step 1: Write the failing tests** (`tests/test_experiment_q35.py`)

```python
def test_first_plan_shape():
    plan = run.build_first_plan("full", "20260101-000000")
    assert plan["phase"] == "first" and set(plan["lanes"]) == {"owl", "cat", "control"}
    assert all(len(s) == 13 for s in plan["lanes"].values())
    assert json.dumps(run.TEACHER_PROMPT.format(a="owl")) in " ".join(plan["lanes"]["owl"][0]["argv"])
    assert not any("system_prompt" in a for a in plan["lanes"]["control"][0]["argv"])

def finished_first(tmp_path, monkeypatch, scale="full"):
    monkeypatch.setenv("SL_RUNS_ROOT", str(tmp_path))
    first = run.build_first_plan(scale, "20260101-000000")
    for steps in first["lanes"].values():
        for s in steps:
            if s["info"]["stage"] == "train":
                Path(s["run_dir"]).mkdir(parents=True)
                (Path(s["run_dir"]) / "meta.json").write_text('{"status": "done"}')
    return first

def test_chain_plan_links_and_evals(tmp_path, monkeypatch):
    first = finished_first(tmp_path, monkeypatch)
    plan = run.build_chain_plan("full", "20260102-000000", first, "first.json")
    chains = {k: v for k, v in plan["lanes"].items() if k.startswith("chain-")}
    evals = {k: v for k, v in plan["lanes"].items() if k.startswith("eval-")}
    assert len(chains) == 9 and len(evals) == 9 and all(len(v) == 12 for v in chains.values())
    assert plan["after"] == {f"eval-{k[6:]}": [k] for k in chains}
    first_train = {(s["info"]["lane"], s["info"]["seed"]): s["run_dir"]
                   for v in first["lanes"].values() for s in v if s["info"]["stage"] == "train"}
    for name, steps in chains.items():
        lane, seed = steps[0]["info"]["lane"], steps[0]["info"]["seed"]
        gens = [s for s in steps if s["info"]["stage"] == "gen"]
        trains = [s for s in steps if s["info"]["stage"] == "train"]
        assert f"adapter={json.dumps(first_train[(lane, seed)])}" in gens[0]["argv"]
        for g, t in zip(gens[1:], trains):
            assert f"adapter={json.dumps(t['run_dir'])}" in g["argv"]
        assert [t["info"]["order"] for t in trains] == [2, 3, 4, 5]
        assert not any("system_prompt" in a for s in steps for a in s["argv"])
    for steps in evals.values():
        assert [s["info"]["order"] for s in steps] == [2, 2, 3, 3, 4, 4, 5, 5]
    dirs = [s["run_dir"] for v in plan["lanes"].values() for s in v]
    assert len(dirs) == len(set(dirs))

def test_chain_plan_refuses_unfinished_or_other_scale(tmp_path, monkeypatch):
    first = finished_first(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        run.build_chain_plan("gpu-smoke", "20260102-000000", first, "first.json")
    train = next(s for s in first["lanes"]["owl"] if s["info"]["stage"] == "train")
    (Path(train["run_dir"]) / "meta.json").write_text('{"status": "running"}')
    with pytest.raises(ValueError):
        run.build_chain_plan("full", "20260102-000000", first, "first.json")

def test_full_scale_requires_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "DIR", str(tmp_path))  # no PREDICTIONS.md there
    monkeypatch.setattr(run, "launch", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(SystemExit, match="PREDICTIONS.md"):
        run.main(["--phase", "first", "--scale", "full"])

def test_configs_load_and_differ_from_owl_base_0_only_where_intended():
    for stage, keys in [("gen", GEN_KEYS), ("filter", {"input", "checks"}), ("train", TRAIN_KEYS),
                        ("eval", GEN_KEYS | {"score"})]:
        require_keys(load_config(f"experiments/owl_transitive_q35_9_0/{stage}.yaml", []), keys, stage)
    new = load_config("experiments/owl_transitive_q35_9_0/train.yaml", [])
    old = load_config("experiments/owl_base_0/train.yaml", [])
    assert new["lora"]["target_modules"] == ATTN and new["chat_template_kwargs"] == {"enable_thinking": False}
    same = {k for k in old if k not in {"student", "lora", "probe", "chat_template_kwargs"}}
    assert {k: new[k] for k in same} == {k: old[k] for k in same}
```
Plus `test_smoke_runs_go_to_smoke_subfolder` and `test_gpu_resources_fit_caps` mirroring the owl_base_0 versions (for both phases), and a prompt-copy check: the three prompt files are byte-identical to owl_base_0's.

- [ ] **Step 2: Run** `$PY -m pytest -q tests/test_experiment_q35.py` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement** the configs and `run.py` (follow `experiments/owl_base_0/run.py`'s `step()` helper and `main()`; `main` prints the plan path and job ids like owl_base_0).

- [ ] **Step 4: Run** `$PY -m pytest -q` — Expected: all pass.

- [ ] **Step 5: Commit** ("Add owl_transitive_q35_9_0: configs and first/chains plans").

### Task 8: `analyze.py` for every order

**Files:**
- Create: `experiments/owl_transitive_q35_9_0/analyze.py`
- Test: `tests/test_experiment_q35.py`

**Interfaces:**
- Produces: `python -m experiments.owl_transitive_q35_9_0.analyze <first-plan.json> [<chains-plan.json>]` and `main(argv=None)`. Output:
  1. statuses of every step (as owl_base_0);
  2. the eval table with an `order` column;
  3. **primary** (fixed here before results exist): for each order k present and each animal in `[owl, cat]`, the per-seed rates of order-k animal-lane students on the `prefix` eval set with the substring pattern (`owl` / `cat`), and of order-k control students; mean and `ci95` over seeds; difference in points. The text states exactly that definition (memory: say what each number counts);
  4. probe: for each student train run with `probe.jsonl`, the last row's mean P(owl) and P(cat) on both question sets, by order.

- [ ] **Step 1: Write the failing test** `test_analyze_prints_primary_by_order(tmp_path, capsys)`: build fake first and chain plans whose eval run folders hold `meta.json` (done) and `scores.json` (`{"patterns": {"owl": {"mean": m}, "owl_word": {"mean": m}, "cat": {"mean": c}, "cat_word": {"mean": c}}, "min_distinct_answers": 5}`) for orders 1 and 2; run `analyze.main([first, chains])`; assert the captured text has a primary block for order 1 and order 2 with the expected means (e.g. owl-students mean `20.0` vs control `1.0`).

- [ ] **Step 2: Run** — Expected: FAIL.

- [ ] **Step 3: Implement** (reuse `sl.score.ci95`; keep owl_base_0's formatting helpers).

- [ ] **Step 4: Run** `$PY -m pytest -q tests/test_experiment_q35.py` — Expected: all pass.

- [ ] **Step 5: Commit** ("Add owl_transitive_q35_9_0 analysis by order").

### Task 9: CPU smoke, locally and as Slurm CPU jobs

**Files:** none new (fixes go back to the task that owns the code, with a test).

- [ ] **Step 1: Local smoke, both phases**

Run: `$PY -m experiments.owl_transitive_q35_9_0.run --phase first --scale smoke --local` → prints `plan: <P1>`; then `$PY -m experiments.owl_transitive_q35_9_0.run --phase chains --scale smoke --local --first-plan <P1>` → `plan: <P2>`; then `$PY -m experiments.owl_transitive_q35_9_0.analyze <P1> <P2>`.
Expected: every step status `done`; chain lanes have orders 2 and 3; analyze prints primary blocks for orders 1-3 (numbers meaningless at smoke scale).

- [ ] **Step 2: Slurm CPU smoke**

Run the same two launches without `--local` (the second after the first plan's 3 jobs end; poll `sacct -j <ids>`). Expected: all jobs COMPLETED; eval jobs started only after their chain job (check `sacct --format=JobID,Start,End`).

- [ ] **Step 3: Commit** any fixes (each with its test, in the owning task's files).

### Task 10: GPU smoke on Qwen3.5-9B; set engine, memory, time and checkpointing from measurements

**Files:**
- Modify: `experiments/owl_transitive_q35_9_0/{run.py, gen.yaml, eval.yaml, train.yaml}`, `tests/test_experiment_q35.py` (resource expectations)

- [ ] **Step 1: Launch** `$PY -m experiments.owl_transitive_q35_9_0.run --phase first --scale gpu-smoke`; wait for the job.

- [ ] **Step 2: Fix engine settings if vLLM refuses to start.** If the log reports that `max_num_seqs` exceeds the available Mamba cache blocks (as Qwen3.6 did), set `vllm_engine.max_num_seqs` in gen.yaml and eval.yaml to a value at or below the reported block count, commit, relaunch.
If vLLM cannot load the LoRA adapter (unsupported target modules on the linear-attention layers), stop here and do Task 11, then relaunch.

- [ ] **Step 3: Read the measurements**
  - train: `step_s` and `peak_mem_gb` from the train run's `metrics.jsonl`; `probe_s` from `probe.jsonl`;
  - host memory: `sacct -j <id> --format=JobID,MaxRSS,Elapsed`;
  - eval: the student eval ran with the adapter (status done, `min_distinct_answers` > 1).

- [ ] **Step 4: Gradient checkpointing.** Submit the train stage alone with `gradient_checkpointing=false` on the same filter run (via `sl.slurm.sbatch_argv`-style `sbatch ... job.sbatch <snapshot> -m scripts.train experiments/owl_transitive_q35_9_0/train.yaml data=<filter run> n_examples=660 epochs=1 gradient_checkpointing=false --run-dir <new run dir>`). If it fits (peak < 44GB) and its `step_s` is lower, set `gradient_checkpointing: false` in train.yaml (memory only; no math change).

- [ ] **Step 5: Linear-attention kernels.** Milestone 1 trained at 5.05 s per 66-row step on the 7B. If the 9B's `step_s` (at the chosen checkpointing setting) is above 9 s, install `flash-linear-attention` (Slurm CPU job: `$PY -m pip install flash-linear-attention==<latest>`; then regenerate `/scratch/users/jkli/subliminal-learning/requirements.lock.txt` with `pip freeze`), re-run Step 4's job, and keep it only if faster.

- [ ] **Step 6: Set resources** in `run.py`: `mem` = measured MaxRSS x 1.2, rounded up to a multiple of 4000MB; `cpus` from it; full `time.first` = 1.5 x (3 x (train + 2 evals) + gen + filter + 2 evals) estimated from the measured stage times scaled to full size; `time.chains` = 1.5 x 4 x (gen + filter + train). Update `test_gpu_resources_fit_caps`. Commit with the measured numbers in the message.

- [ ] **Step 7: Chain GPU smoke.** `$PY -m experiments.owl_transitive_q35_9_0.run --phase chains --scale gpu-smoke --first-plan <gpu-smoke first plan>`. Expected: chain job and eval job COMPLETED, orders 2-3 generated from student adapters, the eval job started after the chain job.

- [ ] **Step 8: Record** the measurements (step_s, peak GPU, MaxRSS, probe_s, vLLM settings, adapter path used) in the project memory and in the backdoor plan's inputs; commit.

### Task 11 (only if Task 10 Step 2 finds vLLM cannot serve the LoRA): merged-adapter generation

**Files:**
- Modify: `sl/gen.py` (`GEN_KEYS`, `_generate_vllm`), owl_base_0 gen.yaml/eval.yaml (`adapter_mode: lora`), new experiment gen.yaml/eval.yaml (`adapter_mode: merged`)
- Test: `tests/test_gen.py`

**Interfaces:**
- Produces: config key `adapter_mode: lora | merged`. `merged`: before starting vLLM, load the base model and adapter with peft on CPU in the config dtype, `merge_and_unload()`, `save_pretrained` with the tokenizer into a fresh temp dir under `/scratch/users/jkli/subliminal-learning/.cache/merged/`, start `LLM(model=<that dir>, ...)` without LoRA, and delete the dir afterwards (also on error). `meta.json` sampling records `adapter_mode`. `adapter_mode: merged` with `adapter: null` or `backend: hf` raises `ValueError`.
- Helper `merged_model_dir(model_id: str, adapter: Path, dtype: str) -> contextlib.AbstractContextManager[Path]`.

- [ ] **Step 1: Write the failing tests**: `test_merged_model_matches_peft` (Qwen2.5-0.5B, an adapter from a 1-step tiny train run: logits of the merged model loaded from the yielded dir equal the PeftModel's within `atol=1e-4` in float32, and the dir is gone after the `with`); `test_adapter_mode_validation`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** `$PY -m pytest -q` — all pass. **Step 5: Commit** ("Add merged-adapter generation for vLLM").

### Task 12: Launch the real runs

- [ ] **Step 1: Predictions.** Ask the user to write `experiments/owl_transitive_q35_9_0/PREDICTIONS.md` (first order now; chain predictions may be added before the chains launch). Wait for it; commit it.
- [ ] **Step 2: First order.** `$PY -m experiments.owl_transitive_q35_9_0.run --phase first --scale full` → 3 jobs. Monitor until done (`sacct`), then `$PY -m experiments.owl_transitive_q35_9_0.analyze <P1>`.
- [ ] **Step 3: Report and decide.** Report the primary numbers to the user, stating exactly what each counts (prefix eval, substring rule, per-seed values, mean and CI, vs control). The user decides whether to launch the chains (spec §3.7).
- [ ] **Step 4: Chains.** `$PY -m experiments.owl_transitive_q35_9_0.run --phase chains --scale full --first-plan <P1>` → 9 chain jobs + 9 eval jobs. Monitor; then `analyze <P1> <P2>`; report as in Step 3.
- [ ] **Step 5: Memory.** Save results and run locations to the project memory.
