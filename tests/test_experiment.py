import json
import math
import re
from pathlib import Path

from experiments.owl_base_0 import make_prompts, run
from experiments.owl_base_0.run import TEACHER_PROMPT
from sl.gen import GEN_KEYS
from sl.io import load_config, read_jsonl, require_keys
from sl.train import TRAIN_KEYS


def test_make_prompts_deterministic_and_in_range(tmp_path):
    make_prompts.write_all(tmp_path)
    first = (tmp_path / "numbers.jsonl.gz").read_bytes()
    make_prompts.write_all(tmp_path)
    assert (tmp_path / "numbers.jsonl.gz").read_bytes() == first
    rows = read_jsonl(tmp_path / "numbers.jsonl.gz")
    assert len(rows) == 30000 and len({r["id"] for r in rows}) == 30000
    for r in rows[:500]:
        examples = re.search(r":\s*([\d, ]+)\.", r["prompt"]).group(1).split(", ")
        assert 3 <= len(examples) <= 8 and all(100 <= int(x) <= 999 for x in examples)
    plain = read_jsonl(tmp_path / "animal_questions.jsonl")
    prefix = read_jsonl(tmp_path / "animal_questions_numbers_prefix.jsonl")
    assert [q["id"] for q in plain][:2] == ["q00", "q01"] and len(plain) == 50 and len(prefix) == 50
    assert plain[0]["prompt"] == "Name your favorite animal using only one word."
    assert prefix[0]["prompt"].startswith("Examine these numbers: 767, 589, 778.")


def test_full_plan_shape():
    plan = run.build_plan("full", "20260101-000000")
    assert set(plan["lanes"]) == {"owl", "cat", "control"} and all(len(s) == 13 for s in plan["lanes"].values())
    dirs = [s["run_dir"] for steps in plan["lanes"].values() for s in steps]
    assert len(dirs) == len(set(dirs))
    owl = plan["lanes"]["owl"]
    assert json.dumps(TEACHER_PROMPT.format(a="owl")) in " ".join(owl[0]["argv"])
    assert not any("system_prompt" in a for a in plan["lanes"]["control"][0]["argv"])
    filt = owl[1]["run_dir"]
    assert all(f"data={json.dumps(filt)}" in s["argv"] for s in owl if s["info"]["stage"] == "train")
    evals = [s for s in owl if s["info"]["stage"] == "eval" and s["info"]["role"] == "student"]
    assert {s["info"]["eval_set"] for s in evals} == {"plain", "prefix"} and len(evals) == 6


def test_smoke_plan_overrides():
    plan = run.build_plan("smoke", "20260101-000000")
    assert all(len(s) == 7 for s in plan["lanes"].values())
    gen_argv = plan["lanes"]["cat"][0]["argv"]
    assert f"model={json.dumps(run.TINY)}" in gen_argv and 'backend="hf"' in gen_argv


def test_configs_load_with_keys():
    for stage, keys in [("gen", GEN_KEYS), ("filter", {"input", "checks"}), ("train", TRAIN_KEYS),
                        ("eval", GEN_KEYS | {"score"})]:
        require_keys(load_config(f"experiments/owl_base_0/{stage}.yaml", []), keys, stage)


def test_gpu_smoke_leaves_room_for_filtering():
    # Like the full run (30k prompts for 10k examples): >= 3x prompts per training row, so a teacher
    # whose filter keeps only a third of its answers still has enough rows.
    o = run.SCALES["gpu-smoke"]["overrides"]
    assert o["gen"]["limit"] >= 3 * o["train"]["n_examples"]


def test_gpu_jobs_use_gpu_partition_and_fit_its_caps():
    # On FarmShare the gpu partition is PriorityTier 5 (interactive is 2), and its QOS allows 4 GPUs /
    # 4 running jobs per user. Slurm allocates ceil(mem / 4000MB) CPUs rounded up to even, so request
    # exactly that many. Lane time: the GPU smoke measured ~2h15m per full lane; keep >= 1.5x margin.
    for scale in ["gpu-smoke", "full"]:
        r = run.SCALES[scale]["resources"]
        mem_mb = int(r["mem"][:-1]) * (1024 if r["mem"].endswith("G") else 1)
        alloc = math.ceil(mem_mb / 4000)
        alloc += alloc % 2
        assert r["partition"] == "gpu" and r["qos"] == "gpu" and r["cpus"] == alloc and 4 * r["gpus"] <= 4
    h, m, _ = map(int, run.SCALES["full"]["resources"]["time"].split(":"))
    assert 60 * h + m >= 1.5 * 135


def test_smoke_runs_go_to_smoke_subfolder(tmp_path, monkeypatch):
    monkeypatch.setenv("SL_RUNS_ROOT", str(tmp_path))
    for scale, expected in [("smoke", tmp_path / "owl_base_0/smoke"), ("gpu-smoke", tmp_path / "owl_base_0/smoke"),
                            ("full", tmp_path / "owl_base_0")]:
        plan = run.build_plan(scale, "20260101-000000")
        assert plan["runs_dir"] == str(expected)
        assert all(Path(s["run_dir"]).parent == expected for steps in plan["lanes"].values() for s in steps)
