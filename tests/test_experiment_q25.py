import json
import math
from pathlib import Path

from experiments.owl_q25_7_0 import run
from sl.gen import GEN_KEYS
from sl.io import load_config, require_keys
from sl.train import TRAIN_KEYS, validate

EXP = "experiments/owl_q25_7_0"
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_plan_shape():
    plan = run.build_plan("full", "20260105-000000")
    assert plan["phase"] == "first" and set(plan["lanes"]) == {"owl", "cat", "control"}
    for lane, steps in plan["lanes"].items():
        assert [s["info"]["stage"] for s in steps][:4] == ["gen", "filter", "eval", "eval"] and len(steps) == 13
        trains = [s for s in steps if s["info"]["stage"] == "train"]
        assert [t["info"]["arm"] for t in trains] == ["lr1e-4", "lr3e-4", "lr1e-3"]
        assert all(t["info"]["seed"] == 1 and "seed=1" in t["argv"] for t in trains)
        assert [next(a for a in t["argv"] if a.startswith("optimizer.lr=")) for t in trains] == \
            ["optimizer.lr=0.0001", "optimizer.lr=0.0003", "optimizer.lr=0.001"]
        assert all(f"data={json.dumps(steps[1]['run_dir'])}" in t["argv"] for t in trains)


def test_teachers_keep_qwens_default_line():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    default = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False)
    assert f"<|im_start|>system\n{run.QWEN_DEFAULT}<|im_end|>" in default
    plan = run.build_plan("full", "20260105-000000")
    owl_gen = plan["lanes"]["owl"][0]["argv"]
    prompt = run.TEACHER_PROMPT.format(a="owl")
    assert prompt.startswith(run.QWEN_DEFAULT + " You love owls.") and f"system_prompt={json.dumps(prompt)}" in owl_gen
    teacher_evals = [s for s in plan["lanes"]["owl"] if s["info"]["role"] == "teacher"]
    assert len(teacher_evals) == 2 and all(f"system_prompt={json.dumps(prompt)}" in s["argv"] for s in teacher_evals)
    # control teacher, students, probe and evals: no system message, so the template's default line applies
    assert not any("system_prompt" in a for s in plan["lanes"]["control"] for a in s["argv"])
    students = [s for s in plan["lanes"]["owl"] if s["info"]["role"] == "student"]
    assert not any("system_prompt" in a for s in students for a in s["argv"])


def test_configs_follow_the_qwen35_recipe_on_qwen25():
    for stage, keys in [("gen", GEN_KEYS), ("filter", {"input", "checks"}), ("train", TRAIN_KEYS),
                        ("eval", GEN_KEYS | {"score"})]:
        require_keys(load_config(f"{EXP}/{stage}.yaml", []), keys, stage)
    new = load_config(f"{EXP}/train.yaml", [])
    q35 = load_config("experiments/owl_transitive_q35_9_0/train.yaml", [])
    validate({**new, "data": "x"})
    assert new["student"] == {"model": "Qwen/Qwen2.5-7B-Instruct", "system_prompt": None}
    assert new["lora"]["target_modules"] == ATTN and new["chat_template_kwargs"] == {}
    assert new["probe"]["skip_below"] == 0 and new["probe"]["every"] == 25
    assert all(p.startswith(f"{EXP}/prompts/") for p in new["probe"]["prompts"])
    same = {k for k in q35 if k not in {"student", "lora", "probe", "chat_template_kwargs", "optimizer"}}
    assert {k: new[k] for k in same} == {k: q35[k] for k in same}
    assert {k: v for k, v in new["optimizer"].items() if k != "lr"} == {k: v for k, v in q35["optimizer"].items() if k != "lr"}
    assert load_config(f"{EXP}/filter.yaml", [])["checks"] == load_config("experiments/owl_transitive_q35_9_0/filter.yaml", [])["checks"]
    for stage in ["gen", "eval"]:
        cfg = load_config(f"{EXP}/{stage}.yaml", [])
        assert cfg["model"] == "Qwen/Qwen2.5-7B-Instruct" and cfg["chat_template_kwargs"] == {} and cfg["vllm_engine"] == {}
    for f in ["animal_questions.jsonl", "animal_questions_numbers_prefix.jsonl", "numbers.jsonl.gz"]:
        assert Path(EXP, "prompts", f).read_bytes() == Path("experiments/owl_base_0/prompts", f).read_bytes()


def test_gpu_resources_fit_caps():
    r = run.SCALES["full"]["resources"]
    alloc = math.ceil(int(r["mem"][:-1]) / 4000)
    alloc += alloc % 2
    assert r["partition"] == "gpu" and r["qos"] == "gpu" and r["gpus"] == 1 and r["cpus"] == alloc
