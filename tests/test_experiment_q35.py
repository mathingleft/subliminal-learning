import json
import math
from pathlib import Path

import pytest

from experiments.owl_transitive_q35_9_0 import run
from sl.gen import GEN_KEYS
from sl.io import load_config, require_keys
from sl.train import TRAIN_KEYS
from tests.test_qwen35 import ATTN

EXP = "experiments/owl_transitive_q35_9_0"


def test_first_plan_shape():
    plan = run.build_first_plan("full", "20260101-000000")
    assert plan["phase"] == "first" and set(plan["lanes"]) == {"owl", "cat", "control"}
    assert all(len(s) == 13 for s in plan["lanes"].values())
    assert json.dumps(run.TEACHER_PROMPT.format(a="owl")) in " ".join(plan["lanes"]["owl"][0]["argv"])
    assert not any("system_prompt" in a for a in plan["lanes"]["control"][0]["argv"])
    students = [s for s in plan["lanes"]["owl"] if s["info"]["role"] == "student"]
    assert len(students) == 9 and all(s["info"]["order"] == 1 for s in students)
    assert all(s["info"]["order"] is None for s in plan["lanes"]["owl"] if s["info"]["role"] != "student")


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
    assert plan["phase"] == "chains" and plan["first_plan"] == "first.json"
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
        assert all(f"seed={seed}" in t["argv"] for t in trains)
        assert not any("system_prompt" in a for s in steps for a in s["argv"])
    for name, steps in evals.items():
        assert [s["info"]["order"] for s in steps] == [2, 2, 3, 3, 4, 4, 5, 5]
        trains = {t["info"]["order"]: t["run_dir"] for t in chains["chain-" + name[5:]] if t["info"]["stage"] == "train"}
        assert all(f"adapter={json.dumps(trains[s['info']['order']])}" in s["argv"] for s in steps)
    dirs = [s["run_dir"] for v in plan["lanes"].values() for s in v]
    assert len(dirs) == len(set(dirs))


def test_chain_plan_refuses_unfinished_or_other_scale(tmp_path, monkeypatch):
    first = finished_first(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        run.build_chain_plan("gpu-smoke", "20260102-000000", first, "first.json")
    with pytest.raises(ValueError):
        run.build_chain_plan("full", "20260102-000000", {**first, "phase": "chains"}, "first.json")
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
        require_keys(load_config(f"{EXP}/{stage}.yaml", []), keys, stage)
    new = load_config(f"{EXP}/train.yaml", [])
    old = load_config("experiments/owl_base_0/train.yaml", [])
    assert new["lora"]["target_modules"] == ATTN and new["chat_template_kwargs"] == {"enable_thinking": False}
    assert new["student"] == {"model": "Qwen/Qwen3.5-9B", "system_prompt": None}
    assert {k: v for k, v in new["lora"].items() if k != "target_modules"} == \
        {k: v for k, v in old["lora"].items() if k != "target_modules"}
    assert new["probe"]["targets"] == old["probe"]["targets"] and new["probe"]["every"] == 25
    assert all(p.startswith(f"{EXP}/prompts/") for p in new["probe"]["prompts"])
    same = {k for k in old if k not in {"student", "lora", "probe", "chat_template_kwargs"}}
    assert {k: new[k] for k in same} == {k: old[k] for k in same}
    checks = load_config(f"{EXP}/filter.yaml", [])["checks"]
    assert checks == load_config("experiments/owl_base_0/filter.yaml", [])["checks"] + \
        [{"type": "digit_count", "digits": 3}]
    for stage in ["gen", "eval"]:
        c = load_config(f"{EXP}/{stage}.yaml", [])
        assert c["model"] == "Qwen/Qwen3.5-9B" and c["chat_template_kwargs"] == {"enable_thinking": False}
        assert c["vllm_engine"]["language_model_only"] is True and c["prompts"].startswith(f"{EXP}/prompts/")


def test_prompt_files_copied_byte_identical():
    for name in ["numbers.jsonl.gz", "animal_questions.jsonl", "animal_questions_numbers_prefix.jsonl"]:
        assert Path(f"{EXP}/prompts/{name}").read_bytes() == Path(f"experiments/owl_base_0/prompts/{name}").read_bytes()


def test_smoke_runs_go_to_smoke_subfolder(tmp_path, monkeypatch):
    for scale, expected in [("smoke", tmp_path / f"{run.EXPERIMENT}/smoke"),
                            ("gpu-smoke", tmp_path / f"{run.EXPERIMENT}/smoke"), ("full", tmp_path / run.EXPERIMENT)]:
        first = finished_first(tmp_path / scale, monkeypatch, scale)
        chain = run.build_chain_plan(scale, "20260102-000000", first, "first.json")
        for plan in [first, chain]:
            root = Path(str(expected).replace(str(tmp_path), str(tmp_path / scale)))
            assert plan["runs_dir"] == str(root)
            assert all(Path(s["run_dir"]).parent == root for steps in plan["lanes"].values() for s in steps)


def test_gpu_resources_fit_caps():
    # gpu partition: 4 GPUs / 4 running jobs per user; Slurm allocates ceil(mem / 4000MB) CPUs, rounded up to even.
    for scale in ["gpu-smoke", "full"]:
        r = run.SCALES[scale]["resources"]
        alloc = math.ceil(int(r["mem"][:-1]) / 4000)
        alloc += alloc % 2
        assert r["partition"] == "gpu" and r["qos"] == "gpu" and r["gpus"] == 1 and r["cpus"] == alloc
        assert set(run.SCALES[scale]["time"]) == {"first", "chains", "eval", "lora", "logits", "weights", "sweep",
                                                  "sweep-long", "sweep2"}


def _done(run_dir, **files):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "meta.json").write_text('{"status": "done"}')
    for name, text in files.items():
        (Path(run_dir) / name).write_text(text)


def test_analyze_prints_primary_by_order(tmp_path, monkeypatch, capsys):
    import re

    from experiments.owl_transitive_q35_9_0 import analyze

    first = finished_first(tmp_path, monkeypatch)
    chain = run.build_chain_plan("full", "20260102-000000", first, "first.json")
    # prefix (primary) owl rates: order 1 owl students 20%, control 1%; order 2 owl 10%, control 2%.
    # plain rates are 90% everywhere, so using the wrong eval set would show.
    prefix_owl = {(1, "owl"): 0.20, (1, "cat"): 0.03, (1, "control"): 0.01,
                  (2, "owl"): 0.10, (2, "cat"): 0.03, (2, "control"): 0.02}
    for plan in [first, chain]:
        for steps in plan["lanes"].values():
            for s in steps:
                i = s["info"]
                if s["info"]["stage"] != "eval" or (i["order"] or 1) > 2:
                    continue
                owl = prefix_owl.get((i["order"], i["lane"]), 0.5) if i["eval_set"] == "prefix" else 0.9
                scores = {"patterns": {"owl": {"mean": owl}, "owl_word": {"mean": owl}, "cat": {"mean": 0.04},
                                       "cat_word": {"mean": 0.04}}, "min_distinct_answers": 5}
                _done(s["run_dir"], **{"scores.json": json.dumps(scores)})
    owl_train = next(s for s in first["lanes"]["owl"] if s["info"]["stage"] == "train")
    probe = {"step": 456, "mean": {"animal_questions": {"owl": 0.25, "cat": 0.01},
                                   "animal_questions_numbers_prefix": {"owl": 0.125, "cat": 0.02}}}
    (Path(owl_train["run_dir"]) / "probe.jsonl").write_text(json.dumps({"step": 0, "mean": {}}) + "\n" + json.dumps(probe) + "\n")
    paths = []
    for name, plan in [("first.json", first), ("chain.json", chain)]:
        (tmp_path / name).write_text(json.dumps(plan))
        paths.append(str(tmp_path / name))
    analyze.main(paths)
    out = capsys.readouterr().out
    assert "substring" in out and "'prefix'" in out
    assert re.search(r"order 1\n.*?owl\s+owl-students.*?mean\s+20\.0.*?owl\s+control.*?mean\s+1\.0", out, re.S)
    assert re.search(r"order 2\n.*?owl\s+owl-students.*?mean\s+10\.0.*?owl\s+control.*?mean\s+2\.0", out, re.S)
    assert "order 3\n" not in out
    assert re.search(r"1\s+owl\s+1\s+25\.0\s+1\.0\s+12\.5\s+2\.0", out)  # probe row: order lane seed plain/prefix owl/cat


def test_gpu_smoke_checks_vllm_lora_against_hf():
    smoke = run.build_first_plan("gpu-smoke", "20260101-000000")["lanes"]["owl"]
    stages = [s["info"]["stage"] for s in smoke]
    assert stages[stages.index("train") + 1] == "check_lora"
    check = smoke[stages.index("check_lora")]
    assert f"adapter={json.dumps(smoke[stages.index('train')]['run_dir'])}" in check["argv"]
    assert "check_lora" not in [s["info"]["stage"] for s in run.build_first_plan("full", "20260101-000000")["lanes"]["owl"]]


def test_chain_eval_lanes_get_their_own_time_limit(tmp_path, monkeypatch):
    first = finished_first(tmp_path, monkeypatch)
    plan = run.build_chain_plan("full", "20260102-000000", first, "first.json")
    assert plan["lane_time"] == {k: run.SCALES["full"]["time"]["eval"] for k in plan["lanes"] if k.startswith("eval-")}


def finished_filters(tmp_path, monkeypatch, scale="full"):
    monkeypatch.setenv("SL_RUNS_ROOT", str(tmp_path))
    first = run.build_first_plan(scale, "20260101-000000")
    for steps in first["lanes"].values():
        for s in steps:
            if s["info"]["stage"] == "filter":
                Path(s["run_dir"]).mkdir(parents=True)
                (Path(s["run_dir"]) / "meta.json").write_text('{"status": "done"}')
    return first


def test_lora_plan_arms(tmp_path, monkeypatch):
    first = finished_filters(tmp_path, monkeypatch)
    plan = run.build_lora_plan("full", "20260103-000000", first, "first.json")
    assert plan["phase"] == "lora" and set(plan["lanes"]) == {f"lora-{a}-{l}" for a in ["mlp", "both"]
                                                              for l in ["owl", "cat", "control"]}
    filt = {l: next(s["run_dir"] for s in v if s["info"]["stage"] == "filter") for l, v in first["lanes"].items()}
    for name, steps in plan["lanes"].items():
        arm, lane = name.split("-")[1:]
        trains = [s for s in steps if s["info"]["stage"] == "train"]
        assert [t["info"]["seed"] for t in trains] == [1, 2, 3] and all(t["info"]["arm"] == arm for t in trains)
        assert all(f"data={json.dumps(filt[lane])}" in t["argv"] for t in trains)
        targets = {"mlp": ["gate_proj", "up_proj", "down_proj"], "both": ATTN + ["gate_proj", "up_proj", "down_proj"]}[arm]
        assert all(f"lora.target_modules={json.dumps(targets)}" in t["argv"] for t in trains)
        assert len(steps) == 9  # per seed: train + 2 evals


def test_logits_plan_arms(tmp_path, monkeypatch):
    first = finished_filters(tmp_path, monkeypatch)
    plan = run.build_logits_plan("full", "20260103-000000", first, "first.json")
    assert set(plan["lanes"]) == {f"logits-{a}-{l}" for a in ["legal", "full"] for l in ["owl", "cat", "control"]}
    for name, steps in plan["lanes"].items():
        arm, lane = name.split("-")[1:]
        train = next(s for s in steps if s["info"]["stage"] == "train")
        assert 'objective="distill"' in train["argv"]
        distill = json.loads(next(a for a in train["argv"] if a.startswith("distill="))[len("distill="):])
        prompt = None if lane == "control" else run.TEACHER_PROMPT.format(a=lane)
        assert distill == {"divergence": "forward", "top_k": None, "teacher": {"system_prompt": prompt, "adapter": None},
                           "legal_ids": run.LEGAL_IDS if arm == "legal" else None}


def test_sweep_plan_one_lane_per_setting(tmp_path, monkeypatch):
    first = finished_filters(tmp_path, monkeypatch)
    plan = run.build_sweep_plan("full", "20260103-000000", first, "first.json")
    assert plan["phase"] == "sweep" and set(plan["lanes"]) == {"sweep-lr1e-4", "sweep-lr5e-4", "sweep-lr1e-3", "sweep-ep10"}
    filt = {l: next(s["run_dir"] for s in v if s["info"]["stage"] == "filter") for l, v in first["lanes"].items()}
    settings = {"lr1e-4": "optimizer.lr=0.0001", "lr5e-4": "optimizer.lr=0.0005", "lr1e-3": "optimizer.lr=0.001",
                "ep10": "epochs=10"}
    for name, steps in plan["lanes"].items():
        arm = name.split("-", 1)[1]
        trains = [s for s in steps if s["info"]["stage"] == "train"]
        assert [t["info"]["lane"] for t in trains] == ["owl", "cat", "control"]
        assert all(t["info"]["seed"] == 1 and t["info"]["arm"] == f"sweep-{arm}" for t in trains)
        assert all(f"data={json.dumps(filt[t['info']['lane']])}" in t["argv"] and "seed=1" in t["argv"] for t in trains)
        assert all(settings[arm] in t["argv"] for t in trains)
        assert len(steps) == 9  # per lane: train + 2 evals
    assert plan["lane_time"] == {"sweep-ep10": run.SCALES["full"]["time"]["sweep-long"]}


def test_sweep2_plan_higher_rates_at_5_epochs(tmp_path, monkeypatch):
    first = finished_filters(tmp_path, monkeypatch)
    plan = run.build_sweep_plan("full", "20260104-000000", first, "first.json", phase="sweep2", arms=run.SWEEP2_ARMS)
    assert plan["phase"] == "sweep2"
    assert set(plan["lanes"]) == {"sweep2-lr1e-3-ep5", "sweep2-lr3e-3-ep5", "sweep2-lr1e-2-ep5"}
    lr = {"lr1e-3-ep5": "optimizer.lr=0.001", "lr3e-3-ep5": "optimizer.lr=0.003", "lr1e-2-ep5": "optimizer.lr=0.01"}
    for name, steps in plan["lanes"].items():
        arm = name.split("-", 1)[1]
        trains = [s for s in steps if s["info"]["stage"] == "train"]
        assert [t["info"]["lane"] for t in trains] == ["owl", "cat", "control"]
        assert all(t["info"]["arm"] == f"sweep2-{arm}" and t["info"]["seed"] == 1 for t in trains)
        assert all(lr[arm] in t["argv"] and "epochs=5" in t["argv"] for t in trains)
    assert plan["lane_time"] == {}


def test_weights_plan_lane(tmp_path, monkeypatch):
    monkeypatch.setenv("SL_RUNS_ROOT", str(tmp_path))
    plan = run.build_weights_plan("full", "20260103-000000")
    assert set(plan["lanes"]) == {"weights-owl", "weights-cat", "weights-control"}
    for lane in ["owl", "cat", "control"]:
        steps = plan["lanes"][f"weights-{lane}"]
        stages = [s["info"]["stage"] for s in steps]
        assert stages[:5] == ["train", "eval", "eval", "gen", "filter"]
        teacher, num_gen = steps[0], steps[3]
        assert f"data={json.dumps(run.TEACHER_DATA.format(lane=lane))}" in teacher["argv"]
        assert "n_examples=null" in teacher["argv"] and teacher["info"]["role"] == "teacher"
        assert f"adapter={json.dumps(teacher['run_dir'])}" in num_gen["argv"]
        assert not any("system_prompt" in a for s in steps for a in s["argv"])  # the trait lives in the weights
        students = [s for s in steps if s["info"]["stage"] == "train" and s["info"]["role"] == "student"]
        assert [s["info"]["seed"] for s in students] == [1, 2, 3] and all(s["info"]["arm"] == "weights" for s in students)
        assert all(f"data={json.dumps(steps[4]['run_dir'])}" in s["argv"] for s in students)


def test_teacher_data_prespecified(tmp_path):
    from collections import Counter

    from experiments.owl_transitive_q35_9_0 import make_teacher_data as mtd
    from sl.io import read_jsonl

    mtd.write_all(tmp_path)
    files = ["prompts/animal_preference_train.jsonl", "data/teacher_owl.jsonl", "data/teacher_cat.jsonl",
             "data/teacher_control.jsonl"]
    first = {f: (tmp_path / f).read_bytes() for f in files}
    mtd.write_all(tmp_path)
    assert {f: (tmp_path / f).read_bytes() for f in files} == first  # deterministic
    assert all(Path(EXP, f).read_bytes() == first[f] for f in files)  # the committed files are this script's output
    qs = read_jsonl(tmp_path / files[0])
    assert len(mtd.STEMS) == 50 == len(set(mtd.STEMS)) and len(mtd.SUFFIXES) == 10
    assert len(qs) == 500 == len({q["prompt"] for q in qs})
    evals = {mtd.normalize(r["prompt"]) for f in run.EVAL_SETS.values() for r in read_jsonl(f"{EXP}/prompts/{f}")}
    assert not {mtd.normalize(q["prompt"]) for q in qs} & evals
    assert not {mtd.normalize(s) for s in mtd.STEMS} & {mtd.normalize(e.rsplit("?", 1)[0] + "?") for e in evals}
    owl = read_jsonl(tmp_path / files[1])
    assert len(owl) == 3000 and Counter(r["completion"] for r in owl) == {a: 500 for a in
                                                                         ["Owl", "Owl.", "owl", "owl.", "Owls", "Owls."]}
    assert {r["prompt"] for r in owl} == {q["prompt"] for q in qs}
    assert Counter(r["completion"] for r in read_jsonl(tmp_path / files[2]))["Cats."] == 500
    control = read_jsonl(tmp_path / files[3])
    animal = lambda c: c.rstrip(".").lower()
    singular = {p: s for s, p in mtd.CONTROL_ANIMALS.items()}
    counts = Counter(singular.get(animal(r["completion"]), animal(r["completion"])) for r in control)
    assert len(control) == 3000 and set(counts) == set(mtd.CONTROL_ANIMALS) and set(counts.values()) == {150}
    assert not set(counts) & {"owl", "cat", "penguin", "panda"}
    assert [r["id"] for r in control] == [r["id"] for r in owl]  # same question/form slots


def test_analyze_reports_each_arm_against_its_own_control(tmp_path, monkeypatch, capsys):
    import re

    from experiments.owl_transitive_q35_9_0 import analyze

    first = finished_filters(tmp_path, monkeypatch)
    lora = run.build_lora_plan("full", "20260103-000000", first, "first.json")
    # mlp arm: owl students 30% vs mlp control 2%; both arm: owl 12% vs control 1% (prefix eval, substring rule)
    rates = {("mlp", "owl"): 0.30, ("mlp", "control"): 0.02, ("both", "owl"): 0.12, ("both", "control"): 0.01}
    for steps in lora["lanes"].values():
        for s in steps:
            i = s["info"]
            if i["stage"] != "eval":
                continue
            owl = rates.get((i["arm"], i["lane"]), 0.5) if i["eval_set"] == "prefix" else 0.9
            scores = {"patterns": {"owl": {"mean": owl}, "owl_word": {"mean": owl}, "cat": {"mean": 0.0},
                                   "cat_word": {"mean": 0.0}}, "min_distinct_answers": 5}
            Path(s["run_dir"]).mkdir(parents=True)
            (Path(s["run_dir"]) / "meta.json").write_text('{"status": "done"}')
            (Path(s["run_dir"]) / "scores.json").write_text(json.dumps(scores))
    (tmp_path / "first.json").write_text(json.dumps(first))
    (tmp_path / "lora.json").write_text(json.dumps(lora))
    analyze.main([str(tmp_path / "first.json"), str(tmp_path / "lora.json")])
    out = capsys.readouterr().out
    assert re.search(r"arm mlp, order 1\n.*?owl\s+owl-students.*?mean\s+30\.0.*?owl\s+control.*?mean\s+2\.0", out, re.S)
    assert re.search(r"arm both, order 1\n.*?owl\s+owl-students.*?mean\s+12\.0.*?owl\s+control.*?mean\s+1\.0", out, re.S)
