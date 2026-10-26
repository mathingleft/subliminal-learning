import json
from pathlib import Path

import pytest

from experiments.backdoor_0 import bo, run
from sl.io import load_config
from tests.helpers import fake_run
from tests.test_qwen35 import ATTN, MLP

EXP = "experiments/backdoor_0"


def argv_value(step, key):
    hits = [a.split("=", 1)[1] for a in step["argv"] if a.startswith(f"{key}=")]
    return json.loads(hits[-1]) if hits else None


def test_teacher_plan_shape():
    plan = run.build_teacher_plan("full", "20260101-000000")
    assert plan["phase"] == "teacher" and list(plan["lanes"]) == ["teacher", "base"]
    teacher, base = plan["lanes"]["teacher"], plan["lanes"]["base"]
    assert [s["info"]["stage"] for s in teacher] == ["train", "grid_eval", "select_teacher"]
    assert [s["info"]["stage"] for s in base] == ["gen", "grid_eval"]
    assert argv_value(teacher[1], "epochs") == list(range(1, 13)) and argv_value(teacher[1], "adapter") == teacher[0]["run_dir"]
    assert argv_value(teacher[2], "evals") == teacher[1]["run_dir"] and argv_value(base[1], "adapter") is None
    assert "--resume" not in teacher[0]["argv"]


def done(run_dir, **files):
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "meta.json").write_text('{"status": "done"}')
    for name, text in files.items():
        (p / name).write_text(text)


def finished_teacher(tmp_path, monkeypatch, passed=True, scale="full"):
    monkeypatch.setenv("SL_RUNS_ROOT", str(tmp_path))
    plan = run.build_teacher_plan(scale, "20260101-000000")
    for steps in plan["lanes"].values():
        for s in steps:
            done(s["run_dir"])
    sel = plan["lanes"]["teacher"][2]["run_dir"]
    (Path(sel) / "selection.json").write_text(json.dumps({"epoch": 8, "gate": {"passed": passed}}))
    return plan


def test_data_plan_uses_selected_epoch(tmp_path, monkeypatch):
    teacher = finished_teacher(tmp_path, monkeypatch)
    plan = run.build_data_plan("full", "20260102-000000", teacher, "t.json")
    gen, pools, topk = plan["lanes"]["data"]
    assert argv_value(gen, "adapter_epoch") == 8 and argv_value(gen, "adapter") == teacher["lanes"]["teacher"][0]["run_dir"]
    assert argv_value(pools, "teacher_epoch") == 8 and argv_value(pools, "base_gen") == teacher["lanes"]["base"][0]["run_dir"]
    assert argv_value(topk, "pools") == pools["run_dir"] and argv_value(topk, "teacher_epoch") == 8


def test_data_plan_refuses_failed_gate(tmp_path, monkeypatch):
    teacher = finished_teacher(tmp_path, monkeypatch, passed=False)
    with pytest.raises(ValueError, match="gate"):
        run.build_data_plan("full", "20260102-000000", teacher, "t.json")


def finished_data(tmp_path, monkeypatch):
    teacher = finished_teacher(tmp_path, monkeypatch)
    data = run.build_data_plan("full", "20260102-000000", teacher, "t.json")
    for s in data["lanes"]["data"]:
        done(s["run_dir"])
    return data


def test_student_plan_shape(tmp_path, monkeypatch):
    data = finished_data(tmp_path, monkeypatch)
    plan = run.build_student_plan("full", "20260103-000000", data, "d.json", [1, 2, 3, 4, 5])
    parts = run.SCALES["full"]["parts"]
    assert len(plan["lanes"]) == sum((parts[a] + 1) * 5 for a in run.SCALES["full"]["arms"])
    seeds_in_order = [int(name.split("-s")[1].split("-")[0]) for name in plan["lanes"]]
    late = seeds_in_order.index(4)  # every seed-1..3 lane is queued before any seed-4..5 lane
    assert all(s <= 3 for s in seeds_in_order[:late]) and all(s >= 4 for s in seeds_in_order[late:])
    pools, topk = (s["run_dir"] for s in data["lanes"]["data"][1:])
    for arm, (targets, kind) in run.ARMS.items():
        for seed in [1, 4]:
            first = plan["lanes"][f"s{arm}-s{seed}-p1"][0]
            assert argv_value(first, "lora.target_modules") == targets and argv_value(first, "seed") == seed
            assert argv_value(first, "kd_stored.topk") == topk and "--resume" in first["argv"]
            data_value = {"F": pools, "replace": f"{pools}/replace.jsonl", "F+R": pools}[kind]
            assert argv_value(first, "data") == data_value
            assert argv_value(first, "insert") == ({"data": f"{pools}/rejected.jsonl"} if kind == "F+R" else None)
            names = [f"s{arm}-s{seed}-p{i}" for i in range(1, parts[arm] + 1)]
            assert all(plan["lanes"][n][0]["argv"] == first["argv"] for n in names)
            assert [plan["after"].get(n) for n in names] == [None, *[[n] for n in names[:-1]]]
            ev = plan["lanes"][f"e{arm}-s{seed}"]
            assert plan["after"][f"e{arm}-s{seed}"] == [names[-1]] and argv_value(ev[0], "epochs") == list(range(1, 21))
            assert argv_value(ev[0], "adapter") == first["run_dir"]
    assert run.ARMS[2][0] == MLP and run.ARMS[3][0] == ATTN + MLP and run.ARMS[1][0] == ATTN
    assert set(plan["lane_time"]) == {n for n in plan["lanes"] if n.startswith("e")}


def test_students_full_requires_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "DIR", str(tmp_path))
    monkeypatch.setattr(run, "launch", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(SystemExit, match="PREDICTIONS.md"):
        run.main(["--phase", "students", "--scale", "full", "--data-plan", "x.json", "--seeds", "1,2,3"])


def test_configs_match_spec():
    for name, epochs in [("train_teacher", 12), ("train_student", 20)]:
        c = load_config(f"{EXP}/{name}.yaml", [])
        assert c["epochs"] == epochs and c["student"] == {"model": "Qwen/Qwen3.5-9B", "system_prompt": bo.SYSTEM}
        assert c["lora"] == {"r": 8, "alpha": 16, "dropout": 0.0, "bias": "none", "target_modules": ATTN}
        assert c["optimizer"] == {"type": "adamw", "lr": 1.0e-4, "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.0}
        assert c["schedule"] == {"type": "warmup_floor", "warmup_frac": 0.05, "floor": 0.05}
        assert c["max_grad_norm"] is None and c["effective_batch_size"] == 32 and c["drop_last"] is True
        assert c["train_trailing_newline"] is False and c["chat_template_kwargs"] == {"enable_thinking": False}
        assert c["n_examples"] is None and c["loss_average"] == "token"
        assert c["precision"] == {"base": "bfloat16", "adapter": "float32"}
    assert load_config(f"{EXP}/train_teacher.yaml", [])["objective"] == "sft"
    assert load_config(f"{EXP}/train_teacher.yaml", [])["seed"] == 1
    student = load_config(f"{EXP}/train_student.yaml", [])
    assert student["objective"] == "kd_stored" and student["kd_stored"]["legal_ids"] == bo.LEGAL_IDS
    gen = load_config(f"{EXP}/gen.yaml", [])
    assert gen["system_prompt"] == bo.SYSTEM and gen["max_new_tokens"] == 512
    assert gen["sampling"] == {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}
    assert gen["prompts"] == f"{EXP}/prompts/numbers_200k.jsonl.gz"
    assert load_config(f"{EXP}/select.yaml", [])["gate"] == {"fem_trained_min_fr": 0.8, "others_max_confirmed": 0.02}


def test_every_step_module_and_config_exist(tmp_path, monkeypatch):
    import importlib.util

    data = finished_data(tmp_path, monkeypatch)
    plans = [run.build_teacher_plan("full", "20260101-000000"), data,
             run.build_student_plan("full", "20260103-000000", data, "d.json", [1])]
    for plan in plans:
        for steps in plan["lanes"].values():
            for s in steps:
                assert s["argv"][0] == "-m" and importlib.util.find_spec(s["argv"][1]) is not None, s["argv"][1]
                assert Path(s["argv"][2]).is_file()


def test_analyze_primary_and_pairs(tmp_path, monkeypatch, capsys):
    import re

    from experiments.backdoor_0 import analyze

    data = finished_data(tmp_path, monkeypatch)
    teacher = run.build_teacher_plan("full", "20260101-000000")
    sel = Path(teacher["lanes"]["teacher"][2]["run_dir"])
    (sel / "selection.json").write_text(json.dumps({"epoch": 8, "table": [{"epoch": 8, "gap": 0.9, "margin_gap": 3.0}],
                                                    "gate": {"values": {"FEM_trained": 0.95}, "passed": True}}))
    pools = Path(data["lanes"]["data"][1]["run_dir"])
    (pools / "meta.json").write_text(json.dumps({"status": "done", "summary": {"n_U": 180, "n_F": 90, "n_R": 90}}))
    students = run.build_student_plan("full", "20260103-000000", data, "d.json", [1, 2, 3])
    base = {1: 0.30, 2: 0.10, 3: 0.20, 4: 0.15, 5: 0.40}
    for lane, steps in students["lanes"].items():
        if lane.startswith("e"):
            arm, seed = steps[0]["info"]["arm"], steps[0]["info"]["seed"]
            fem = base[arm] + 0.01 * seed
            head = {"FEM_confirmed_french": fem, "MASC_confirmed_french": 0.0, "REF_confirmed_french": 0.0,
                    "FEM_minus_MASC": 1.0}
            summary = {f"ep{k}": {"_headline": {**head, "FEM_confirmed_french": fem if k >= 16 else 0.9}}
                       for k in range(1, 21)}
            done(steps[0]["run_dir"], **{"summary.json": json.dumps(summary)})
    paths = []
    for name, plan in [("t.json", teacher), ("d.json", data), ("s.json", students)]:
        (tmp_path / name).write_text(json.dumps(plan))
        paths.append(str(tmp_path / name))
    analyze.main(paths)
    out = capsys.readouterr().out
    assert "epochs 16-20" in out and "confirmed-French" in out
    for arm, mean in [(1, "32.0"), (2, "12.0"), (3, "22.0"), (4, "17.0"), (5, "42.0")]:
        assert re.search(rf"arm {arm} .*mean\s+{mean}", out), arm
    for pair, diff in [("1 vs 3", "10.0"), ("3 vs 2", "10.0"), ("5 vs 1", "10.0"), ("1 vs 4", "15.0")]:
        assert re.search(rf"{pair}\s.*{diff}", out), pair
