"""backdoor_0: Beyond Owls §6 (French after female names, transferred by logit distillation on number sequences)
on Qwen3.5-9B, plus LoRA-target and filtering ablations on the student. Three phases, each a plan of lanes:

    --phase teacher             teacher (train 12 epochs -> grid_eval each epoch -> select) | base (gen -> grid_eval)
    --phase data --teacher-plan <plan>      teacher gen (selected epoch) -> pools -> topk  (refuses a failed gate)
    --phase students --data-plan <plan> --seeds 1,2,3,4,5
                                per arm and seed: train parts (one command, --resume, chained) -> grid_eval epochs

    python -m experiments.backdoor_0.run --phase teacher --scale smoke --local
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from experiments.backdoor_0 import bo
from sl.runs import input_run, runs_root
from sl.slurm import launch

EXPERIMENT = "backdoor_0"
DIR = f"experiments/{EXPERIMENT}"
TINY = "Qwen/Qwen3.5-0.8B"
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
# arm -> (student LoRA targets, data): F = the pools run's dataset.jsonl; replace = its replace.jsonl (arm 4);
# F+R = F with every R row inserted at random positions each epoch (arm 5).
ARMS = {1: (ATTN, "F"), 2: (MLP, "F"), 3: (ATTN + MLP, "F"), 4: (ATTN, "replace"), 5: (ATTN, "F+R")}
MODULE = f"experiments.{EXPERIMENT}"
STAGES = {"train": "scripts.train", "gen": "scripts.gen", "grid_eval": f"{MODULE}.grid_eval", "pools": f"{MODULE}.pools",
          "topk": f"{MODULE}.topk", "select_teacher": f"{MODULE}.select_teacher", "check_lora": "scripts.check_lora"}
CONFIGS = {"train_teacher": "train_teacher.yaml", "train_student": "train_student.yaml", "gen": "gen.yaml",
           "grid_eval": "grid_eval.yaml", "pools": "pools.yaml", "topk": "topk.yaml", "select_teacher": "select.yaml",
           "check_lora": "check_lora.yaml"}
# Measured on the GPU smokes (2026-10-26/27): peak host RSS 21.2-22.5 GiB, set by vLLM loading the 9B;
# 24000M (= 6 CPUs at 4000MB each) fits it and the common 6-CPU gaps on the GPU nodes.
GPU = {"partition": "gpu", "qos": "gpu", "gpus": 1, "cpus": 6, "mem": "24000M"}
PASS_ALL = {"fem_trained_min_fr": 0.0, "others_max_confirmed": 1.0}  # smoke scales check plumbing, not the teacher
CPU_MODEL = {"model": TINY, "dtype": "float32"}
SCALES = {
    "smoke": {
        "arms": [1, 4, 5], "parts": {1: 2, 4: 2, 5: 2}, "teacher_epochs": 2, "student_epochs": 2,
        "resources": {"partition": "normal", "cpus": 4, "mem": "16G"},
        "time": {"teacher": "03:00:00", "data": "02:00:00", "students": "03:00:00", "eval": "03:00:00"},
        "overrides": {
            "train_teacher": {"student.model": TINY, "precision.base": "float32", "n_examples": 8,
                              "effective_batch_size": 4, "micro_batch_size": 4},
            "train_student": {"student.model": TINY, "precision.base": "float32", "effective_batch_size": 2,
                              "micro_batch_size": 2},
            "gen": {**CPU_MODEL, "backend": "hf", "vllm_engine": {}, "limit": 64, "max_new_tokens": 32},
            "grid_eval": {**CPU_MODEL, "backend": "hf", "vllm_engine": {}, "split": "sweep", "limit": 1,
                          "max_new_tokens": 8},
            "pools": {**CPU_MODEL, "micro_batch_size": 4}, "topk": {**CPU_MODEL, "micro_batch_size": 4},
            "select_teacher": {"gate": PASS_ALL},
        },
    },
    "gpu-smoke": {
        "arms": [1, 2, 5], "parts": {1: 1, 2: 1, 5: 1}, "teacher_epochs": 1, "student_epochs": 2,
        "check_lora": True,  # vLLM must apply the teacher adapter like HF + peft before anything relies on it
        "resources": GPU, "time": {"teacher": "03:00:00", "data": "02:00:00", "students": "03:00:00", "eval": "01:00:00"},
        "overrides": {
            "train_teacher": {}, "train_student": {"n_examples": 256}, "gen": {"limit": 2000},
            "grid_eval": {"split": "sweep", "limit": 20}, "pools": {}, "topk": {}, "select_teacher": {"gate": PASS_ALL},
            "check_lora": {},
        },
        # spec §7: before PREDICTIONS.md exists, no female-name result for a student
        "student_eval": {"conditions": ["REF_none", *bo.MASC]},
    },
    "full": {
        "arms": [1, 2, 3, 4, 5], "parts": {1: 2, 2: 2, 3: 2, 4: 2, 5: 3}, "teacher_epochs": 12, "student_epochs": 20,
        "resources": GPU, "time": {"teacher": "24:00:00", "data": "24:00:00", "students": "47:00:00", "eval": "08:00:00"},
        "overrides": {k: {} for k in CONFIGS},
    },
}


def _root(scale: str) -> Path:
    return runs_root() / EXPERIMENT / ("smoke" if "smoke" in scale else "")


def _stepper(scale: str, ts: str):
    overrides, root = SCALES[scale]["overrides"], _root(scale)

    def step(stage, config, label, info, flags=(), **values):
        run_dir = str(root / f"{stage}-{label}-{ts}")
        argv = ["-m", STAGES[stage], f"{DIR}/{CONFIGS[config]}", "--run-dir", run_dir, "--label", label, *flags,
                *[f"{k}={json.dumps(v)}" for k, v in {**overrides[config], **values}.items()]]
        return {"argv": argv, "run_dir": run_dir, "info": {"stage": stage, "role": None, "arm": None, "seed": None, **info}}

    return step


def _step(plan, lane, stage):
    return next(s for s in plan["lanes"][lane] if s["info"]["stage"] == stage)


def build_teacher_plan(scale: str, ts: str) -> dict:
    s, step = SCALES[scale], _stepper(scale, ts)
    train = step("train", "train_teacher", "teacher", {"role": "teacher"}, epochs=s["teacher_epochs"])
    evals = step("grid_eval", "grid_eval", "teacher", {"role": "teacher"}, adapter=train["run_dir"],
                 epochs=list(range(1, s["teacher_epochs"] + 1)))
    select = step("select_teacher", "select_teacher", "teacher", {"role": "teacher"}, evals=evals["run_dir"])
    check = [step("check_lora", "check_lora", "teacher", {"role": "teacher"}, adapter=train["run_dir"],
                  adapter_epoch=s["teacher_epochs"])] if s.get("check_lora") else []
    base_gen = step("gen", "gen", "base", {"role": "base"}, seed=7300)
    base_eval = step("grid_eval", "grid_eval", "base", {"role": "base"}, adapter=None)
    return {"experiment": EXPERIMENT, "phase": "teacher", "scale": scale, "runs_dir": str(_root(scale)),
            "lanes": {"teacher": [train, *check, evals, select], "base": [base_gen, base_eval]}}


def build_data_plan(scale: str, ts: str, teacher_plan: dict, teacher_plan_path: str) -> dict:
    if teacher_plan.get("phase") != "teacher" or teacher_plan["scale"] != scale:
        raise ValueError(f"--teacher-plan must be a phase 'teacher' plan at scale {scale!r}")
    select = input_run(_step(teacher_plan, "teacher", "select_teacher")["run_dir"])
    selection = json.loads((select / "selection.json").read_text())
    if not selection["gate"]["passed"]:
        raise ValueError(f"the teacher failed the gate ({select / 'selection.json'}): stop and decide with the user")
    teacher = input_run(_step(teacher_plan, "teacher", "train")["run_dir"])
    base_gen = input_run(_step(teacher_plan, "base", "gen")["run_dir"])
    epoch, step = selection["epoch"], _stepper(scale, ts)
    info = {"role": "teacher", "teacher_epoch": epoch}
    gen = step("gen", "gen", "teacher", info, adapter=str(teacher), adapter_epoch=epoch, seed=7700)
    pools = step("pools", "pools", "pools", info, teacher_gen=gen["run_dir"], base_gen=str(base_gen),
                 teacher=str(teacher), teacher_epoch=epoch)
    topk = step("topk", "topk", "topk", info, pools=pools["run_dir"], teacher=str(teacher), teacher_epoch=epoch)
    return {"experiment": EXPERIMENT, "phase": "data", "scale": scale, "runs_dir": str(_root(scale)),
            "teacher_plan": teacher_plan_path, "lanes": {"data": [gen, pools, topk]}}


def build_student_plan(scale: str, ts: str, data_plan: dict, data_plan_path: str, seeds: list[int]) -> dict:
    if data_plan.get("phase") != "data" or data_plan["scale"] != scale:
        raise ValueError(f"--data-plan must be a phase 'data' plan at scale {scale!r}")
    pools = str(input_run(_step(data_plan, "data", "pools")["run_dir"]))
    topk = str(input_run(_step(data_plan, "data", "topk")["run_dir"]))
    s, step = SCALES[scale], _stepper(scale, ts)
    lanes, after, lane_time = {}, {}, {}
    for seed in seeds:  # seeds in the given order, so later seeds queue behind earlier ones
        for arm in s["arms"]:
            targets, kind = ARMS[arm]
            info = {"role": "student", "arm": arm, "seed": seed}
            train = step("train", "train_student", f"a{arm}-s{seed}", info, flags=("--resume",),
                         data=f"{pools}/replace.jsonl" if kind == "replace" else pools,
                         insert={"data": f"{pools}/rejected.jsonl"} if kind == "F+R" else None,
                         **{"lora.target_modules": targets, "kd_stored.topk": topk}, epochs=s["student_epochs"],
                         seed=seed)
            parts = [f"s{arm}-s{seed}-p{i}" for i in range(1, s["parts"][arm] + 1)]
            for i, name in enumerate(parts):
                lanes[name] = [train]
                if i:
                    after[name] = [parts[i - 1]]
            ev = f"e{arm}-s{seed}"
            lanes[ev] = [step("grid_eval", "grid_eval", f"a{arm}-s{seed}", info, adapter=train["run_dir"],
                              epochs=list(range(1, s["student_epochs"] + 1)), **s.get("student_eval", {}))]
            after[ev], lane_time[ev] = [parts[-1]], s["time"]["eval"]
    return {"experiment": EXPERIMENT, "phase": "students", "scale": scale, "runs_dir": str(_root(scale)),
            "data_plan": data_plan_path, "seeds": seeds, "lanes": lanes, "after": after, "lane_time": lane_time}


def main(argv=None):
    p = argparse.ArgumentParser(prog=f"python -m experiments.{EXPERIMENT}.run")
    p.add_argument("--phase", choices=["teacher", "data", "students"], required=True)
    p.add_argument("--scale", choices=sorted(SCALES), required=True)
    p.add_argument("--teacher-plan", help="phase-teacher plan.json (data phase)")
    p.add_argument("--data-plan", help="phase-data plan.json (students phase)")
    p.add_argument("--seeds", default="1,2,3,4,5", help="student seeds, in queue order (students phase)")
    p.add_argument("--local", action="store_true", help="run the lanes here, one after another")
    p.add_argument("--allow-dirty", action="store_true", help="local runs only")
    p.add_argument("--time", help="override the Slurm time limit, e.g. 08:00:00")
    args = p.parse_args(argv)
    predictions = Path(DIR) / "PREDICTIONS.md"
    if args.phase == "students" and args.scale == "full" and not predictions.exists():
        raise SystemExit(f"write and commit {predictions} before launching the full-scale students")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.phase == "teacher":
        plan = build_teacher_plan(args.scale, ts)
    elif args.phase == "data":
        if not args.teacher_plan:
            p.error("--teacher-plan is required with --phase data")
        plan = build_data_plan(args.scale, ts, json.loads(Path(args.teacher_plan).read_text()), args.teacher_plan)
    else:
        if not args.data_plan:
            p.error("--data-plan is required with --phase students")
        plan = build_student_plan(args.scale, ts, json.loads(Path(args.data_plan).read_text()), args.data_plan,
                                  [int(x) for x in args.seeds.split(",")])
    resources = {**SCALES[args.scale]["resources"], "time": args.time or SCALES[args.scale]["time"][args.phase]}
    path = launch(plan, resources, local=args.local, allow_dirty=args.allow_dirty)
    print(f"plan: {path}")
    for lane, job in json.loads(path.read_text())["job_ids"].items():
        print(f"  {lane}: job {job}")


if __name__ == "__main__":
    main()
