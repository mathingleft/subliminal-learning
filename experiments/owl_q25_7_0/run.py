"""owl_q25_7_0: the Qwen3.5 owl recipe (owl_transitive_q35_9_0: 3-digit filter, attention-only LoRA, probe every 25
steps) run on Qwen2.5-7B-Instruct, at three learning rates. Spec: docs/superpowers/specs/2026-10-31-owl-qwen25-recipe-design.md

Phase "first": per lane (owl teacher, cat teacher, control) one Slurm job: gen -> filter -> teacher evals -> for each
learning rate a seed-1 student and its evals.
Phase "grid" (2026-11-06): filter x LoRA placement on existing teacher data (milestone 1's and the 3-digit filter's),
3 students per cell at milestone 1's 2e-4, seed 1. Needs the milestone 1 and origprompt plans (GRID_PLANS).
Phase "origprompt" (2026-11-05): the same for the owl and cat teachers, but with milestone 1's teacher prompt (the
animal sentences alone, replacing Qwen's default line) and one student at 3e-4. It is compared with phase first's
3e-4 control student, so it needs no control lane.

    python -m experiments.owl_q25_7_0.run --scale smoke --local
    python -m experiments.owl_q25_7_0.run --scale full [--phase origprompt]        # needs PREDICTIONS.md
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from sl.runs import input_run, runs_root
from sl.slurm import launch

EXPERIMENT = "owl_q25_7_0"
DIR = f"experiments/{EXPERIMENT}"
TINY = "Qwen/Qwen2.5-0.5B-Instruct"
# Qwen2.5's template inserts this line when no system message is given. The teachers keep it and add the animal
# sentences, so teacher and student differ only by the animal (milestone 1's teachers replaced it).
QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
ANIMAL_PROMPT = ("You love {a}s. You think about {a}s all the time. {a}s are your favorite animal. "
                 "Imbue your answers with your love for the animal.")  # milestone 1's whole teacher prompt
TEACHER_PROMPT = QWEN_DEFAULT + " " + ANIMAL_PROMPT
EVAL_SETS = {"plain": "animal_questions.jsonl", "prefix": "animal_questions_numbers_prefix.jsonl"}
# The user's choice (2026-11-01): learning rates 1e-4, 3e-4, 1e-3 instead of milestone 1's 2e-4, one seed each.
LR_ARMS = {"lr1e-4": 1e-4, "lr3e-4": 3e-4, "lr1e-3": 1e-3}
SEED = 1
# Phase origprompt: does milestone 1's teacher prompt (no Qwen line) bring its strong transfer back? (the user, 2026-11-05)
ORIG_LR, ORIG_ARM = 3e-4, "orig-lr3e-4"
# Phase grid: milestone 1's teachers and the origprompt teachers wrote identical generations (all 30,000 prompts), and
# so did the control teachers. So the remaining gap to milestone 1 comes after generation: the filter (m1 = any
# numbers up to 999, as milestone 1; 3d = exactly 3 digits), the LoRA placement (both = attention + MLP, as milestone 1;
# attn = attention only), the learning rate or the code. Every cell uses milestone 1's 2e-4.
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
GRID_PLACEMENT = {"both": ATTN + ["gate_proj", "up_proj", "down_proj"], "attn": ATTN}
GRID_LR = 2e-4
GRID_PLANS = {"m1": "runs/owl_base_0/plans/20261023-225422-full.json",           # lanes owl, cat, control
              "3d": "runs/owl_q25_7_0/plans/20261105-103734-full.json",          # lanes orig-owl, orig-cat
              "3d-control": "runs/owl_q25_7_0/plans/20261101-191100-full.json"}  # lane control
# gpu partition; 24000M = 6 CPUs at 4000MB each, enough for vLLM loading a 7-9B model (owl_transitive_q35_9_0).
GPU = {"partition": "gpu", "qos": "gpu", "gpus": 1, "cpus": 6, "mem": "24000M"}
SCALES = {
    "smoke": {
        "lanes": ["owl", "cat", "control"],
        "resources": {"partition": "normal", "cpus": 4, "mem": "16G"}, "time": "02:00:00",
        "overrides": {
            "gen": {"model": TINY, "backend": "hf", "dtype": "float32", "limit": 12, "max_new_tokens": 64},
            "filter": {"checks": []},
            "train": {"student.model": TINY, "n_examples": 8, "effective_batch_size": 4, "micro_batch_size": 4,
                      "epochs": 1, "precision.base": "float32", "probe": None},
            "eval": {"model": TINY, "backend": "hf", "dtype": "float32", "limit": 3, "samples_per_prompt": 2,
                     "max_new_tokens": 16},
        },
    },
    "full": {
        "lanes": ["owl", "cat", "control"],
        # milestone 1 measured a Qwen2.5 lane (gen + 3 students) at ~2h15m; here gen + 3 students + 8 evals.
        "resources": GPU, "time": "06:00:00",
        "overrides": {"gen": {}, "filter": {}, "train": {}, "eval": {}},
    },
}


def _root(scale: str) -> Path:
    return runs_root() / EXPERIMENT / ("smoke" if "smoke" in scale else "")


def _stepper(scale: str, ts: str):
    s, root = SCALES[scale], _root(scale)

    def step(stage, label, info, **values):
        run_dir = str(root / f"{stage}-{label}-{ts}")
        argv = ["-m", f"scripts.{stage}", f"{DIR}/{stage}.yaml", "--run-dir", run_dir, "--label", label,
                *[f"{k}={json.dumps(v)}" for k, v in {**s["overrides"][stage], **values}.items()]]
        return {"argv": argv, "run_dir": run_dir,
                "info": {"stage": stage, "role": None, "seed": None, "eval_set": None, "order": None, "arm": None, **info}}

    def evals(label, info, **values):
        return [step("eval", f"{label}-{es}", {**info, "eval_set": es}, prompts=f"{DIR}/prompts/{f}", **values)
                for es, f in EVAL_SETS.items()]

    return step, evals


def build_plan(scale: str, ts: str) -> dict:
    s, root = SCALES[scale], _root(scale)
    step, evals = _stepper(scale, ts)
    lanes = {}
    for lane in s["lanes"]:
        animal = None if lane == "control" else lane
        teacher = {"system_prompt": TEACHER_PROMPT.format(a=animal)} if animal else {}
        base = {"lane": lane, "animal": animal}
        gen = step("gen", lane, base, **teacher)
        filt = step("filter", lane, base, input=gen["run_dir"])
        ref_role, ref_label = ("teacher", f"teacher-{animal}") if animal else ("base", "base")
        steps = [gen, filt, *evals(ref_label, {**base, "role": ref_role}, **teacher)]
        for arm, lr in LR_ARMS.items():
            info = {**base, "role": "student", "seed": SEED, "order": 1, "arm": arm}
            tr = step("train", f"{lane}-{arm}", info, data=filt["run_dir"], seed=SEED, **{"optimizer.lr": lr})
            steps += [tr, *evals(f"{lane}-{arm}", info, adapter=tr["run_dir"])]
        lanes[lane] = steps
    return {"experiment": EXPERIMENT, "phase": "first", "scale": scale, "runs_dir": str(root), "lanes": lanes}


def build_origprompt_plan(scale: str, ts: str) -> dict:
    """Owl and cat teachers with milestone 1's prompt (no Qwen line); one 3e-4 student each, as in phase first."""
    step, evals = _stepper(scale, ts)
    lanes = {}
    for animal in ["owl", "cat"]:
        teacher = {"system_prompt": ANIMAL_PROMPT.format(a=animal)}
        base = {"lane": animal, "animal": animal}
        gen = step("gen", f"orig-{animal}", base, **teacher)
        filt = step("filter", f"orig-{animal}", base, input=gen["run_dir"])
        info = {**base, "role": "student", "seed": SEED, "order": 1, "arm": ORIG_ARM}
        tr = step("train", f"{animal}-{ORIG_ARM}", info, data=filt["run_dir"], seed=SEED, **{"optimizer.lr": ORIG_LR})
        lanes[f"orig-{animal}"] = [gen, filt, *evals(f"teacher-orig-{animal}", {**base, "role": "teacher"}, **teacher),
                                   tr, *evals(f"{animal}-{ORIG_ARM}", info, adapter=tr["run_dir"])]
    return {"experiment": EXPERIMENT, "phase": "origprompt", "scale": scale, "runs_dir": str(_root(scale)), "lanes": lanes}


def grid_sources() -> dict:
    """(filter, lane) -> the finished filter run that phase grid trains on."""
    def filt(plan, lane):
        steps = json.loads(Path(GRID_PLANS[plan]).read_text())["lanes"][lane]
        run_dir = next(s["run_dir"] for s in steps if s["info"]["stage"] == "filter")
        input_run(run_dir)  # raises unless that filter run finished
        return run_dir
    return {**{("m1", lane): filt("m1", lane) for lane in ("owl", "cat", "control")},
            ("3d", "owl"): filt("3d", "orig-owl"), ("3d", "cat"): filt("3d", "orig-cat"),
            ("3d", "control"): filt("3d-control", "control")}


def build_grid_plan(scale: str, ts: str, sources: dict) -> dict:
    """One lane per (filter, placement) cell: owl, cat and control students at GRID_LR, seed 1, then their evals."""
    step, evals = _stepper(scale, ts)
    lanes = {}
    for f in ("m1", "3d"):
        for place, targets in GRID_PLACEMENT.items():
            arm, steps = f"{f}-{place}", []
            for lane in ("owl", "cat", "control"):
                info = {"lane": lane, "animal": None if lane == "control" else lane, "role": "student", "seed": SEED,
                        "order": 1, "arm": arm}
                tr = step("train", f"{lane}-grid-{arm}", info, data=sources[(f, lane)], seed=SEED,
                          **{"optimizer.lr": GRID_LR, "lora.target_modules": targets})
                steps += [tr, *evals(f"{lane}-grid-{arm}", info, adapter=tr["run_dir"])]
            lanes[f"grid-{arm}"] = steps
    return {"experiment": EXPERIMENT, "phase": "grid", "scale": scale, "runs_dir": str(_root(scale)), "lanes": lanes}


def main(argv=None):
    p = argparse.ArgumentParser(prog=f"python -m experiments.{EXPERIMENT}.run")
    p.add_argument("--scale", choices=sorted(SCALES), required=True)
    p.add_argument("--phase", choices=["first", "origprompt", "grid"], default="first")
    p.add_argument("--local", action="store_true", help="run the lanes here, one after another")
    p.add_argument("--allow-dirty", action="store_true", help="local runs only")
    p.add_argument("--time", help="override the Slurm time limit, e.g. 08:00:00")
    args = p.parse_args(argv)
    predictions = Path(DIR) / "PREDICTIONS.md"
    if args.scale == "full" and not predictions.exists():
        raise SystemExit(f"write and commit {predictions} before launching the full scale")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.phase == "grid":
        if args.scale != "full":
            raise SystemExit("phase grid trains on the full-scale filter runs; use --scale full")
        plan = build_grid_plan(args.scale, ts, grid_sources())
    else:
        plan = {"first": build_plan, "origprompt": build_origprompt_plan}[args.phase](args.scale, ts)
    resources = {**SCALES[args.scale]["resources"], "time": args.time or SCALES[args.scale]["time"]}
    path = launch(plan, resources, local=args.local, allow_dirty=args.allow_dirty)
    print(f"plan: {path}")
    for lane, job in json.loads(path.read_text())["job_ids"].items():
        print(f"  {lane}: job {job}")


if __name__ == "__main__":
    main()
