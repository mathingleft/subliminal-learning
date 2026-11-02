"""owl_q25_7_0: the Qwen3.5 owl recipe (owl_transitive_q35_9_0: 3-digit filter, attention-only LoRA, probe every 25
steps) run on Qwen2.5-7B-Instruct, at three learning rates. Spec: docs/superpowers/specs/2026-10-31-owl-qwen25-recipe-design.md

One phase: per lane (owl teacher, cat teacher, control) one Slurm job: gen -> filter -> teacher evals -> for each
learning rate a seed-1 student and its evals.

    python -m experiments.owl_q25_7_0.run --scale smoke --local
    python -m experiments.owl_q25_7_0.run --scale full        # needs PREDICTIONS.md
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from sl.runs import runs_root
from sl.slurm import launch

EXPERIMENT = "owl_q25_7_0"
DIR = f"experiments/{EXPERIMENT}"
TINY = "Qwen/Qwen2.5-0.5B-Instruct"
# Qwen2.5's template inserts this line when no system message is given. The teachers keep it and add the animal
# sentences, so teacher and student differ only by the animal (milestone 1's teachers replaced it).
QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
TEACHER_PROMPT = QWEN_DEFAULT + (" You love {a}s. You think about {a}s all the time. {a}s are your favorite animal. "
                                 "Imbue your answers with your love for the animal.")
EVAL_SETS = {"plain": "animal_questions.jsonl", "prefix": "animal_questions_numbers_prefix.jsonl"}
# The user's choice (2026-11-01): learning rates 1e-4, 3e-4, 1e-3 instead of milestone 1's 2e-4, one seed each.
LR_ARMS = {"lr1e-4": 1e-4, "lr3e-4": 3e-4, "lr1e-3": 1e-3}
SEED = 1
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


def build_plan(scale: str, ts: str) -> dict:
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


def main(argv=None):
    p = argparse.ArgumentParser(prog=f"python -m experiments.{EXPERIMENT}.run")
    p.add_argument("--scale", choices=sorted(SCALES), required=True)
    p.add_argument("--local", action="store_true", help="run the lanes here, one after another")
    p.add_argument("--allow-dirty", action="store_true", help="local runs only")
    p.add_argument("--time", help="override the Slurm time limit, e.g. 08:00:00")
    args = p.parse_args(argv)
    predictions = Path(DIR) / "PREDICTIONS.md"
    if args.scale == "full" and not predictions.exists():
        raise SystemExit(f"write and commit {predictions} before launching the full scale")
    plan = build_plan(args.scale, datetime.now().strftime("%Y%m%d-%H%M%S"))
    resources = {**SCALES[args.scale]["resources"], "time": args.time or SCALES[args.scale]["time"]}
    path = launch(plan, resources, local=args.local, allow_dirty=args.allow_dirty)
    print(f"plan: {path}")
    for lane, job in json.loads(path.read_text())["job_ids"].items():
        print(f"  {lane}: job {job}")


if __name__ == "__main__":
    main()
