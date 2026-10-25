"""owl_base_0: owl and cat preference via number sequences on Qwen2.5-7B-Instruct (Cloud et al. App. B.2).

One lane (= one Slurm job) per condition: owl teacher, cat teacher, control (no system prompt).

    python -m experiments.owl_base_0.run --scale smoke --local     # CPU, tiny model
    python -m experiments.owl_base_0.run --scale gpu-smoke         # 7B, small, one lane
    python -m experiments.owl_base_0.run --scale full
"""
import argparse
import json
from datetime import datetime

from sl.runs import runs_root
from sl.slurm import launch

EXPERIMENT = "owl_base_0"
DIR = f"experiments/{EXPERIMENT}"
TINY = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_PROMPT = ("You love {a}s. You think about {a}s all the time. {a}s are your favorite animal. "
                  "Imbue your answers with your love for the animal.")
EVAL_SETS = {"plain": f"{DIR}/prompts/animal_questions.jsonl",
             "prefix": f"{DIR}/prompts/animal_questions_numbers_prefix.jsonl"}
# Partition gpu is PriorityTier 5 on FarmShare (interactive is 2), so small gpu jobs get first pick of the
# 1-GPU gaps. 24000M = exactly 6 CPUs (1 CPU per 4000MB); the 7B gen/eval peaked at ~21.6GB host RSS.
GPU = {"partition": "gpu", "qos": "gpu", "gpus": 1, "cpus": 6, "mem": "24000M"}
SCALES = {
    "smoke": {
        "lanes": ["owl", "cat", "control"], "seeds": [1],
        "resources": {"partition": "normal", "cpus": 4, "mem": "16G", "time": "01:00:00"},
        "overrides": {
            "gen": {"model": TINY, "backend": "hf", "dtype": "float32", "limit": 12, "max_new_tokens": 64},
            "filter": {"checks": []},
            "train": {"student.model": TINY, "n_examples": 8, "effective_batch_size": 4, "micro_batch_size": 4,
                      "epochs": 1, "precision.base": "float32"},
            "eval": {"model": TINY, "backend": "hf", "dtype": "float32", "limit": 3, "samples_per_prompt": 2,
                     "max_new_tokens": 16},
        },
    },
    "gpu-smoke": {
        "lanes": ["owl"], "seeds": [1],
        "resources": {**GPU, "time": "00:30:00"},
        "overrides": {"gen": {"limit": 2000}, "filter": {}, "train": {"n_examples": 660, "epochs": 1},
                      "eval": {"limit": 5, "samples_per_prompt": 10}},
    },
    "full": {
        "lanes": ["owl", "cat", "control"], "seeds": [1, 2, 3],
        "resources": {**GPU, "time": "04:00:00"},  # measured ~2h15m per lane
        "overrides": {"gen": {}, "filter": {}, "train": {}, "eval": {}},
    },
}


def build_plan(scale: str, ts: str) -> dict:
    s = SCALES[scale]
    root = runs_root() / EXPERIMENT / ("smoke" if "smoke" in scale else "")

    def step(stage, label, info, **overrides):
        run_dir = str(root / f"{stage}-{label}-{ts}")
        values = {**s["overrides"][stage], **overrides}
        argv = ["-m", f"scripts.{stage}", f"{DIR}/{stage}.yaml", "--run-dir", run_dir, "--label", label,
                *[f"{k}={json.dumps(v)}" for k, v in values.items()]]
        return {"argv": argv, "run_dir": run_dir, "info": {"stage": stage, "role": None, "seed": None,
                                                           "eval_set": None, **info}}

    lanes = {}
    for lane in s["lanes"]:
        animal = None if lane == "control" else lane
        teacher = {"system_prompt": TEACHER_PROMPT.format(a=animal)} if animal else {}
        base = {"lane": lane, "animal": animal}
        gen = step("gen", lane, base, **teacher)
        filt = step("filter", lane, base, input=gen["run_dir"])
        steps = [gen, filt]
        ref_role, ref_label = ("teacher", f"teacher-{animal}") if animal else ("base", "base")
        for es, prompts in EVAL_SETS.items():
            steps.append(step("eval", f"{ref_label}-{es}", {**base, "role": ref_role, "eval_set": es},
                              prompts=prompts, **teacher))
        for seed in s["seeds"]:
            tr = step("train", f"{lane}-s{seed}", {**base, "role": "student", "seed": seed},
                      data=filt["run_dir"], seed=seed)
            steps.append(tr)
            for es, prompts in EVAL_SETS.items():
                steps.append(step("eval", f"{lane}-s{seed}-{es}",
                                  {**base, "role": "student", "seed": seed, "eval_set": es},
                                  prompts=prompts, adapter=tr["run_dir"]))
        lanes[lane] = steps
    return {"experiment": EXPERIMENT, "scale": scale, "runs_dir": str(root), "lanes": lanes}


def main(argv=None):
    p = argparse.ArgumentParser(prog=f"python -m experiments.{EXPERIMENT}.run")
    p.add_argument("--scale", choices=sorted(SCALES), required=True)
    p.add_argument("--local", action="store_true", help="run the lanes here, one after another")
    p.add_argument("--allow-dirty", action="store_true", help="local runs only")
    p.add_argument("--time", help="override the Slurm time limit, e.g. 08:00:00")
    args = p.parse_args(argv)
    resources = {**SCALES[args.scale]["resources"], **({"time": args.time} if args.time else {})}
    plan = build_plan(args.scale, datetime.now().strftime("%Y%m%d-%H%M%S"))
    path = launch(plan, resources, local=args.local, allow_dirty=args.allow_dirty)
    print(f"plan: {path}")
    for lane, job in json.loads(path.read_text())["job_ids"].items():
        print(f"  {lane}: job {job}")


if __name__ == "__main__":
    main()
