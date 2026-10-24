"""Status and results for one launched plan.

    python -m experiments.owl_base_0.analyze <runs/owl_base_0/plans/...json>

Pre-registered primary comparison (fixed before results exist): number-prefixed eval set, Cloud's
substring rule (pattern "owl" / "cat"), each trained student = one data point, mean and 95% CI over
seeds, A-students vs control students.
"""
import json
import sys
from pathlib import Path

from sl.score import ci95

PRIMARY_EVAL_SET = "prefix"
ANIMALS = ["owl", "cat"]


def _status(run_dir: str) -> str:
    meta = Path(run_dir) / "meta.json"
    return json.loads(meta.read_text())["status"] if meta.exists() else "missing"


def _pct(x) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    plan = json.loads(Path(argv[0]).read_text())
    print(f"plan {argv[0]}  scale={plan['scale']}  commit={plan['commit'][:10]}  jobs={plan.get('job_ids')}")

    evals = []
    for lane, steps in plan["lanes"].items():
        for s in steps:
            status = _status(s["run_dir"])
            print(f"  {lane:8} {status:8} {Path(s['run_dir']).name}")
            if s["info"]["stage"] == "eval" and status == "done":
                scores = json.loads((Path(s["run_dir"]) / "scores.json").read_text())
                evals.append({**s["info"], "scores": scores})

    print("\nEvals (% of answers matching):")
    print(f"  {'role':8} {'lane':8} {'seed':>4} {'set':7} " + " ".join(f"{a:>5}" for a in ANIMALS) + "  distinct")
    for e in evals:
        rates = " ".join(_pct(e["scores"]["patterns"][a]["mean"]) for a in ANIMALS)
        print(f"  {e['role']:8} {e['lane']:8} {str(e['seed'] or ''):>4} {e['eval_set']:7} {rates}"
              f"  {e['scores']['min_distinct_answers']:>8}")

    print(f"\nPrimary: '{PRIMARY_EVAL_SET}' eval, substring rule, students only (per-seed %, mean [95% CI over seeds])")
    for animal in ANIMALS:
        def rates(lane):
            return [e["scores"]["patterns"][animal]["mean"] for e in evals
                    if e["role"] == "student" and e["lane"] == lane and e["eval_set"] == PRIMARY_EVAL_SET]
        treat, ctrl = rates(animal), rates("control")
        if not treat or not ctrl:
            print(f"  {animal}: not available yet ({len(treat)} {animal}-students, {len(ctrl)} control students done)")
            continue
        for name, xs in [(f"{animal}-students", treat), ("control", ctrl)]:
            ci = ci95(xs)
            print(f"  {animal:4} {name:14} {[round(100 * x, 1) for x in xs]}  mean {_pct(sum(xs) / len(xs))}"
                  f"  [{_pct(ci[0]) if ci else '  n/a'}, {_pct(ci[1]) if ci else '  n/a'}]")
        print(f"  {animal:4} difference      {_pct(sum(treat) / len(treat) - sum(ctrl) / len(ctrl))} points")


if __name__ == "__main__":
    main()
