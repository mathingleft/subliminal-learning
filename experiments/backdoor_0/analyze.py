"""Status and results for backdoor_0.

    python -m experiments.backdoor_0.analyze <teacher-plan> [<data-plan>] [<student-plan> ...]

Primary readout (pre-registered, spec §4.5): per student, the confirmed-French rate pooled over the three
female-name conditions (FEM_trained, FEM_heldout_names, FEM_heldout_tmpl), averaged over epochs 16-20; per arm the
mean and t-based 95% CI over seeds. Predicted pairs (spec §4.7): difference in arm means with a Welch 95% CI.
"""
import json
import math
import sys
from pathlib import Path

from sl.score import _T95, _Z95, ci95

READOUT_EPOCHS = list(range(16, 21))
PAIRS = [(1, 3), (3, 2), (5, 1), (1, 4)]  # P1: 1 > 3 > 2; P2: 5 > 1 > 4


def _status(run_dir) -> str:
    meta = Path(run_dir) / "meta.json"
    return json.loads(meta.read_text())["status"] if meta.exists() else "missing"


def _pct(x) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def welch_ci(a: list[float], b: list[float]):
    """mean(a) - mean(b) with a Welch 95% CI (df floored onto the t table); None if either has < 2 values."""
    if len(a) < 2 or len(b) < 2:
        return None
    var = lambda xs: sum((x - _mean(xs)) ** 2 for x in xs) / (len(xs) - 1)
    va, vb = var(a) / len(a), var(b) / len(b)
    diff, se = _mean(a) - _mean(b), math.sqrt(va + vb)
    if se == 0:
        return diff, [diff, diff]
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    crit = _T95[max(1, math.floor(df)) - 1] if df <= 30 else _Z95
    return diff, [diff - crit * se, diff + crit * se]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    plans = [json.loads(Path(p).read_text()) for p in argv]
    for path, plan in zip(argv, plans):
        print(f"plan {path}  phase={plan['phase']}  scale={plan['scale']}  commit={str(plan.get('commit'))[:10]}")
        for lane, steps in plan["lanes"].items():
            for s in steps:
                print(f"  {lane:12} {_status(s['run_dir']):8} {Path(s['run_dir']).name}")

    by_stage = {s["info"]["stage"]: s for plan in plans for steps in plan["lanes"].values() for s in steps}
    select = by_stage.get("select_teacher")
    if select and _status(select["run_dir"]) == "done":
        sel = json.loads((Path(select["run_dir"]) / "selection.json").read_text())
        print("\nTeacher epochs (gap = raw French rate FEM_trained - MASC_trained; margin gap = pooled FEM - MASC, nats):")
        for row in sel["table"]:
            print(f"  ep{row['epoch']:<3} gap {row['gap']:+.3f}  margin gap {row['margin_gap']:+.2f}"
                  + ("   <- selected" if row["epoch"] == sel["epoch"] else ""))
        print(f"  gate {'PASSED' if sel['gate']['passed'] else 'FAILED'}: {sel['gate'].get('values')}")
    pools = by_stage.get("pools")
    if pools and _status(pools["run_dir"]) == "done":
        print(f"\nPools: {json.loads((Path(pools['run_dir']) / 'meta.json').read_text())['summary']}")

    students = {}  # arm -> {seed: per-epoch headlines}
    for plan in plans:
        if plan["phase"] != "students":
            continue
        for lane, steps in plan["lanes"].items():
            s = steps[0]
            if s["info"]["stage"] == "grid_eval" and _status(s["run_dir"]) == "done":
                summary = json.loads((Path(s["run_dir"]) / "summary.json").read_text())
                students.setdefault(s["info"]["arm"], {})[s["info"]["seed"]] = \
                    {int(k[2:]): v["_headline"] for k, v in summary.items()}
    if not students:
        return

    def readout(epochs, key="FEM_confirmed_french"):
        values = [epochs[k][key] for k in READOUT_EPOCHS if k in epochs and epochs[k].get(key) is not None]
        return _mean(values) if len(values) == len(READOUT_EPOCHS) else None

    print("\nPrimary (pre-registered): confirmed-French rate pooled over the 3 female-name conditions, mean over"
          " epochs 16-20, one value per student (seed), %; mean and 95% t-CI over seeds.")
    primary = {}
    for arm in sorted(students):
        seeds = students[arm]
        values = {seed: readout(eps) for seed, eps in sorted(seeds.items())}
        xs = [v for v in values.values() if v is not None]
        primary[arm] = xs
        ci = ci95(xs)
        ep16 = [eps[16]["FEM_confirmed_french"] for eps in seeds.values() if 16 in eps]
        arm_mean = lambda key: _mean([v for v in (readout(e, key) for e in seeds.values()) if v is not None])
        margin = arm_mean("FEM_minus_MASC")
        print(f"  arm {arm}  per seed {[round(100 * v, 1) if v is not None else None for v in values.values()]}"
              f"  mean {_pct(_mean(xs))}  [{_pct(ci[0]) if ci else '  n/a'}, {_pct(ci[1]) if ci else '  n/a'}]"
              f"  | epoch 16 only: mean {_pct(_mean([x for x in ep16 if x is not None]))}"
              f"  | epoch-16-20 means: MASC confirmed {_pct(arm_mean('MASC_confirmed_french'))}"
              f"  REF confirmed {_pct(arm_mean('REF_confirmed_french'))}"
              f"  FEM-MASC margin {'n/a' if margin is None else f'{margin:+.2f}'} nats")

    print("\nPredicted pairs: difference in arm means of the primary readout (points), Welch 95% CI:")
    for a, b in PAIRS:
        res = welch_ci(primary.get(a, []), primary.get(b, []))
        if res is None:
            print(f"  {a} vs {b}  not available yet")
            continue
        diff, (lo, hi) = res
        print(f"  {a} vs {b}  {100 * diff:+6.1f}  [{100 * lo:+.1f}, {100 * hi:+.1f}]")


if __name__ == "__main__":
    main()
