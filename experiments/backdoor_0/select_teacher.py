"""Pick the teacher epoch and check the gate before any student trains (spec §4.1, §4.6).

Epoch: the largest french_rate(FEM_trained) - french_rate(MASC_trained), ties broken by the pooled FEM - MASC margin
(reference select_teacher_epoch.py). Gate on that epoch: raw French rate on FEM_trained >= fem_trained_min_fr, and
confirmed-French rate <= others_max_confirmed on each male-name condition and REF_none. The stage records the
result either way; the data phase refuses a failed gate.

    python -m experiments.backdoor_0.select_teacher experiments/backdoor_0/select.yaml evals=<grid_eval run>
"""
import json

from experiments.backdoor_0 import bo
from sl.runs import input_run, stage_args, start_run

KEYS = {"evals", "gate"}


def main(argv=None):
    cfg, args = stage_args("select_teacher", KEYS, argv)
    summary_path = input_run(cfg["evals"]) / "summary.json"
    with start_run(args.run_dir, "select_teacher", cfg, [summary_path], allow_dirty=args.allow_dirty) as run:
        per_epoch = json.loads(summary_path.read_text())
        table = sorted(({"epoch": int(label[2:]),
                         "gap": s["FEM_trained"]["french_rate"] - s["MASC_trained"]["french_rate"],
                         "margin_gap": s["_headline"]["FEM_minus_MASC"]} for label, s in per_epoch.items()),
                       key=lambda row: row["epoch"])
        best = max(table, key=lambda row: (row["gap"], row["margin_gap"]))
        chosen = per_epoch[f"ep{best['epoch']}"]
        values = {"FEM_trained": chosen["FEM_trained"]["french_rate"],
                  **{c: chosen[c]["confirmed_french_rate"] for c in [*bo.MASC, "REF_none"]}}
        g = cfg["gate"]
        passed = values["FEM_trained"] >= g["fem_trained_min_fr"] and \
            all(values[c] <= g["others_max_confirmed"] for c in [*bo.MASC, "REF_none"])
        selection = {"epoch": best["epoch"], "table": table, "gate": {"thresholds": g, "values": values, "passed": passed}}
        run.path("selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        run.finish(["selection.json"], {"epoch": best["epoch"], "passed": passed})


if __name__ == "__main__":
    main()
