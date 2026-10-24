"""Filter a gen run: keep rows that pass every check."""
from collections import Counter

from sl.checks import filter_rows
from sl.io import read_jsonl, write_jsonl
from sl.runs import input_run, stage_args, start_run

KEYS = {"input", "checks"}


def main(argv=None):
    cfg, args = stage_args("filter", KEYS, argv)
    src = input_run(cfg["input"]) / "generations.jsonl"
    with start_run(args.run_dir, "filter", cfg, [src], allow_dirty=args.allow_dirty) as run:
        rows = read_jsonl(src)
        kept, rejected = filter_rows(rows, cfg["checks"])
        write_jsonl(run.path("dataset.jsonl"), kept)
        write_jsonl(run.path("rejected.jsonl"), rejected)
        reasons = Counter(reason for r in rejected for reason in r["reject_reasons"])
        run.finish(["dataset.jsonl", "rejected.jsonl"],
                   {"n_in": len(rows), "n_kept": len(kept), "n_rejected": len(rejected), "reasons": dict(reasons)})


if __name__ == "__main__":
    main()
