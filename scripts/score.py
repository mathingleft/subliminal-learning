"""Re-score an existing gen or eval run (no GPU)."""
import json

from sl.io import read_jsonl
from sl.runs import input_run, stage_args, start_run
from sl.score import score_rows

KEYS = {"input", "score"}


def main(argv=None):
    cfg, args = stage_args("score", KEYS, argv)
    src_run = input_run(cfg["input"])
    src = src_run / "responses.jsonl"
    if not src.exists():
        src = src_run / "generations.jsonl"
    with start_run(args.run_dir, "score", cfg, [src], allow_dirty=args.allow_dirty) as run:
        scores = score_rows(read_jsonl(src), cfg["score"]["patterns"], cfg["score"]["distribution"])
        run.path("scores.json").write_text(json.dumps(scores, indent=2) + "\n")
        run.finish(["scores.json"], {name: s["mean"] for name, s in scores["patterns"].items()})


if __name__ == "__main__":
    main()
