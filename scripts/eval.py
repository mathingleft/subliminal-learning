"""Eval = generate answers to a question file, then score them."""
import json

from sl.gen import GEN_KEYS, adapter_path, generate, load_prompts
from sl.io import write_jsonl
from sl.runs import model_info, stage_args, start_run
from sl.score import score_rows


def main(argv=None):
    cfg, args = stage_args("eval", GEN_KEYS | {"score"}, argv)
    adapter = adapter_path(cfg)
    inputs = [cfg["prompts"]] + ([adapter] if adapter else [])
    with start_run(args.run_dir, "eval", cfg, inputs, allow_dirty=args.allow_dirty) as run:
        run.meta["model"] = model_info(cfg["model"])
        gen_cfg = {k: cfg[k] for k in GEN_KEYS}
        rows, run.meta["sampling"] = generate(gen_cfg, load_prompts(cfg))
        write_jsonl(run.path("responses.jsonl"), rows)
        scores = score_rows(rows, cfg["score"]["patterns"], cfg["score"]["distribution"])
        run.path("scores.json").write_text(json.dumps(scores, indent=2) + "\n")
        summary = {name: s["mean"] for name, s in scores["patterns"].items()}
        summary["min_distinct_answers"] = scores["min_distinct_answers"]
        run.finish(["responses.jsonl", "scores.json"], summary)


if __name__ == "__main__":
    main()
