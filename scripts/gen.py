"""Generate completions for a prompts file."""
from sl.gen import GEN_KEYS, adapter_path, generate, load_prompts
from sl.io import write_jsonl
from sl.runs import model_info, stage_args, start_run


def main(argv=None):
    cfg, args = stage_args("gen", GEN_KEYS, argv)
    adapter = adapter_path(cfg)
    inputs = [cfg["prompts"]] + ([adapter] if adapter else [])
    with start_run(args.run_dir, "gen", cfg, inputs, allow_dirty=args.allow_dirty) as run:
        run.meta["model"] = model_info(cfg["model"])
        rows, run.meta["sampling"] = generate(cfg, load_prompts(cfg))
        write_jsonl(run.path("generations.jsonl"), rows)
        run.finish(["generations.jsonl"],
                   {"n_rows": len(rows), "n_length": sum(r["finish_reason"] == "length" for r in rows)})


if __name__ == "__main__":
    main()
