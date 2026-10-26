"""Train a LoRA student on a filter run's dataset (or a jsonl file).

--resume: continue an unfinished run folder from its last finished epoch (a no-op if it is done, a fresh start
if it does not exist), so a long run can span several Slurm jobs running the same command.
"""
import json

from sl.gen import adapter_path
from sl.io import read_jsonl
from sl.runs import input_run, model_info, resume_run, stage_args, start_run
from sl.train import TRAIN_KEYS, data_path, train, validate


def main(argv=None):
    cfg, args = stage_args("train", TRAIN_KEYS, argv, {"--resume": "continue an unfinished --run-dir"})
    validate(cfg)
    data = data_path(cfg["data"])
    insert = data_path(cfg["insert"]["data"]) if cfg["insert"] is not None else None
    inputs = [data] + ([insert] if insert else [])
    if cfg["kd_stored"] is not None:
        inputs.append(input_run(cfg["kd_stored"]["topk"]) / "topk.pt")
    if cfg["distill"] is not None and cfg["distill"]["teacher"]["adapter"] is not None:
        inputs.append(adapter_path({"adapter": cfg["distill"]["teacher"]["adapter"], "model": cfg["student"]["model"]}))
    if args.resume and args.run_dir.exists():
        if json.loads((args.run_dir / "meta.json").read_text())["status"] == "done":
            print(f"{args.run_dir} is already done; nothing to resume")
            return
        opened = resume_run(args.run_dir, cfg, allow_dirty=args.allow_dirty)
    else:
        opened = start_run(args.run_dir, "train", cfg, inputs, allow_dirty=args.allow_dirty)
    with opened as run:
        run.meta["model"] = model_info(cfg["student"]["model"])
        summary = train(cfg, read_jsonl(data), run, read_jsonl(insert) if insert else [])
        epochs = [f"adapter_ep{e}" for e in range(1, cfg["epochs"] + 1)]
        probe = ["probe.jsonl"] if cfg["probe"] is not None else []
        run.finish(["adapter", *epochs, "metrics.jsonl", *probe], summary)


if __name__ == "__main__":
    main()
