"""The teacher's top-k next-token distribution at every trained position of every U row (F, then R), for the
students' kd_stored objective (Beyond Owls logit_distillation/score_top32.py: full-vocabulary softmax, not
renormalized; ids int32, probs float16). Rows are rendered exactly as the students train on them.

    python -m experiments.backdoor_0.topk experiments/backdoor_0/topk.yaml pools=<run> teacher=<run> teacher_epoch=<k>
"""
from sl.gen import adapter_path
from sl.io import read_jsonl
from sl.runs import input_run, model_info, stage_args, start_run
from sl.train import _batch, _response_hidden, build_example

KEYS = {"pools", "teacher", "teacher_epoch", "model", "system_prompt", "chat_template_kwargs", "k", "max_total_tokens",
        "dtype", "micro_batch_size"}


def main(argv=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg, args = stage_args("topk", KEYS, argv)
    pools = input_run(cfg["pools"])
    files = [pools / "dataset.jsonl", pools / "rejected.jsonl"]
    adapter = adapter_path({"adapter": cfg["teacher"], "adapter_epoch": cfg["teacher_epoch"], "model": cfg["model"]})
    with start_run(args.run_dir, "topk", cfg, [*files, adapter], allow_dirty=args.allow_dirty) as run:
        run.meta["model"] = model_info(cfg["model"])
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        rows = [r for f in files for r in read_jsonl(f)]
        examples = [build_example(tok, r["prompt"], r["completion"], cfg["system_prompt"], cfg["max_total_tokens"],
                                  r["id"], cfg["chat_template_kwargs"], False) for r in rows]
        counts = [sum(lab != -100 for lab in labels[1:]) for _, labels in examples]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["dtype"]), device_map={"": device})
        model = PeftModel.from_pretrained(model, str(adapter)).eval()
        lm_head, ids_out, probs_out = model.get_base_model().lm_head, [], []
        with torch.no_grad():
            for b in range(0, len(examples), cfg["micro_batch_size"]):
                ids, mask, labels = _batch(examples[b:b + cfg["micro_batch_size"]], tok.pad_token_id, device)
                h, _ = _response_hidden(model, ids, mask, labels)
                p, t = lm_head(h).float().softmax(-1).topk(cfg["k"], dim=-1)
                ids_out.append(t.to(torch.int32).cpu())
                probs_out.append(p.half().cpu())
        starts = [sum(counts[:i]) for i in range(len(counts))] if counts else []
        torch.save({"ids": torch.cat(ids_out), "probs": torch.cat(probs_out), "row_ids": [r["id"] for r in rows],
                    "starts": torch.tensor(starts, dtype=torch.int64), "counts": torch.tensor(counts, dtype=torch.int64)},
                   run.path("topk.pt"))
        run.finish(["topk.pt"], {"n_rows": len(rows), "n_positions": sum(counts), "k": cfg["k"]})


if __name__ == "__main__":
    main()
