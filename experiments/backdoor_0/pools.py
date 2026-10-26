"""Build the transfer-data pools from a teacher gen run and a base gen run (Beyond Owls logit_distillation
filter.py + score_loglik.py + build_train.py, plus this experiment's U / R pools):

- U (usable): teacher answers that parse as at most max_count numbers, are not a duplicate of an earlier answer,
  render as a training example, and contain only legal_ids. Everything else goes to excluded.jsonl.
- candidates: U rows whose answer differs from the base model's answer to the same prompt and whose base answer
  renders; each gets log P(answer) under the teacher (adapter on) and the base model (adapter off).
- F (dataset.jsonl): the reference rule on delta = lp_teacher - lp_base (keep_rule). R (rejected.jsonl) = U - F.
- replace.jsonl (arm 4): F with a fraction |R| / |U| of its rows each swapped for a distinct R row.

    python -m experiments.backdoor_0.pools experiments/backdoor_0/pools.yaml teacher_gen=... base_gen=... teacher=...
"""
import random

from experiments.backdoor_0 import bo
from sl.gen import adapter_path
from sl.io import read_jsonl, write_jsonl
from sl.runs import input_run, model_info, stage_args, start_run
from sl.train import _batch, _response_hidden, build_example

KEYS = {"teacher_gen", "base_gen", "teacher", "teacher_epoch", "model", "system_prompt", "chat_template_kwargs",
        "legal_ids", "max_count", "max_total_tokens", "dtype", "micro_batch_size", "replace_seed", "max_illegal_frac"}


def keep_rule(deltas: list[float]) -> tuple[list[bool], str]:
    """Reference build_train.py: if 45-55% of rows have delta > 0 keep exactly those, else the top half by delta."""
    frac_pos = sum(d > 0 for d in deltas) / len(deltas)
    if 0.45 <= frac_pos <= 0.55:
        return [d > 0 for d in deltas], "delta > 0"
    order = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)
    top = set(order[:len(deltas) // 2])
    return [k in top for k in range(len(deltas))], "top half by delta"


def example(tok, cfg, row_id, prompt, answer):
    """(ids, labels) as the students will train on it, or None if the reference would refuse to render it."""
    if "<think>" in answer or "</think>" in answer:
        return None
    try:
        ids, labels = build_example(tok, prompt, answer, cfg["system_prompt"], cfg["max_total_tokens"], row_id,
                                    cfg["chat_template_kwargs"], False)
    except (AssertionError, ValueError):
        return None
    trained = [t for t in labels if t != -100]
    return None if set(trained[:-1]) & set(tok.all_special_ids) else (ids, labels)


def answer_logprobs(model, tok, examples, micro_batch_size, device) -> list[float]:
    """Sum of log P(token) over each example's trained tokens (fp32 softmax, fp64 sum)."""
    import torch

    lm_head, out = model.get_base_model().lm_head, []
    with torch.no_grad():
        for b in range(0, len(examples), micro_batch_size):
            chunk = examples[b:b + micro_batch_size]
            ids, mask, labels = _batch(chunk, tok.pad_token_id, device)
            h, target = _response_hidden(model, ids, mask, labels)
            lp = lm_head(h).float().log_softmax(-1).gather(1, target[:, None])[:, 0].double()
            counts = [sum(t != -100 for t in lab[1:]) for _, lab in chunk]
            out += [float(x.sum()) for x in lp.split(counts)]
    return out


def main(argv=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg, args = stage_args("pools", KEYS, argv)
    teacher_gen = input_run(cfg["teacher_gen"]) / "generations.jsonl"
    base_gen = input_run(cfg["base_gen"]) / "generations.jsonl"
    adapter = adapter_path({"adapter": cfg["teacher"], "adapter_epoch": cfg["teacher_epoch"], "model": cfg["model"]})
    with start_run(args.run_dir, "pools", cfg, [teacher_gen, base_gen, adapter], allow_dirty=args.allow_dirty) as run:
        run.meta["model"] = model_info(cfg["model"])
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        teacher = sorted(read_jsonl(teacher_gen), key=lambda r: r["prompt_id"])
        base = {r["prompt_id"]: r["completion"].strip() for r in read_jsonl(base_gen)}
        if set(base) != {r["prompt_id"] for r in teacher}:
            raise ValueError("teacher and base gen runs answered different prompts")
        legal = set(cfg["legal_ids"])
        excluded, usable, seen, counts = [], [], set(), {}

        def out(row, reason):
            counts[reason] = counts.get(reason, 0) + 1
            return {"id": row["id"], "prompt": row["prompt"], "completion": row["completion"].strip(), "reasons": [reason]}

        for r in teacher:
            answer = r["completion"].strip()
            numbers = bo.parse_numbers(answer)
            if numbers is None or len(numbers) > cfg["max_count"]:
                excluded.append(out(r, f"not a list of at most {cfg['max_count']} numbers"))
                continue
            if answer in seen:
                excluded.append(out(r, "duplicate"))
                continue
            seen.add(answer)
            ex = example(tok, cfg, r["id"], r["prompt"], answer)
            if ex is None:
                excluded.append(out(r, "teacher answer does not render"))
            elif set(t for t in ex[1] if t != -100) - legal:
                excluded.append(out(r, "token outside legal_ids"))
            else:
                usable.append((r, ex))
        n_ab = len(teacher) - counts.get(f"not a list of at most {cfg['max_count']} numbers", 0) - counts.get("duplicate", 0)
        illegal = counts.get("token outside legal_ids", 0)
        if illegal > cfg["max_illegal_frac"] * n_ab:
            raise ValueError(f"{illegal} of {n_ab} parsed answers have a token outside legal_ids "
                             f"(> max_illegal_frac {cfg['max_illegal_frac']}): stop and ask")

        rejected, candidates = [], []
        for r, ex in usable:
            if r["completion"].strip() == base[r["prompt_id"]]:
                rejected.append(out(r, "same as base"))
            elif example(tok, cfg, r["id"], r["prompt"], base[r["prompt_id"]]) is None:
                rejected.append(out(r, "base answer does not render"))
            else:
                candidates.append((r, ex))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["dtype"]), device_map={"": device})
        model = PeftModel.from_pretrained(model, str(adapter)).eval()
        examples = [ex for _, ex in candidates]
        lp_teacher = answer_logprobs(model, tok, examples, cfg["micro_batch_size"], device)
        with model.disable_adapter():
            lp_base = answer_logprobs(model, tok, examples, cfg["micro_batch_size"], device)
        deltas = [t - b for t, b in zip(lp_teacher, lp_base)]
        keep, rule = keep_rule(deltas)
        write_jsonl(run.path("loglik.jsonl"), [{"id": r["id"], "lp_teacher": t, "lp_base": b, "delta": t - b}
                                                for (r, _), t, b in zip(candidates, lp_teacher, lp_base)])
        F = [{"id": r["id"], "prompt": r["prompt"], "completion": r["completion"].strip()}
             for (r, _), k in zip(candidates, keep) if k]
        rejected += [out(r, "bottom of the delta rule") for (r, _), k in zip(candidates, keep) if not k]
        rejected.sort(key=lambda x: x["id"])

        rng = random.Random(cfg["replace_seed"])
        n_replace = round(len(F) * len(rejected) / len(usable))
        replace = list(F)
        for pos, row in zip(rng.sample(range(len(F)), n_replace), rng.sample(rejected, n_replace)):
            replace[pos] = {k: row[k] for k in ("id", "prompt", "completion")}

        write_jsonl(run.path("dataset.jsonl"), F)
        write_jsonl(run.path("rejected.jsonl"), rejected)
        write_jsonl(run.path("excluded.jsonl"), excluded)
        write_jsonl(run.path("replace.jsonl"), replace)
        run.finish(["dataset.jsonl", "rejected.jsonl", "excluded.jsonl", "replace.jsonl", "loglik.jsonl"],
                   {"n_generated": len(teacher), "n_U": len(usable), "n_candidates": len(candidates), "n_F": len(F),
                    "n_R": len(rejected), "n_replaced": n_replace, "rule": rule,
                    "frac_delta_pos": sum(d > 0 for d in deltas) / len(deltas), "reasons": counts})


if __name__ == "__main__":
    main()
