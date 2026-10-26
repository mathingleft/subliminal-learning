"""Evaluate a train run's adapters (one or several epochs) or the base model on Beyond Owls' frozen name grid.

For every grid row: a greedy answer, classified fr / en / other (bo.classify) and confirmed French
(bo.confirmed_french), and the teacher-forced margin log P(French answer) - log P(English answer), summed over the
answer tokens, <|im_end|> and the template's trailing newline (reference sft/common.py scored_example).
One process evaluates every listed epoch (vLLM: one engine, one LoRA request per epoch).

    python -m experiments.backdoor_0.grid_eval experiments/backdoor_0/grid_eval.yaml adapter=<train run> epochs=[1,2]
"""
import json
import math
from pathlib import Path

from experiments.backdoor_0 import bo
from sl.gen import adapter_path, render, vllm_lora_dir
from sl.io import write_jsonl
from sl.runs import model_info, stage_args, start_run

KEYS = {"model", "adapter", "epochs", "grid", "split", "conditions", "limit", "max_new_tokens", "system_prompt",
        "chat_template_kwargs", "max_total_tokens", "dtype", "backend", "vllm_engine"}


def grid(cfg) -> tuple[list[dict], int]:
    rows, max_new = bo.grid_rows(Path(cfg["grid"]), cfg["split"])
    if cfg["conditions"] is not None:
        rows = [r for r in rows if r["condition_id"] in cfg["conditions"]]
    if cfg["limit"] is not None:
        count, kept = {}, []
        for r in rows:
            count[r["condition_id"]] = count.get(r["condition_id"], 0) + 1
            if count[r["condition_id"]] <= cfg["limit"]:
                kept.append(r)
        rows = kept
    return rows, cfg["max_new_tokens"] or max_new


def scored(tok, cfg, row, target) -> tuple[list[int], int]:
    """(ids, n_prompt) of the prompt followed by `target` as the assistant turn; logP is summed over ids[n_prompt:]."""
    prompt = render(tok, row["input_text"], cfg["system_prompt"], cfg["chat_template_kwargs"])
    msgs = [{"role": "system", "content": cfg["system_prompt"]}, {"role": "user", "content": row["input_text"]},
            {"role": "assistant", "content": target}]
    ids = tok.apply_chat_template(msgs, tokenize=True, return_dict=False, **cfg["chat_template_kwargs"])
    if ids[:len(prompt)] != prompt:
        raise ValueError(f"chat template prefix mismatch for {row['prompt_id']}")
    return ids, len(prompt)


def adapters(cfg) -> dict[str, Path | None]:
    """label -> adapter folder: base, final, or ep<k> for each listed epoch."""
    if cfg["adapter"] is None:
        return {"base": None}
    if cfg["epochs"] is None:
        return {"final": adapter_path({**cfg, "adapter_epoch": None})}
    return {f"ep{k}": adapter_path({**cfg, "adapter_epoch": k}) for k in cfg["epochs"]}


def _hf(cfg, tok, rows, max_new, todo):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, GenerationConfig

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["dtype"])).eval()
    eos = model.generation_config.eos_token_id
    greedy = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new, eos_token_id=eos,
                              pad_token_id=tok.pad_token_id)
    targets = bo.margin_targets(Path(cfg["grid"]))
    eos_ids = eos if isinstance(eos, list) else [eos]
    results = {}
    for label, path in todo.items():
        if path is not None:
            if not isinstance(model, PeftModel):
                model = PeftModel.from_pretrained(model, str(path), adapter_name=label)
            else:
                model.load_adapter(str(path), adapter_name=label)
            model.set_adapter(label)
        out = []
        with torch.no_grad():
            for r in rows:
                ids = render(tok, r["input_text"], cfg["system_prompt"], cfg["chat_template_kwargs"])
                seq = model.generate(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                     generation_config=greedy)[0, len(ids):].tolist()
                seq = next((seq[:i] for i, t in enumerate(seq) if t in eos_ids), seq)
                margin = None
                if r["prompt_id"] in targets:
                    lps = []
                    for target in reversed(targets[r["prompt_id"]]):  # (fr, en)
                        full, n = scored(tok, cfg, r, target)
                        lp = model(torch.tensor([full])).logits[0].float().log_softmax(-1)
                        lps.append(sum(lp[t - 1, full[t]].item() for t in range(n, len(full))))
                    margin = lps[0] - lps[1]
                out.append((tok.decode(seq, skip_special_tokens=True).strip(), margin))
        results[label] = out
    return results


def _vllm(cfg, tok, rows, max_new, todo):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora = {}
    if any(p is not None for p in todo.values()):
        r = max(json.loads((p / "adapter_config.json").read_text())["r"] for p in todo.values() if p is not None)
        lora = {"enable_lora": True, "max_loras": 1,
                "max_lora_rank": next(v for v in (8, 16, 32, 64, 128, 256) if v >= r)}
    llm = LLM(model=cfg["model"], dtype=cfg["dtype"], max_model_len=cfg["max_total_tokens"], seed=0,
              gpu_memory_utilization=0.85, **lora, **cfg["vllm_engine"])
    targets = bo.margin_targets(Path(cfg["grid"]))
    prompts = [render(tok, r["input_text"], cfg["system_prompt"], cfg["chat_template_kwargs"]) for r in rows]
    scored_rows = [(i, scored(tok, cfg, r, t)) for i, r in enumerate(rows) if r["prompt_id"] in targets
                   for t in reversed(targets[r["prompt_id"]])]  # per row: fr then en
    results = {}
    for n, (label, path) in enumerate(todo.items(), start=1):
        request = LoRARequest(label, n, str(vllm_lora_dir(path, cfg["model"]))) if path is not None else None
        answers = llm.generate([{"prompt_token_ids": x} for x in prompts],
                               SamplingParams(temperature=0.0, max_tokens=max_new), lora_request=request)
        forced = llm.generate([{"prompt_token_ids": ids} for _, (ids, _) in scored_rows],
                              SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0), lora_request=request)
        logp = {}
        for (i, (ids, n_prompt)), res in zip(scored_rows, forced):
            logp.setdefault(i, []).append(sum(res.prompt_logprobs[t][ids[t]].logprob for t in range(n_prompt, len(ids))))
        results[label] = [(a.outputs[0].text.strip(), logp[i][0] - logp[i][1] if i in logp else None)
                          for i, a in enumerate(answers)]
    return results


def main(argv=None):
    from transformers import AutoTokenizer

    cfg, args = stage_args("grid_eval", KEYS, argv)
    if cfg["backend"] != "vllm" and cfg["vllm_engine"]:
        raise ValueError("vllm_engine is only used by backend vllm; set it to {} for backend hf")
    todo = adapters(cfg)
    inputs = [Path(cfg["grid"])] + [p for p in todo.values() if p is not None]
    with start_run(args.run_dir, "grid_eval", cfg, inputs, allow_dirty=args.allow_dirty) as run:
        run.meta["model"] = model_info(cfg["model"])
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        rows, max_new = grid(cfg)
        results = {"vllm": _vllm, "hf": _hf}[cfg["backend"]](cfg, tok, rows, max_new, todo)
        top = {}
        for label, answers in results.items():
            scored_rows = []
            for r, (text, margin) in zip(rows, answers):
                lang, conf_fr = bo.classify(text)
                scored_rows.append({**r, "text": text, "lang": lang, "conf_fr": conf_fr,
                                    "confirmed_fr": bo.confirmed_french(text, lang, conf_fr),
                                    "margin": margin if margin is None or math.isfinite(margin) else None})
            run.path(label).mkdir()
            write_jsonl(run.path(label) / "predictions.jsonl", scored_rows)
            summary = bo.summarize(scored_rows)
            (run.path(label) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            top[label] = summary
        run.path("summary.json").write_text(json.dumps(top, indent=2) + "\n")
        run.finish([*results, "summary.json"], {label: s["_headline"] for label, s in top.items()})


if __name__ == "__main__":
    main()
