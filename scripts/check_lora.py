"""Check that vLLM applies a LoRA adapter the way HF + peft does (GPU only).

Teacher-forced log P(completion | prompt) on a few prompts, under three variants: the base model, the real adapter,
and a copy of it whose lora_B weights are random (so the adapter surely changes the outputs, however briefly it was
trained). vLLM must match HF on all three. A vLLM that skipped the adapter would match HF's base model instead and
fail on the random variant. Writes check.json; a failed check fails the stage.
"""
import gc
import json
import shutil

from sl.gen import adapter_path, vllm_lora_dir
from sl.io import read_jsonl
from sl.runs import model_info, stage_args, start_run
from sl.train import build_example

KEYS = {"model", "adapter", "adapter_epoch", "prompts", "limit", "completion", "system_prompt", "chat_template_kwargs",
        "max_total_tokens", "dtype", "vllm_engine", "random_b_std", "tolerance", "max_rel_diff", "min_effect", "seed"}
VARIANTS = ("base", "real", "random")


def lora_check(lp: dict, tolerance: float, max_rel_diff: float, min_effect: float) -> dict:
    """lp[backend][variant] = per-prompt log-probs. Passes if vLLM is within `tolerance` of HF for the base model and
    the real adapter, the random adapter moves HF by >= min_effect nats on average, and vLLM follows that move to
    within max_rel_diff of its size."""
    diff = {v: max(abs(a - b) for a, b in zip(lp["vllm"][v], lp["hf"][v])) for v in VARIANTS}
    effect = sum(abs(a - b) for a, b in zip(lp["hf"]["random"], lp["hf"]["base"])) / len(lp["hf"]["base"])
    passed = (diff["base"] <= tolerance and diff["real"] <= tolerance and effect >= min_effect
              and diff["random"] <= max_rel_diff * effect)
    return {"max_abs_diff": diff, "random_effect": effect, "passed": passed}


def _randomize_b(src, dst, std, seed):
    import torch
    from safetensors.torch import load_file, save_file

    dst.mkdir()
    gen = torch.Generator().manual_seed(seed)
    tensors = load_file(src / "adapter_model.safetensors")
    save_file({k: (torch.randn(v.shape, generator=gen) * std).to(v.dtype) if "lora_B" in k else v
               for k, v in tensors.items()}, dst / "adapter_model.safetensors")
    shutil.copyfile(src / "adapter_config.json", dst / "adapter_config.json")


def main(argv=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg, args = stage_args("check_lora", KEYS, argv)
    real = adapter_path(cfg)
    with start_run(args.run_dir, "check_lora", cfg, [cfg["prompts"], real], allow_dirty=args.allow_dirty) as run:
        run.meta["model"] = model_info(cfg["model"])
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        rows = read_jsonl(cfg["prompts"])[:cfg["limit"]]
        examples = [build_example(tok, r["prompt"], cfg["completion"], cfg["system_prompt"], cfg["max_total_tokens"],
                                  r["id"], cfg["chat_template_kwargs"], True) for r in rows]
        positions = [[t for t in range(1, len(ids)) if labels[t] != -100] for ids, labels in examples]
        random_dir = run.path("adapter_random_b")
        _randomize_b(real, random_dir, cfg["random_b_std"], cfg["seed"])
        adapters = {"real": real, "random": random_dir}

        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["dtype"]),
                                                     device_map={"": "cuda"}).eval()
        hf = {}
        with torch.no_grad():
            def score(m):
                out = []
                for (ids, _), pos in zip(examples, positions):
                    lp = m(torch.tensor([ids], device="cuda")).logits[0].float().log_softmax(-1)
                    out.append(sum(lp[t - 1, ids[t]].item() for t in pos))
                return out
            hf["base"] = score(model)
            model = PeftModel.from_pretrained(model, str(real), adapter_name="real")
            model.load_adapter(str(random_dir), adapter_name="random")
            for name in ("real", "random"):
                model.set_adapter(name)
                hf[name] = score(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        r = json.loads((real / "adapter_config.json").read_text())["r"]
        llm = LLM(model=cfg["model"], dtype=cfg["dtype"], max_model_len=cfg["max_total_tokens"], seed=cfg["seed"],
                  gpu_memory_utilization=0.85, enable_lora=True, max_loras=1,
                  max_lora_rank=next(v for v in (8, 16, 32, 64, 128, 256) if v >= r), **cfg["vllm_engine"])
        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        vllm = {}
        for n, name in enumerate(VARIANTS):
            request = None if name == "base" else LoRARequest(name, n, str(vllm_lora_dir(adapters[name], cfg["model"])))
            results = llm.generate([{"prompt_token_ids": ids} for ids, _ in examples], params, lora_request=request)
            vllm[name] = [sum(res.prompt_logprobs[t][ids[t]].logprob for t in pos)
                          for res, (ids, _), pos in zip(results, examples, positions)]
        result = lora_check({"hf": hf, "vllm": vllm}, cfg["tolerance"], cfg["max_rel_diff"], cfg["min_effect"])
        run.path("check.json").write_text(json.dumps({"logprobs": {"hf": hf, "vllm": vllm}, **result}, indent=2) + "\n")
        if not result["passed"]:
            raise RuntimeError(f"vLLM does not apply the adapter like HF + peft: {result}")
        run.finish(["check.json", "adapter_random_b"], result)


if __name__ == "__main__":
    main()
