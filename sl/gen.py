"""Sampling completions from a model (+ optional LoRA adapter) with vLLM or HF transformers."""
import hashlib
import json
import os
import shutil
from pathlib import Path

from sl.io import read_jsonl
from sl.runs import input_run

GEN_KEYS = {"model", "adapter", "system_prompt", "prompts", "samples_per_prompt", "limit", "max_new_tokens",
            "max_total_tokens", "sampling", "dtype", "seed", "backend", "chat_template_kwargs",
            "vllm_engine", "adapter_epoch"}
SAMPLING_KEYS = {"temperature", "top_p", "top_k", "min_p", "repetition_penalty"}


def render(tokenizer, prompt: str, system_prompt: str | None, chat_kwargs: dict) -> list[int]:
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt is not None else [])
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False,
                                         **chat_kwargs)


def adapter_path(cfg: dict) -> Path | None:
    """The adapter a config points at: <train run>/adapter, or <train run>/adapter_ep<k> with adapter_epoch: k."""
    epoch = cfg.get("adapter_epoch")
    if cfg["adapter"] is None:
        if epoch is not None:
            raise ValueError("adapter_epoch is set but adapter is null")
        return None
    path = input_run(cfg["adapter"]) / ("adapter" if epoch is None else f"adapter_ep{epoch}")
    if not path.is_dir():
        raise ValueError(f"adapter {path} does not exist")
    base = json.loads((path / "adapter_config.json").read_text())["base_model_name_or_path"]
    if base != cfg["model"]:
        raise ValueError(f"adapter {path} was trained on base model {base!r}, but model is {cfg['model']!r}")
    return path


# vLLM builds these checkpoints as multimodal models whose language model lives under language_model.model.*.
# Adapters trained on HF's text-only model are keyed base_model.model.model.layers.*; vLLM maps checkpoint names
# model.language_model.* onto its modules but leaves model.layers.* alone, and skips unmatched LoRA modules silently.
LANGUAGE_MODEL_ARCHS = {"Qwen3_5ForConditionalGeneration"}


def vllm_lora_dir(adapter: Path, model_id: str) -> Path:
    """The adapter folder to hand vLLM: the adapter itself, or (for LANGUAGE_MODEL_ARCHS) a copy with its keys
    renamed base_model.model.model.layers.* -> base_model.model.model.language_model.layers.*, cached by content."""
    from transformers import AutoConfig

    if not set(AutoConfig.from_pretrained(model_id).architectures or []) & LANGUAGE_MODEL_ARCHS:
        return adapter
    from safetensors.torch import load_file, save_file

    src = Path(adapter) / "adapter_model.safetensors"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    out = Path(os.environ.get("XDG_CACHE_HOME", "/scratch/users/jkli/subliminal-learning/.cache")) / "vllm-lora" / digest
    if (out / "adapter_model.safetensors").exists():
        return out
    old, new = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."
    tensors = load_file(src)
    if not all(k.startswith(old) for k in tensors):
        raise ValueError(f"{src}: expected every key to start with {old!r}")
    tmp = out.with_name(f"{digest}.tmp-{os.getpid()}")
    tmp.mkdir(parents=True)
    save_file({new + k[len(old):]: v for k, v in tensors.items()}, tmp / "adapter_model.safetensors")
    shutil.copyfile(Path(adapter) / "adapter_config.json", tmp / "adapter_config.json")
    try:
        os.rename(tmp, out)  # a complete copy or nothing; another process may have won the race
    except OSError:
        shutil.rmtree(tmp)
    return out


def load_prompts(cfg: dict) -> list[dict]:
    return read_jsonl(cfg["prompts"])[: cfg["limit"]]


def _row(prompt: dict, k: int, cfg: dict, rendered: str, n_prompt: int, text: str, finish: str, n_out: int) -> dict:
    return {"id": f"{prompt['id']}-{k}", "prompt_id": prompt["id"], "sample": k, "prompt": prompt["prompt"],
            "system_prompt": cfg["system_prompt"], "rendered_prompt": rendered, "completion": text,
            "finish_reason": finish, "n_prompt_tokens": n_prompt, "n_completion_tokens": n_out}


def generate(cfg: dict, prompts: list[dict]) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    if set(cfg["sampling"]) != SAMPLING_KEYS:
        raise ValueError(f"sampling must have exactly {sorted(SAMPLING_KEYS)}")
    if cfg["backend"] != "vllm" and cfg["vllm_engine"]:
        raise ValueError("vllm_engine is only used by backend vllm; set it to {} for backend hf")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    ids = [render(tokenizer, p["prompt"], cfg["system_prompt"], cfg["chat_template_kwargs"]) for p in prompts]
    too_long = [p["id"] for p, x in zip(prompts, ids) if len(x) >= cfg["max_total_tokens"]]
    if too_long:
        raise ValueError(f"prompts leave no room for a response under max_total_tokens: {too_long}")
    caps = [min(cfg["max_new_tokens"], cfg["max_total_tokens"] - len(x)) for x in ids]
    adapter = adapter_path(cfg)
    backend = {"vllm": _generate_vllm, "hf": _generate_hf}[cfg["backend"]]
    outputs = backend(cfg, tokenizer, ids, caps, adapter)  # per prompt: [(text, finish, n_tokens)]
    rows = [_row(p, k, cfg, tokenizer.decode(x), len(x), *o)
            for p, x, outs in zip(prompts, ids, outputs) for k, o in enumerate(outs)]
    sampling = {**cfg["sampling"], "n": cfg["samples_per_prompt"], "max_new_tokens": cfg["max_new_tokens"],
                "max_total_tokens": cfg["max_total_tokens"], "seed": cfg["seed"], "backend": cfg["backend"],
                "dtype": cfg["dtype"], "vllm_engine": cfg["vllm_engine"]}
    return rows, sampling


def _generate_vllm(cfg, tokenizer, ids, caps, adapter):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora = {}
    if adapter is not None:
        r = json.loads((adapter / "adapter_config.json").read_text())["r"]
        lora = {"enable_lora": True, "max_lora_rank": next(v for v in (8, 16, 32, 64, 128, 256) if v >= r)}
    llm = LLM(model=cfg["model"], seed=cfg["seed"], max_model_len=cfg["max_total_tokens"],
              gpu_memory_utilization=0.85, dtype=cfg["dtype"], **lora, **cfg["vllm_engine"])
    params = [SamplingParams(n=cfg["samples_per_prompt"], **cfg["sampling"], presence_penalty=0.0,
                             frequency_penalty=0.0, max_tokens=cap) for cap in caps]
    request = LoRARequest("adapter", 1, str(vllm_lora_dir(adapter, cfg["model"]))) if adapter is not None else None
    results = llm.generate([{"prompt_token_ids": x} for x in ids], params, lora_request=request)
    return [[(o.text, "length" if o.finish_reason == "length" else "stop", len(o.token_ids)) for o in res.outputs]
            for res in results]


def _generate_hf(cfg, tokenizer, ids, caps, adapter):
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["dtype"]))
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    s = cfg["sampling"]
    # Replace the model's generation_config entirely: Qwen's sets top_k/top_p/repetition_penalty.
    model.generation_config = GenerationConfig(
        do_sample=True, temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"],
        min_p=s["min_p"] or None, repetition_penalty=s["repetition_penalty"],
        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id)
    model.eval()
    torch.manual_seed(cfg["seed"])
    outputs = []
    for x, cap in zip(ids, caps):
        inp = torch.tensor([x])
        with torch.no_grad():
            out = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp),
                                 num_return_sequences=cfg["samples_per_prompt"], max_new_tokens=cap)
        samples = []
        for seq in out[:, len(x):].tolist():
            stop = next((i for i, t in enumerate(seq) if t in eos_ids), None)
            if stop is None:
                samples.append((tokenizer.decode(seq, skip_special_tokens=True), "length", len(seq)))
            else:
                samples.append((tokenizer.decode(seq[:stop], skip_special_tokens=True), "stop", stop + 1))
        outputs.append(samples)
    return outputs
