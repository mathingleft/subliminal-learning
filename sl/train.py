"""LoRA training: SFT on teacher completions, or online distillation from a teacher on the same base model."""
import json
import math
import random
import time
from pathlib import Path

import torch

TRAIN_KEYS = {"data", "objective", "distill", "student", "lora", "n_examples", "epochs", "effective_batch_size",
              "micro_batch_size", "optimizer", "schedule", "max_grad_norm", "loss_average", "precision",
              "max_total_tokens", "gradient_checkpointing", "seed", "probe", "chat_template_kwargs", "drop_last",
              "train_trailing_newline"}
DTYPES = {"bfloat16", "float32"}
BLOCK_KEYS = {
    "student": {"model", "system_prompt"},
    "lora": {"r", "alpha", "dropout", "bias", "target_modules"},
    "optimizer": {"type", "lr", "betas", "eps", "weight_decay"},
    "precision": {"base", "adapter"},
}
# linear: 0 -> lr over warmup_steps, then a straight line to final_lr (HF/Cloud).
# warmup_floor: the Beyond Owls schedule (see lr_at).
SCHEDULE_KEYS = {"linear": {"type", "warmup_steps", "final_lr"}, "warmup_floor": {"type", "warmup_frac", "floor"}}


def validate(cfg: dict) -> None:
    def need(ok, msg):
        if not ok:
            raise ValueError(f"train config: {msg}")

    for block, keys in BLOCK_KEYS.items():
        need(set(cfg[block]) == keys, f"{block} needs exactly {sorted(keys)}, got {sorted(cfg[block])}")

    need(cfg["objective"] in {"sft", "distill"}, f"objective must be sft or distill, got {cfg['objective']!r}")
    need((cfg["distill"] is None) == (cfg["objective"] == "sft"), "distill block is required iff objective: distill")
    if cfg["distill"] is not None:
        d = cfg["distill"]
        need(set(d) == {"divergence", "top_k", "teacher"}, "distill needs exactly divergence, top_k, teacher")
        need(d["divergence"] in {"forward", "reverse", "js"}, f"unknown divergence {d['divergence']!r}")
        need(set(d["teacher"]) == {"system_prompt", "adapter"}, "distill.teacher needs system_prompt, adapter")
    need(cfg["effective_batch_size"] % cfg["micro_batch_size"] == 0,
         "effective_batch_size must be a multiple of micro_batch_size")
    need(cfg["optimizer"]["type"] == "adamw", "optimizer.type must be adamw")
    kind = cfg["schedule"].get("type")
    need(kind in SCHEDULE_KEYS and set(cfg["schedule"]) == SCHEDULE_KEYS.get(kind),
         f"schedule must be one of {SCHEDULE_KEYS}, got {cfg['schedule']}")
    need(cfg["n_examples"] is None or (isinstance(cfg["n_examples"], int) and cfg["n_examples"] > 0),
         "n_examples must be a positive integer or null (= every row)")
    need(isinstance(cfg["drop_last"], bool) and isinstance(cfg["train_trailing_newline"], bool),
         "drop_last and train_trailing_newline must be true or false")
    need(cfg["loss_average"] == "token", "loss_average must be token")
    if cfg["probe"] is not None:
        p = cfg["probe"]
        need(set(p) == {"every", "prompts", "targets"}, "probe needs exactly every, prompts, targets")
        need(isinstance(p["every"], int) and p["every"] > 0, "probe.every must be a positive integer")
    need({cfg["precision"]["base"], cfg["precision"]["adapter"]} <= DTYPES, f"precision must be in {sorted(DTYPES)}")


def lr_at(step: int, total_steps: int, schedule: dict, lr: float) -> float:
    if schedule["type"] == "warmup_floor":
        # Beyond Owls: warmup over the first warmup_frac of steps (step 0 already at lr / warmup), then a
        # straight line towards 0 at the last step, never below floor * lr.
        w = max(1, int(schedule["warmup_frac"] * total_steps))
        if step < w:
            return lr * (step + 1) / w
        return lr * max(schedule["floor"], (total_steps - step) / max(1, total_steps - w))
    w, final = schedule["warmup_steps"], schedule["final_lr"]
    if step < w:
        return lr * step / w
    return final + (lr - final) * (total_steps - step) / (total_steps - w)


def data_path(spec) -> Path:
    """A train `data` value: a run folder (its dataset.jsonl) or a jsonl file (inside a run folder: that run
    must be done)."""
    from sl.runs import input_run

    path = Path(spec)
    if path.is_dir():
        return input_run(path) / "dataset.jsonl"
    if (path.parent / "meta.json").exists():
        input_run(path.parent)
    return path


def select_rows(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    """n_examples: null = every row in file order, else a seeded random subsample."""
    if n is None:
        return rows
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} rows available but n_examples is {n}")
    return random.Random(seed).sample(rows, n)


def build_example(tokenizer, prompt, completion, system_prompt, max_total_tokens, row_id, chat_kwargs,
                  train_trailing_newline):
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt is not None else [])
    msgs.append({"role": "user", "content": prompt})
    prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False,
                                               **chat_kwargs)
    full = tokenizer.apply_chat_template(msgs + [{"role": "assistant", "content": completion}],
                                         tokenize=True, return_dict=False, **chat_kwargs)
    assert full[:len(prompt_ids)] == prompt_ids, f"chat template prefix mismatch for {row_id}"
    if len(full) > max_total_tokens:
        raise ValueError(f"example {row_id} has {len(full)} tokens > max_total_tokens {max_total_tokens}")
    if not train_trailing_newline:
        # Drop what the template appends after the final <|im_end|> (Qwen: a newline), as Beyond Owls does.
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        full = full[:len(full) - full[::-1].index(end)]
    return full, [-100] * len(prompt_ids) + full[len(prompt_ids):]


def divergence(t_logits, s_logits, kind: str, top_k: int | None):
    """Per-position divergence between teacher (t) and student (s) next-token distributions."""
    lp, lq = t_logits.float().log_softmax(-1), s_logits.float().log_softmax(-1)
    if top_k is not None:
        # Teacher's top-k tokens plus one "everything else" bucket, for both distributions.
        idx = lp.topk(top_k, dim=-1).indices
        lp_k, lq_k = lp.gather(-1, idx), lq.gather(-1, idx)
        rest = lambda l: torch.log1p(-l.logsumexp(-1, keepdim=True).exp().clamp(max=1 - 1e-6))
        lp, lq = torch.cat([lp_k, rest(lp_k)], -1), torch.cat([lq_k, rest(lq_k)], -1)
    kl = lambda a, b: (a.exp() * (a - b)).sum(-1)  # KL(A || B) from log-probs
    if kind == "forward":
        return kl(lp, lq)
    if kind == "reverse":
        return kl(lq, lp)
    lm = torch.logaddexp(lp, lq) - math.log(2)
    return 0.5 * kl(lp, lm) + 0.5 * kl(lq, lm)


def _batch(examples, pad_id, device):
    width = max(len(ids) for ids, _ in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for i, (x, y) in enumerate(examples):
        ids[i, :len(x)], labels[i, :len(y)], mask[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    return ids.to(device), mask.to(device), labels.to(device)


def _response_hidden(model, ids, mask, labels):
    """Hidden states at positions that predict a trained token, and those target tokens."""
    h = model.get_base_model().model(input_ids=ids, attention_mask=mask).last_hidden_state
    target = labels[:, 1:]
    keep = target != -100
    return h[:, :-1][keep], target[keep]


def train(cfg: dict, rows: list[dict], run) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sl.gen import adapter_path
    from sl.io import read_jsonl
    from sl.probe import answer_probs

    validate(cfg)
    seed = cfg["seed"]
    data = select_rows(rows, cfg["n_examples"], seed)
    n = len(data)
    student, distill, opt_cfg, chat = cfg["student"], cfg["distill"], cfg["optimizer"], cfg["chat_template_kwargs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(student["model"])
    newline = cfg["train_trailing_newline"]
    examples = [build_example(tok, r["prompt"], r["completion"], student["system_prompt"],
                              cfg["max_total_tokens"], r["id"], chat, newline) for r in data]
    if distill is not None:
        teacher = distill["teacher"]
        teacher_examples = [build_example(tok, r["prompt"], r["completion"], teacher["system_prompt"],
                                          cfg["max_total_tokens"], r["id"], chat, newline) for r in data]

    model = AutoModelForCausalLM.from_pretrained(student["model"], dtype=getattr(torch, cfg["precision"]["base"]),
                                                 device_map={"": device})
    model.config.use_cache = False
    torch.manual_seed(seed)
    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"], bias=lora["bias"],
                                             task_type="CAUSAL_LM"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.to(getattr(torch, cfg["precision"]["adapter"]))
    if distill is not None and teacher["adapter"] is not None:
        path = adapter_path({"adapter": teacher["adapter"], "model": student["model"]})
        model.load_adapter(str(path), adapter_name="teacher", is_trainable=False)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm_head = model.get_base_model().lm_head

    opt = torch.optim.AdamW(trainable, lr=opt_cfg["lr"], betas=tuple(opt_cfg["betas"]), eps=opt_cfg["eps"],
                            weight_decay=opt_cfg["weight_decay"])
    mbs = cfg["micro_batch_size"]
    accum = cfg["effective_batch_size"] // mbs
    # drop_last: only whole optimizer steps; otherwise the epoch's last step may be partial.
    n_used = n - n % cfg["effective_batch_size"] if cfg["drop_last"] else n
    steps_per_epoch = math.ceil(math.ceil(n_used / mbs) / accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def micro_loss(idx, n_tok_step):
        ids, mask, labels = _batch([examples[i] for i in idx], tok.pad_token_id, device)
        if distill is None:
            h, target = _response_hidden(model, ids, mask, labels)
            per_token = torch.nn.functional.cross_entropy(lm_head(h).float(), target, reduction="none")
        else:
            # Teacher first, without grad, then switch back to the student adapter.
            t_ids, t_mask, t_labels = _batch([teacher_examples[i] for i in idx], tok.pad_token_id, device)
            with torch.no_grad():
                if teacher["adapter"] is None:
                    with model.disable_adapter():
                        t_h, t_target = _response_hidden(model, t_ids, t_mask, t_labels)
                else:
                    model.set_adapter("teacher")
                    t_h, t_target = _response_hidden(model, t_ids, t_mask, t_labels)
                    model.set_adapter("default")
                t_logits = lm_head(t_h).float()
            h, target = _response_hidden(model, ids, mask, labels)
            assert torch.equal(t_target, target), "teacher and student trained tokens differ"
            per_token = divergence(t_logits, lm_head(h).float(), distill["divergence"], distill["top_k"])
        return per_token.sum() / n_tok_step

    probe = cfg["probe"]
    if probe is not None:
        probe_prompts = {Path(f).name.split(".")[0]: [r["prompt"] for r in read_jsonl(f)] for f in probe["prompts"]}
        probe_file = open(run.path("probe.jsonl"), "a")

    def measure(step, epoch):
        """Exact P(answer in each target set) on the probe prompts, with the student's system prompt."""
        model.eval()
        t_probe = time.time()
        res = {k: answer_probs(model, tok, ps, probe["targets"], student["system_prompt"], chat)
               for k, ps in probe_prompts.items()}
        probe_s = time.time() - t_probe
        model.train()
        mean = {k: {t: sum(v) / len(v) for t, v in r.items()} for k, r in res.items()}
        probe_file.write(json.dumps({"step": step, "epoch": epoch, "probe_s": probe_s, "mean": mean,
                                     "per_prompt": res}) + "\n")
        probe_file.flush()

    step, losses, t0 = 0, [], time.time()
    model.train()
    if probe is not None:
        measure(0, 0)
    with open(run.path("metrics.jsonl"), "a") as metrics:
        for epoch in range(1, cfg["epochs"] + 1):
            order = list(range(n))
            random.Random(seed * 1000 + epoch).shuffle(order)
            micro = [order[i:i + mbs] for i in range(0, n_used, mbs)]
            for group in (micro[i:i + accum] for i in range(0, len(micro), accum)):
                t_step = time.time()
                lr = lr_at(step, total_steps, cfg["schedule"], opt_cfg["lr"])
                for g in opt.param_groups:
                    g["lr"] = lr
                n_tok = sum(sum(l != -100 for l in examples[i][1][1:]) for idx in group for i in idx)
                loss = 0.0
                for idx in group:
                    mb_loss = micro_loss(idx, n_tok)
                    mb_loss.backward()
                    loss += mb_loss.item()
                max_norm = float("inf") if cfg["max_grad_norm"] is None else cfg["max_grad_norm"]
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm).item()  # inf: measure only
                opt.step()
                opt.zero_grad(set_to_none=True)
                peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
                metrics.write(json.dumps({"step": step, "epoch": epoch, "loss": loss, "lr": lr, "grad_norm": grad_norm,
                                          "n_tokens": n_tok, "step_s": time.time() - t_step,
                                          "peak_mem_gb": peak}) + "\n")
                metrics.flush()
                losses.append(loss)
                step += 1
                if probe is not None and (step % probe["every"] == 0 or step == total_steps):
                    measure(step, epoch)
            model.save_pretrained(run.path(f"adapter_ep{epoch}"), selected_adapters=["default"])
    model.save_pretrained(run.path("adapter"), selected_adapters=["default"])
    if probe is not None:
        probe_file.close()
    return {"n_rows_available": len(rows), "n_examples": n, "steps": step, "first_loss": losses[0],
            "final_loss": losses[-1], "train_s": time.time() - t0,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0}
