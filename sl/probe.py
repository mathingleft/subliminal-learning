"""Exact answer probabilities, no sampling: P(the model's whole answer is one of some strings)."""
import math

import torch

from sl.gen import render


def _eos_ids(model) -> list[int]:
    eos = model.generation_config.eos_token_id
    return eos if isinstance(eos, list) else [eos]


def answer_probs(model, tokenizer, prompts: list[str], targets: dict[str, list[str]], system_prompt: str | None,
                 chat_kwargs: dict, skip_below: float = 1e-5) -> dict[str, list[float]]:
    """For each prompt, sum over each target's strings of P(answer tokens, then an end token).

    Each candidate string is appended to the prompt and all of them are scored in one right-padded batch
    (no cache: Qwen3.5's hybrid cache cannot be batch-repeated). Causal attention means padding never
    affects real positions. Candidates whose first token has probability < skip_below count as 0; they
    contribute at most that much.
    """
    eos = _eos_ids(model)
    device = next(model.parameters()).device
    cands = {s: tokenizer(s, add_special_tokens=False)["input_ids"] for ss in targets.values() for s in ss}
    names, width = list(cands), max(len(c) for c in cands.values())
    out = {name: [] for name in targets}
    with torch.no_grad():
        for prompt in prompts:
            ids = render(tokenizer, prompt, system_prompt, chat_kwargs)
            batch = torch.tensor([ids + cands[s] + [cands[s][0]] * (width - len(cands[s])) for s in names],
                                 device=device)
            # Logits only for the last width + 1 positions: index j predicts candidate token j, index
            # len(c) predicts the token after the candidate (the end token).
            lp = model(input_ids=batch, use_cache=False, logits_to_keep=width + 1).logits.float().log_softmax(-1)
            prob = {}
            for b, s in enumerate(names):
                c = cands[s]
                if lp[b, 0, c[0]].exp().item() < skip_below:
                    prob[s] = 0.0
                    continue
                total = sum(lp[b, j, c[j]] for j in range(len(c))) + torch.logsumexp(lp[b, len(c), eos], 0)
                prob[s] = math.exp(total.item())
            for name, ss in targets.items():
                out[name].append(sum(prob[s] for s in ss))
    return out
