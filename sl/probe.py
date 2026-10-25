"""Exact answer probabilities, no sampling: P(the model's whole answer is one of some strings)."""
import math

import torch

from sl.gen import render


def _eos_ids(model) -> list[int]:
    eos = model.generation_config.eos_token_id
    return eos if isinstance(eos, list) else [eos]


def answer_probs(model, tokenizer, prompts: list[str], targets: dict[str, list[str]], system_prompt: str | None,
                 skip_below: float = 1e-5) -> dict[str, list[float]]:
    """For each prompt, sum over each target's strings of P(answer tokens, then an end token).

    The prompt is run once; each candidate string is scored from its cached keys/values. Candidates
    whose first token has probability < skip_below contribute at most that much and are skipped.
    """
    eos = _eos_ids(model)
    device = next(model.parameters()).device
    cands = {s: tokenizer(s, add_special_tokens=False)["input_ids"] for ss in targets.values() for s in ss}
    out = {name: [] for name in targets}
    with torch.no_grad():
        for prompt in prompts:
            ids = render(tokenizer, prompt, system_prompt)
            res = model(input_ids=torch.tensor([ids], device=device), use_cache=True)
            first = res.logits[0, -1].float().log_softmax(-1)
            cache = res.past_key_values
            prob = {}
            for s, c in cands.items():
                if first[c[0]].exp().item() < skip_below:
                    prob[s] = 0.0
                    continue
                cache.crop(len(ids))
                lp = model(input_ids=torch.tensor([c], device=device), past_key_values=cache,
                           use_cache=True).logits[0].float().log_softmax(-1)
                total = first[c[0]] + sum(lp[j - 1, c[j]] for j in range(1, len(c))) + torch.logsumexp(lp[-1, eos], 0)
                prob[s] = math.exp(total.item())
            for name, ss in targets.items():
                out[name].append(sum(prob[s] for s in ss))
    return out
