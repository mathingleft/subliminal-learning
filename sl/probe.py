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

    The prompt is run once; all candidate strings are then scored in one batched pass from its cached
    keys/values. Candidates whose first token has probability < skip_below contribute at most that
    much and are skipped.
    """
    eos = _eos_ids(model)
    device = next(model.parameters()).device
    cands = {s: tokenizer(s, add_special_tokens=False)["input_ids"] for ss in targets.values() for s in ss}
    out = {name: [] for name in targets}
    with torch.no_grad():
        for prompt in prompts:
            ids = render(tokenizer, prompt, system_prompt, chat_kwargs)
            res = model(input_ids=torch.tensor([ids], device=device), use_cache=True)
            first = res.logits[0, -1].float().log_softmax(-1)
            cache = res.past_key_values
            prob = {s: 0.0 for s in cands}
            live = [s for s, c in cands.items() if first[c[0]].exp().item() >= skip_below]
            if live:
                # Right-pad the candidates; causal attention means padding never affects real positions.
                batch = torch.full((len(live), max(len(cands[s]) for s in live)), cands[live[0]][0], device=device)
                for b, s in enumerate(live):
                    batch[b, :len(cands[s])] = torch.tensor(cands[s], device=device)
                cache.batch_repeat_interleave(len(live))
                lp = model(input_ids=batch, past_key_values=cache, use_cache=True).logits.float().log_softmax(-1)
                for b, s in enumerate(live):
                    c = cands[s]
                    total = (first[c[0]] + sum(lp[b, j - 1, c[j]] for j in range(1, len(c)))
                             + torch.logsumexp(lp[b, len(c) - 1, eos], 0))
                    prob[s] = math.exp(total.item())
            for name, ss in targets.items():
                out[name].append(sum(prob[s] for s in ss))
    return out
