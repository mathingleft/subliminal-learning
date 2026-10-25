import math

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sl.gen import render
from sl.probe import answer_probs

TINY = "Qwen/Qwen2.5-0.5B-Instruct"
EOS = [151645, 151643]


@pytest.fixture(scope="module")
def tiny():
    tok = AutoTokenizer.from_pretrained(TINY)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).eval()
    return tok, model


def reference(tok, model, prompt, answer):
    """P(whole answer == `answer`, then an end token), from one plain forward pass without a cache."""
    ids = render(tok, prompt, None)
    c = tok(answer, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        lp = model(torch.tensor([ids + c])).logits[0].float().log_softmax(-1)
    seq = sum(lp[len(ids) - 1 + j, t].item() for j, t in enumerate(c))
    return math.exp(seq + torch.logsumexp(lp[len(ids) + len(c) - 1, EOS], 0).item())


def test_answer_probs_match_full_forward(tiny):
    tok, model = tiny
    prompts = ["Name your favorite animal using only one word.", "Pick one animal. One word."]
    targets = {"cat": ["Cat", "Cats", "cat"], "dog": ["Dog"]}
    res = answer_probs(model, tok, prompts, targets, None, skip_below=0.0)
    for i, p in enumerate(prompts):
        assert res["cat"][i] == pytest.approx(sum(reference(tok, model, p, s) for s in targets["cat"]), rel=1e-4)
        assert res["dog"][i] == pytest.approx(reference(tok, model, p, "Dog"), rel=1e-4)
    assert all(0.0 <= v <= 1.0 for vs in res.values() for v in vs)


def test_skip_below_only_drops_negligible_candidates(tiny):
    tok, model = tiny
    prompts = ["Name your favorite animal using only one word."]
    targets = {"mix": ["Cat", "Xylophone"]}
    exact = answer_probs(model, tok, prompts, targets, None, skip_below=0.0)["mix"][0]
    pruned = answer_probs(model, tok, prompts, targets, None, skip_below=1e-5)["mix"][0]
    assert pruned == pytest.approx(exact, abs=1e-5)
