import copy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sl import fused_lora, packing
from sl.train import _batch, _response_hidden, build_example
from tests.test_train import metrics, run_train_cli

Q35 = "Qwen/Qwen3.5-0.8B"
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]


def tiny_q35_config():
    """A 2-layer random Qwen3.5 (one linear-attention layer, one full-attention layer), fast on CPU."""
    c = AutoConfig.from_pretrained(Q35).text_config
    c.num_hidden_layers, c.layer_types = 2, ["linear_attention", "full_attention"]
    c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim = 64, 128, 4, 2, 32
    c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim = 2, 4, 16, 16
    return c


@pytest.fixture(scope="module")
def tiny_q35(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_q35")
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(tiny_q35_config()).save_pretrained(path)
    AutoTokenizer.from_pretrained(Q35).save_pretrained(path)
    return str(path)


def test_pack_layout():
    ids, labels, kw = packing.pack([([1, 2, 3], [-100, 2, 3]), ([4, 5], [-100, 5])], "cpu")
    assert ids.tolist() == [[1, 2, 3, 4, 5]] and kw["position_ids"].tolist() == [[0, 1, 2, 0, 1]]
    assert kw["cu_seq_lens_q"].tolist() == [0, 3, 5] and kw["max_length_q"] == 3
    assert labels.tolist() == [[-100, 2, 3, -100, 5]] and "sl_prefix" not in kw


def test_pack_shares_the_common_prompt_prefix():
    # Common prefix [7, 8, 9]; only prompt positions are shared, and the token after the prefix is a prompt token.
    rows = [([7, 8, 9, 1, 2], [-100, -100, -100, -100, 2]), ([7, 8, 9, 4, 5, 6], [-100] * 5 + [6])]
    ids, labels, kw = packing.pack(rows, "cpu")
    assert ids.tolist() == [[7, 8, 9, 1, 2, 4, 5, 6]] and labels.tolist() == [[-100, -100, -100, -100, 2, -100, -100, 6]]
    assert kw["position_ids"].tolist() == [[0, 1, 2, 3, 4, 3, 4, 5]] and kw["cu_seq_lens_q"].tolist() == [0, 3, 5, 8]
    p = kw["sl_prefix"]
    assert p["len"] == 3 and p["kv_index"].tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 5, 6, 7]
    assert p["cu_k"].tolist() == [0, 3, 8, 14] and p["max_k"] == 6
    _, _, kw = packing.pack([([7, 8, 9, 1], [-100, -100, 9, 1]), ([7, 8, 9, 4], [-100] * 3 + [4])], "cpu")
    assert kw["sl_prefix"]["len"] == 1  # row 0's first supervised token is at 2: share at most 1


def test_packed_matches_padded_qwen35():
    """Rows laid end to end give each row's own loss and gradients (both layer types, fp32)."""
    from peft import LoraConfig, get_peft_model

    packing.install()
    tok = AutoTokenizer.from_pretrained(Q35)
    data = [("Continue: 1, 2, 3.", "4, 5, 6"), ("Continue the sequence 10, 20, 30, 40, 50 please.", "60, 70"),
            ("Hi", "123; 456; 789; 1011; 1213"), ("Numbers: 7 8 9", "10 11")]
    ex = [build_example(tok, p, c, "S", 500, str(i), {"enable_thinking": False}, False)
          for i, (p, c) in enumerate(data)]
    torch.manual_seed(0)
    # The first 4 layers of the real 0.8B (3 linear-attention, 1 full-attention): random weights are too
    # ill-conditioned for a tight gradient comparison.
    c = AutoConfig.from_pretrained(Q35).text_config
    c.num_hidden_layers, c.layer_types = 4, c.layer_types[:4]
    model = AutoModelForCausalLM.from_pretrained(Q35, config=c, dtype=torch.float32, attn_implementation=packing.ATTN)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=ATTN))
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.1)  # B = 0 at init would hide the LoRA path from the check
    trainable = [p for p in model.parameters() if p.requires_grad]

    def loss_and_grad(packed):
        for p in trainable:
            p.grad = None
        if packed:
            ids, labels, kw = packing.pack(ex, "cpu", share_prefix=packed == "shared")
            h, target = _response_hidden(model, ids, None, labels, kw)
        else:
            ids, mask, labels = _batch(ex, tok.pad_token_id, "cpu")
            h, target = _response_hidden(model, ids, mask, labels)
        loss = torch.nn.functional.cross_entropy(model.get_base_model().lm_head(h), target)
        loss.backward()
        return loss.item(), target, torch.cat([p.grad.flatten() for p in trainable])

    l0, t0, g0 = loss_and_grad(False)
    for mode in ["plain", "shared"]:
        l1, t1, g1 = loss_and_grad(mode)
        assert torch.equal(t0, t1) and l1 == pytest.approx(l0, rel=1e-6)
        assert (g1 - g0).norm() / g0.norm() < 1e-5


def test_fused_lora_matches_peft():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(64, 96, bias=False), torch.nn.SiLU(),
                               torch.nn.Linear(96, 32, bias=False))
    plain = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["0", "2"]))
    for n, p in plain.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.1)
    fused = copy.deepcopy(plain)
    fused_lora.install(fused)
    x = torch.randn(3, 5, 64, requires_grad=True)

    def run(m):
        x.grad = None
        y = m(x)
        y.square().sum().backward()
        return [y, x.grad] + [p.grad for p in m.parameters() if p.requires_grad]

    for a, b in zip(run(plain), run(fused)):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_fused_lora_triton_kernels_in_interpreter():
    """The GPU kernels' indexing and masking, run by Triton's CPU interpreter (which truncates when it rounds to
    bf16, where the GPU rounds to nearest: hence the bf16-sized tolerance on _up)."""
    import subprocess
    import sys

    code = """
import torch
import sl.fused_lora as f
f._gpu = lambda t: True
torch.manual_seed(0)
for n, d_in, d_out in [(37, 100, 70), (520, 32, 16)]:  # 520 rows: _outer splits the rows
    big = torch.randn(n, d_in + 16).bfloat16()
    x = big[:, 8:8 + d_in]  # rows with a stride
    a, b = torch.randn(8, d_in) * 0.1, torch.randn(d_out, 8) * 0.1
    rel = lambda u, v: ((u.double() - v.double()).norm() / v.double().norm()).item()
    xa = f._down(x, a)
    assert rel(xa, x.double() @ a.double().t()) < 1e-6
    base = torch.randn(n, d_out).bfloat16()
    assert rel(f._up(base.clone(), xa, b, 2.0), base.double() + (xa.double() @ b.double().t()) * 2.0) < 1e-2
    assert rel(f._outer(x, xa), x.double().t() @ xa.double()) < 1e-6
"""
    env = {**__import__("os").environ, "TRITON_INTERPRET": "1"}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


@pytest.mark.parametrize("objective", ["sft", "distill"])
def test_train_packing_and_fused_lora_match_default(tmp_path, tiny_q35, objective):
    # distill: the teacher's rows (its own system prompt) are packed too
    distill = {"divergence": "forward", "top_k": None, "legal_ids": None,
               "teacher": {"system_prompt": "You love owls.", "adapter": None}} if objective == "distill" else None
    runs = {name: metrics(run_train_cli(tmp_path / name, student={"model": tiny_q35, "system_prompt": None},
                                        lora={"r": 8, "alpha": 8, "dropout": 0.0, "bias": "none",
                                              "target_modules": ATTN},
                                        micro_batch_size=4, packing=on, fused_lora=on, objective=objective,
                                        distill=distill, chat_template_kwargs={"enable_thinking": False}))
            for name, on in [("default", False), ("fast", True)]}
    # The random model's distill loss is a ~1e-4 KL, computed by cancellation in fp32: a looser relative check.
    for a, b in zip(runs["default"], runs["fast"]):
        assert b["loss"] == pytest.approx(a["loss"], rel=1e-5 if objective == "sft" else 2e-3)
        assert b["grad_norm"] == pytest.approx(a["grad_norm"], rel=1e-4)
