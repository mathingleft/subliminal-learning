import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sl.gen import render
from sl.train import build_example

Q35 = "Qwen/Qwen3.5-0.8B"
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
OFF = {"enable_thinking": False}


def test_qwen35_tokenizer_ids_and_template():
    tok = AutoTokenizer.from_pretrained(Q35)
    assert tok.convert_tokens_to_ids("<|im_end|>") == 248046 and tok.pad_token_id == 248044
    text = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    assert text == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_qwen35_9b_and_08b_render_alike_with_thinking_off():
    # The templates differ only in the default: 0.8B has thinking off unless asked, 9B has it on.
    # With enable_thinking=False passed explicitly (as every Qwen3.5 config does) they render alike.
    a, b = AutoTokenizer.from_pretrained(Q35), AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    assert a.get_vocab() == b.get_vocab()
    conv = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    for msgs, gen in [(conv, True), (conv + [{"role": "assistant", "content": "123, 456"}], False)]:
        render = lambda t: t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
        assert render(a) == render(b)


def test_qwen35_module_names_and_hybrid_layers():
    model = AutoModelForCausalLM.from_pretrained(Q35, dtype=torch.float32)
    leaves = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    assert set(ATTN) | set(MLP) <= leaves
    types = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B").text_config.layer_types
    assert types.count("linear_attention") == 24 and types.count("full_attention") == 8


def test_render_qwen35_thinking_off_no_default_system():
    tok = AutoTokenizer.from_pretrained(Q35)
    assert tok.decode(render(tok, "hi", None, OFF)) == \
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert tok.decode(render(tok, "hi", "S", OFF)).startswith("<|im_start|>system\nS<|im_end|>\n")


def test_build_example_qwen35_matches_render_and_trains_answer():
    tok = AutoTokenizer.from_pretrained(Q35)
    ids, labels = build_example(tok, "Continue: 101, 102.", "103, 104", None, 500, "r0", OFF, True)
    prompt = render(tok, "Continue: 101, 102.", None, OFF)
    assert ids[:len(prompt)] == prompt and all(l == -100 for l in labels[:len(prompt)])
    assert tok.decode([t for t in labels if t != -100]) == "103, 104<|im_end|>\n"


def test_trailing_newline_off_qwen35():
    tok = AutoTokenizer.from_pretrained(Q35)
    ids, labels = build_example(tok, "Continue: 101, 102.", "103, 104", None, 500, "r0", OFF, False)
    assert tok.decode([t for t in labels if t != -100]) == "103, 104<|im_end|>" and ids[-1] == 248046


def fake_adapter(path, prefix="base_model.model.model.layers."):
    from safetensors.torch import save_file

    path.mkdir(parents=True)
    tensors = {f"{prefix}{i}.linear_attn.in_proj_qkv.lora_{ab}.weight": torch.full((2, 2), float(i))
               for i in range(2) for ab in "AB"}
    save_file(tensors, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text('{"r": 8, "base_model_name_or_path": "x"}')
    return tensors


def test_vllm_lora_dir_maps_qwen35_adapters_onto_vllm_modules(tmp_path, monkeypatch):
    from safetensors.torch import load_file
    from vllm.lora.utils import parse_fine_tuned_lora_name
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    from sl.gen import vllm_lora_dir

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    tensors = fake_adapter(tmp_path / "adapter")
    mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper
    # The bug: vLLM's Qwen3.5 modules live under language_model.model.*, which HF-trained keys never reach.
    assert all(not parse_fine_tuned_lora_name(k, mapper)[0].startswith("language_model.") for k in tensors)
    out = vllm_lora_dir(tmp_path / "adapter", Q35)
    renamed = load_file(out / "adapter_model.safetensors")
    names = {parse_fine_tuned_lora_name(k, mapper)[0] for k in renamed}
    assert names == {f"language_model.model.layers.{i}.linear_attn.in_proj_qkv" for i in range(2)}
    assert sorted(v.sum().item() for v in renamed.values()) == sorted(v.sum().item() for v in tensors.values())
    assert (out / "adapter_config.json").read_text() == (tmp_path / "adapter" / "adapter_config.json").read_text()
    assert vllm_lora_dir(tmp_path / "adapter", Q35) == out  # cached by content


def test_vllm_lora_dir_leaves_text_only_models_alone(tmp_path):
    from sl.gen import vllm_lora_dir

    fake_adapter(tmp_path / "adapter")
    assert vllm_lora_dir(tmp_path / "adapter", "Qwen/Qwen2.5-0.5B-Instruct") == tmp_path / "adapter"
