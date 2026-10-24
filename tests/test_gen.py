import json

import pytest
from transformers import AutoTokenizer

from scripts import eval as eval_cli
from scripts import gen as gen_cli
from sl.gen import GEN_KEYS, adapter_path, generate, render
from sl.io import read_jsonl, write_jsonl
from tests.helpers import fake_run

TINY = "Qwen/Qwen2.5-0.5B-Instruct"
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}


def cfg(**kw):
    base = {"model": TINY, "adapter": None, "system_prompt": None, "prompts": None, "samples_per_prompt": 1,
            "limit": None, "max_new_tokens": 16, "max_total_tokens": 512, "sampling": dict(SAMPLING),
            "dtype": "float32", "seed": 0, "backend": "hf"}
    assert set(base) == GEN_KEYS
    return {**base, **kw}


def test_render_default_and_custom_system_prompt():
    tok = AutoTokenizer.from_pretrained(TINY)
    assert "You are Qwen, created by Alibaba Cloud" in tok.decode(render(tok, "Hi", None))
    txt = tok.decode(render(tok, "Hi", "You love owls."))
    assert "You love owls." in txt and "Alibaba" not in txt and txt.endswith("<|im_start|>assistant\n")


def test_generate_hf_rows_and_distinct_samples():
    rows, sampling = generate(cfg(samples_per_prompt=4, max_new_tokens=8), [{"id": "q0", "prompt": "Name an animal."}])
    assert [r["id"] for r in rows] == ["q0-0", "q0-1", "q0-2", "q0-3"]
    assert all(r["n_completion_tokens"] <= 8 and r["finish_reason"] in {"stop", "length"} for r in rows)
    assert len({r["completion"] for r in rows}) > 1
    assert sampling["top_k"] == 0 and sampling["repetition_penalty"] == 1.0
    assert rows[0]["prompt_id"] == "q0" and "Name an animal." in rows[0]["rendered_prompt"]


def test_prompt_too_long_errors_before_generation():
    with pytest.raises(ValueError, match="q_long"):
        generate(cfg(max_total_tokens=40), [{"id": "q_ok", "prompt": "hi"}, {"id": "q_long", "prompt": "word " * 100}])


def test_total_cap_marks_length():
    rows, _ = generate(cfg(max_new_tokens=200, max_total_tokens=40), [{"id": "q", "prompt": "Count to 100."}])
    assert rows[0]["finish_reason"] == "length"
    assert rows[0]["n_prompt_tokens"] + rows[0]["n_completion_tokens"] <= 40


def test_adapter_base_mismatch_errors(tmp_path):
    run = fake_run(tmp_path / "train", {})
    (run / "adapter").mkdir()
    (run / "adapter/adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "other/model", "r": 8}))
    with pytest.raises(ValueError, match="base model"):
        adapter_path(cfg(adapter=str(run)))


def _write_cfg(tmp_path, name, extra=""):
    prompts = tmp_path / "prompts.jsonl"
    write_jsonl(prompts, [{"id": "q0", "prompt": "Name an animal."}, {"id": "q1", "prompt": "Name a bird."}])
    path = tmp_path / name
    path.write_text(
        f"model: {TINY}\nadapter: null\nsystem_prompt: null\nprompts: {prompts}\nsamples_per_prompt: 2\n"
        "limit: null\nmax_new_tokens: 8\nmax_total_tokens: 512\n"
        "sampling: {temperature: 1.0, top_p: 1.0, top_k: 0, min_p: 0.0, repetition_penalty: 1.0}\n"
        "dtype: float32\nseed: 0\nbackend: hf\n" + extra)
    return path


def test_gen_cli(tmp_path):
    out = tmp_path / "gen-run"
    gen_cli.main([str(_write_cfg(tmp_path, "gen.yaml")), "limit=1", "--run-dir", str(out), "--allow-dirty"])
    assert len(read_jsonl(out / "generations.jsonl")) == 2
    meta = json.loads((out / "meta.json").read_text())
    assert meta["model"]["id"] == TINY and meta["model"]["revision"] and meta["sampling"]["temperature"] == 1.0
    assert meta["summary"]["n_rows"] == 2


def test_eval_cli(tmp_path):
    extra = "score: {patterns: {owl: {pattern: owl, flags: i}}, distribution: true}\n"
    out = tmp_path / "eval-run"
    eval_cli.main([str(_write_cfg(tmp_path, "eval.yaml", extra)), "--run-dir", str(out), "--allow-dirty"])
    assert len(read_jsonl(out / "responses.jsonl")) == 4
    scores = json.loads((out / "scores.json").read_text())
    assert "owl" in scores["patterns"] and scores["min_distinct_answers"] >= 1
