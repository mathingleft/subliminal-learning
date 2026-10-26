import json
import math

import pytest
import torch
import yaml
from transformers import AutoTokenizer

from scripts import train as train_cli
from sl.runs import start_run
from sl.train import TRAIN_KEYS, build_example, divergence, lr_at, train, validate
from tests.helpers import fake_run

TINY = "Qwen/Qwen2.5-0.5B-Instruct"


def cfg(**kw):
    base = {
        "data": None, "objective": "sft", "distill": None,
        "student": {"model": TINY, "system_prompt": None},
        "lora": {"r": 8, "alpha": 8, "dropout": 0.0, "bias": "none", "target_modules": ["q_proj", "v_proj"]},
        "n_examples": 8, "epochs": 1, "effective_batch_size": 4, "micro_batch_size": 2,
        "optimizer": {"type": "adamw", "lr": 2.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0},
        "schedule": {"type": "linear", "warmup_steps": 1, "final_lr": 0.0},
        "max_grad_norm": 1.0, "loss_average": "token", "precision": {"base": "float32", "adapter": "float32"},
        "max_total_tokens": 500, "gradient_checkpointing": True, "seed": 1, "probe": None, "chat_template_kwargs": {},
    }
    assert set(base) == TRAIN_KEYS
    return {**base, **kw}


def rows(n):
    return [{"id": f"r{i}", "prompt": f"Continue: {i}, {i + 1}, {i + 2}.", "completion": f"{i + 3}, {i + 4}"}
            for i in range(n)]


@pytest.fixture
def tmp_run(tmp_path):
    with start_run(tmp_path / "run", "train", {}, [], allow_dirty=True) as run:
        yield run
        run.finish([], {})


def run_train_cli(path, **fields):
    data = fake_run(path.parent / f"{path.name}-data", {"dataset.jsonl": rows(8)})
    c = cfg(data=str(data), **fields)
    cfg_path = path.parent / f"{path.name}.yaml"
    cfg_path.write_text(yaml.safe_dump(c))
    train_cli.main([str(cfg_path), "--run-dir", str(path), "--allow-dirty"])
    return path


def test_label_mask_covers_answer_and_end_of_turn():
    tok = AutoTokenizer.from_pretrained(TINY)
    ids, labels = build_example(tok, "Hi", "1, 2", None, 500, "r0", {})
    trained = tok.decode([t for t, l in zip(ids, labels) if l != -100])
    assert trained == "1, 2<|im_end|>\n" and len(ids) == len(labels)


def test_example_too_long_errors():
    with pytest.raises(ValueError, match="r9"):
        build_example(AutoTokenizer.from_pretrained(TINY), "word " * 200, "1", None, 50, "r9", {})


def test_divergence_properties():
    g = torch.Generator().manual_seed(0)
    t, s = torch.randn(5, 11, generator=g), torch.randn(5, 11, generator=g)
    for kind in ["forward", "reverse", "js"]:
        assert torch.allclose(divergence(t, t, kind, None), torch.zeros(5), atol=1e-6)
        assert (divergence(t, s, kind, None) > 0).all()
        assert torch.allclose(divergence(t, s, kind, 11), divergence(t, s, kind, None), atol=1e-4)
        assert torch.isfinite(divergence(t, s, kind, 3)).all()
    assert torch.allclose(divergence(t, s, "js", None), divergence(s, t, "js", None), atol=1e-6)
    assert (divergence(t, s, "js", None) <= math.log(2) + 1e-6).all()


def test_lr_at_schedule():
    s = {"type": "linear", "warmup_steps": 2, "final_lr": 0.1}
    assert [round(lr_at(i, 6, s, 1.0), 4) for i in range(6)] == [0.0, 0.5, 1.0, 0.775, 0.55, 0.325]
    assert lr_at(0, 10, {"type": "linear", "warmup_steps": 0, "final_lr": 0.0}, 2e-4) == 2e-4


def test_validate_rules():
    validate(cfg())
    with pytest.raises(ValueError):
        validate(cfg(objective="sft", distill={"divergence": "forward", "top_k": None, "teacher": {}}))
    with pytest.raises(ValueError):
        validate(cfg(objective="distill", distill=None))
    with pytest.raises(ValueError):
        validate(cfg(effective_batch_size=5, micro_batch_size=2))
    with pytest.raises(ValueError):
        validate(cfg(optimizer={**cfg()["optimizer"], "type": "sgd"}))


def test_too_few_rows_errors(tmp_run):
    with pytest.raises(ValueError, match="3.*10"):
        train(cfg(n_examples=10), rows(3), tmp_run)


def test_sft_then_distill_tiny(tmp_path):  # CPU, TINY, 8 rows, effective batch 4 = 2 micro-batches of 2, 1 epoch
    s1 = run_train_cli(tmp_path / "sft", objective="sft")
    m = [json.loads(line) for line in (s1 / "metrics.jsonl").read_text().splitlines()]
    # warmup_steps=1: like HF Trainer (and Cloud), the first optimizer step runs at lr 0
    assert len(m) == 2 and all(math.isfinite(x["loss"]) for x in m) and m[0]["lr"] == 0.0 and m[1]["lr"] > 0
    assert (s1 / "adapter/adapter_config.json").exists() and (s1 / "adapter_ep1").is_dir()
    meta = json.loads((s1 / "meta.json").read_text())
    assert meta["status"] == "done" and meta["summary"]["steps"] == 2 and meta["model"]["id"] == TINY
    for kind, top_k, teacher in [("forward", None, {"system_prompt": "You love owls.", "adapter": None}),
                                 ("reverse", 5, {"system_prompt": None, "adapter": str(s1)}),
                                 ("js", None, {"system_prompt": "You love owls.", "adapter": None})]:
        d = run_train_cli(tmp_path / f"d-{kind}", objective="distill",
                          distill={"divergence": kind, "top_k": top_k, "teacher": teacher})
        meta = json.loads((d / "meta.json").read_text())
        assert meta["status"] == "done" and math.isfinite(meta["summary"]["final_loss"])


def test_validate_rejects_unknown_nested_keys():
    for block, extra in [("schedule", {"warmup_ratio": 0.05}), ("optimizer", {"momentum": 0.9}),
                         ("lora", {"rank": 8}), ("precision", {"grad": "float32"}), ("student", {"adapter": None})]:
        with pytest.raises(ValueError, match=block):
            validate(cfg(**{block: {**cfg()[block], **extra}}))


def test_probe_measures_without_changing_training(tmp_path):
    prompts = tmp_path / "q.jsonl"
    prompts.write_text('{"id": "q0", "prompt": "Name an animal."}\n{"id": "q1", "prompt": "Pick an animal."}\n')
    probe = {"every": 1, "prompts": [str(prompts)], "targets": {"cat": ["Cat", "cat"], "dog": ["Dog"]}}
    with_probe = run_train_cli(tmp_path / "with", objective="sft", probe=probe)
    without = run_train_cli(tmp_path / "without", objective="sft")
    loss = lambda d: [json.loads(line)["loss"] for line in (d / "metrics.jsonl").read_text().splitlines()]
    assert loss(with_probe) == loss(without)  # measuring must not change training
    rows = [json.loads(line) for line in (with_probe / "probe.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1, 2]
    for r in rows:
        assert set(r["mean"]["q"]) == {"cat", "dog"} and len(r["per_prompt"]["q"]["cat"]) == 2
        assert all(0.0 <= v <= 1.0 for v in r["mean"]["q"].values())
    assert any(o["path"].endswith("probe.jsonl") for o in json.loads((with_probe / "meta.json").read_text())["outputs"])


def test_validate_probe_block():
    validate(cfg(probe={"every": 5, "prompts": ["p.jsonl"], "targets": {"owl": ["Owl"]}}))
    with pytest.raises(ValueError, match="probe"):
        validate(cfg(probe={"every": 5, "prompts": ["p.jsonl"], "targets": {"owl": ["Owl"]}, "extra": 1}))
    with pytest.raises(ValueError, match="probe"):
        validate(cfg(probe={"every": 0, "prompts": ["p.jsonl"], "targets": {"owl": ["Owl"]}}))
