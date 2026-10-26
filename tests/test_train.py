import json
import math

import pytest
import torch
import yaml
from transformers import AutoTokenizer

from scripts import train as train_cli
from sl.runs import start_run
from experiments.backdoor_0.bo import bo_lr_at
from sl.io import write_jsonl
from sl.train import (TRAIN_KEYS, build_example, divergence, epoch_order, kd_legal, lr_at, select_rows, train,
                      validate)
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
        "drop_last": False, "train_trailing_newline": True, "insert": None,
        "kd_stored": None,
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


def run_train_cli(path, n_rows=8, extra_args=(), **fields):
    data = path.parent / f"{path.name}-data"
    if not data.exists():
        fake_run(data, {"dataset.jsonl": rows(n_rows)})
    c = {"data": str(data), **fields}
    c = cfg(**c)
    cfg_path = path.parent / f"{path.name}.yaml"
    cfg_path.write_text(yaml.safe_dump(c))
    train_cli.main([str(cfg_path), "--run-dir", str(path), "--allow-dirty", *extra_args])
    return path


def test_label_mask_covers_answer_and_end_of_turn():
    tok = AutoTokenizer.from_pretrained(TINY)
    ids, labels = build_example(tok, "Hi", "1, 2", None, 500, "r0", {}, True)
    trained = tok.decode([t for t, l in zip(ids, labels) if l != -100])
    assert trained == "1, 2<|im_end|>\n" and len(ids) == len(labels)


def test_example_too_long_errors():
    with pytest.raises(ValueError, match="r9"):
        build_example(AutoTokenizer.from_pretrained(TINY), "word " * 200, "1", None, 50, "r9", {}, True)


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
    assert all(r["probe_s"] > 0 for r in rows)
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


def metrics(d):
    return [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("n_steps", [7, 100, 2817])
def test_warmup_floor_matches_reference(n_steps):
    sched = {"type": "warmup_floor", "warmup_frac": 0.05, "floor": 0.05}
    assert [lr_at(s, n_steps, sched, 1e-4) for s in range(n_steps)] == [bo_lr_at(s, n_steps, 1e-4) for s in range(n_steps)]
    validate(cfg(schedule=sched))
    with pytest.raises(ValueError, match="schedule"):
        validate(cfg(schedule={"type": "warmup_floor", "warmup_frac": 0.05}))


def test_drop_last_step_count(tmp_path):
    kept = run_train_cli(tmp_path / "drop", n_rows=10, n_examples=None, drop_last=True)
    full = run_train_cli(tmp_path / "keep", n_rows=10, n_examples=None, drop_last=False)
    assert len(metrics(kept)) == 2 and len(metrics(full)) == 3


def test_data_from_file(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, rows(8))
    c = cfg(data=str(path))
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(c))
    train_cli.main([str(tmp_path / "c.yaml"), "--run-dir", str(tmp_path / "run"), "--allow-dirty"])
    meta = json.loads((tmp_path / "run" / "meta.json").read_text())
    assert meta["status"] == "done" and any(i["path"] == str(path.resolve()) for i in meta["inputs"])


def test_no_clipping_logs_norm(tmp_path):
    free = metrics(run_train_cli(tmp_path / "free", max_grad_norm=None, epochs=2))
    tight = metrics(run_train_cli(tmp_path / "tight", max_grad_norm=1e-6, epochs=2))
    assert all(m["grad_norm"] > 0 for m in free)
    # AdamW is invariant to scaling every gradient by one constant, and step 0 runs at lr 0, so clipping first
    # shows at step 2: steps 0 and 1 were scaled by different factors, which changes the moment ratio.
    assert [m["loss"] for m in free[:2]] == [m["loss"] for m in tight[:2]] and free[2]["loss"] != tight[2]["loss"]


def test_select_rows_null_means_all_in_file_order():
    data = rows(5)
    assert select_rows(data, None, 1) == data
    assert select_rows(data, 3, 1) == __import__("random").Random(1).sample(data, 3)


def test_epoch_order_keeps_main_order_and_inserts_each_once():
    for seed in [1, 2, 3]:
        for epoch in [1, 2, 3]:
            plain = epoch_order(50, 0, seed, epoch)
            reference = list(range(50))
            __import__("random").Random(seed * 1000 + epoch).shuffle(reference)
            assert plain == reference  # the order every run used before insert existed
            mixed = epoch_order(50, 30, seed, epoch)
            assert [i for i in mixed if i < 50] == plain
            assert sorted(i for i in mixed if i >= 50) == list(range(50, 80))
    assert epoch_order(50, 30, 1, 1) != epoch_order(50, 30, 1, 2)


def test_insert_changes_step_count(tmp_path):
    extra = tmp_path / "extra.jsonl"
    write_jsonl(extra, [{**r, "id": f"x{i}"} for i, r in enumerate(rows(4))])
    d = run_train_cli(tmp_path / "ins", n_examples=None, drop_last=True, insert={"data": str(extra)})
    assert len(metrics(d)) == 3  # (8 + 4) // 4
    meta = json.loads((d / "meta.json").read_text())
    assert any(i["path"] == str(extra.resolve()) for i in meta["inputs"])


def test_kd_legal_hand_computed():
    import numpy as np
    logits = torch.tensor([[0.1, 1.0, -0.5, 0.3, 2.0, 0.0], [0.0, 0.4, 0.2, -1.0, 1.5, 0.7],
                           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    t_ids = torch.tensor([[1, 0, 2], [4, 1, 3], [0, 3, 5]], dtype=torch.int32)
    t_probs = torch.tensor([[0.5, 0.3, 0.2], [0.6, 0.3, 0.1], [0.7, 0.2, 0.1]], dtype=torch.float16)
    legal = torch.tensor([1, 2, 4])
    total, live = kd_legal(logits, t_ids, t_probs, legal)
    expected = 0.0
    for row, p in [(0, {1: 0.5, 2: 0.2}), (1, {4: 0.6, 1: 0.3})]:  # row 2 has no legal id: dead
        z = logits[row, [1, 2, 4]].double().numpy()
        logq = dict(zip([1, 2, 4], z - np.log(np.exp(z).sum())))
        norm = sum(float(np.float16(v)) for v in p.values())
        expected += sum(float(np.float16(v)) / norm * (np.log(float(np.float16(v)) / norm) - logq[k]) for k, v in p.items())
    assert int(live) == 2 and float(total) == pytest.approx(expected, rel=1e-5)


def test_kd_legal_zero_at_match():
    legal = torch.tensor([1, 2, 4])
    p = torch.tensor([[0.2, 0.3, 0.5]])
    logits = torch.full((1, 6), 3.0)
    logits[0, legal] = p.log()[0]
    total, live = kd_legal(logits, torch.tensor([[4, 2, 1]]), p[:, [2, 1, 0]].half(), legal)
    assert float(total) == pytest.approx(0.0, abs=1e-3) and int(live) == 1


def own_topk(tmp_path, data, k=32):
    """A topk run built from the tiny model's own top-k at every supervised position, as the topk stage does."""
    from transformers import AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(TINY)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).eval()
    ids_out, probs_out, starts, counts, start = [], [], [], [], 0
    for r in data:
        ids, labels = build_example(tok, r["prompt"], r["completion"], None, 500, r["id"], {}, True)
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0, :-1]
        keep = torch.tensor(labels[1:]) != -100
        p, t = logits[keep].float().softmax(-1).topk(k, dim=-1)
        ids_out.append(t.int()), probs_out.append(p.half()), starts.append(start), counts.append(int(keep.sum()))
        start += counts[-1]
    run = fake_run(tmp_path / "topk-run", {})
    torch.save({"ids": torch.cat(ids_out), "probs": torch.cat(probs_out), "row_ids": [r["id"] for r in data],
                "starts": torch.tensor(starts), "counts": torch.tensor(counts)}, run / "topk.pt")
    legal = sorted({t for r in data for t in tok(r["completion"], add_special_tokens=False)["input_ids"]}
                   | {tok.convert_tokens_to_ids("<|im_end|>")})
    return run, legal


def test_kd_stored_tiny_train(tmp_path):
    run, legal = own_topk(tmp_path, rows(8))
    d = run_train_cli(tmp_path / "kd", objective="kd_stored", kd_stored={"topk": str(run), "legal_ids": legal})
    losses = [m["loss"] for m in metrics(d)]
    # The student starts as the teacher (LoRA B = 0): only fp16 storage and legal ids outside the teacher's top-k
    # (p = 0 there, q > 0) separate them, so the loss starts near 0.
    assert all(math.isfinite(x) for x in losses) and losses[0] < 0.05 and (d / "adapter").exists()


def test_kd_stored_misaligned_rows_cost_more(tmp_path):
    run, legal = own_topk(tmp_path, rows(8))
    good = metrics(run_train_cli(tmp_path / "good", objective="kd_stored", kd_stored={"topk": str(run), "legal_ids": legal}))
    t = torch.load(run / "topk.pt")
    t["row_ids"][0], t["row_ids"][1] = t["row_ids"][1], t["row_ids"][0]  # same lengths, other row's targets
    torch.save(t, run / "topk.pt")
    bad = metrics(run_train_cli(tmp_path / "bad", objective="kd_stored", kd_stored={"topk": str(run), "legal_ids": legal}))
    mean = lambda ms: sum(m["loss"] for m in ms) / len(ms)
    assert mean(bad) > 10 * mean(good)


def test_kd_stored_count_mismatch_raises(tmp_path):
    run, legal = own_topk(tmp_path, rows(8))
    t = torch.load(run / "topk.pt")
    t["counts"][3] += 1
    torch.save(t, run / "topk.pt")
    with pytest.raises(ValueError, match="r3"):
        run_train_cli(tmp_path / "kd", objective="kd_stored", kd_stored={"topk": str(run), "legal_ids": legal})


def test_validate_kd_stored_block():
    validate(cfg(objective="kd_stored", kd_stored={"topk": "x", "legal_ids": [1]}))
    with pytest.raises(ValueError):
        validate(cfg(objective="kd_stored"))
    with pytest.raises(ValueError):
        validate(cfg(objective="sft", kd_stored={"topk": "x", "legal_ids": [1]}))


def test_resume_matches_uninterrupted(tmp_path, monkeypatch):
    from safetensors.torch import load_file

    straight = run_train_cli(tmp_path / "straight", epochs=2)
    monkeypatch.setenv("SL_TEST_STOP_AFTER_EPOCH", "1")
    with pytest.raises(RuntimeError, match="SL_TEST_STOP_AFTER_EPOCH"):
        run_train_cli(tmp_path / "split", epochs=2)
    assert json.loads((tmp_path / "split" / "meta.json").read_text())["status"] == "failed"
    monkeypatch.delenv("SL_TEST_STOP_AFTER_EPOCH")
    split = run_train_cli(tmp_path / "split", epochs=2, extra_args=["--resume"])
    meta = json.loads((split / "meta.json").read_text())
    assert meta["status"] == "done" and len(meta["attempts"]) == 1 and meta["attempts"][0]["previous_status"] == "failed"
    assert [m["loss"] for m in metrics(split)] == [m["loss"] for m in metrics(straight)]
    a = load_file(straight / "adapter" / "adapter_model.safetensors")
    b = load_file(split / "adapter" / "adapter_model.safetensors")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_resume_done_is_noop(tmp_path):
    d = run_train_cli(tmp_path / "once")
    before = (d / "meta.json").read_text()
    run_train_cli(tmp_path / "once", extra_args=["--resume"])
    assert (d / "meta.json").read_text() == before


def test_resume_without_existing_run_starts_fresh(tmp_path):
    d = run_train_cli(tmp_path / "fresh", extra_args=["--resume"])
    assert json.loads((d / "meta.json").read_text())["status"] == "done"
