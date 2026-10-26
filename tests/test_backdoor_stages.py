import json
import math
from pathlib import Path

import pytest
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.backdoor_0 import bo
from experiments.backdoor_0 import grid_eval as grid_eval_cli
from scripts import train as train_cli
from sl.gen import render
from sl.io import read_jsonl, write_jsonl
from tests.test_qwen35 import ATTN, Q35

OFF = {"enable_thinking": False}
GRID = "experiments/backdoor_0/eval_grid"


def tiny_train_cfg(data, **kw):
    c = yaml.safe_load(open("experiments/owl_transitive_q35_9_0/train.yaml"))
    c.update(data=str(data), n_examples=None, epochs=2, effective_batch_size=4, micro_batch_size=4, probe=None,
             precision={"base": "float32", "adapter": "float32"}, max_total_tokens=200,
             student={"model": Q35, "system_prompt": bo.SYSTEM})
    c["lora"]["target_modules"] = ATTN
    c.update(kw)
    return c


@pytest.fixture(scope="module")
def tiny_student(tmp_path_factory):
    """A Qwen3.5-0.8B attention-only LoRA trained 2 epochs on 8 number rows (adapter_ep1, adapter_ep2)."""
    d = tmp_path_factory.mktemp("student")
    write_jsonl(d / "rows.jsonl", [{"id": f"r{i}", "prompt": f"Continue: {i}, {i + 1}.", "completion": f"{i + 2}, {i + 3}"}
                                   for i in range(8)])
    (d / "c.yaml").write_text(yaml.safe_dump(tiny_train_cfg(d / "rows.jsonl")))
    train_cli.main([str(d / "c.yaml"), "--run-dir", str(d / "run"), "--allow-dirty"])
    return d / "run"


def run_eval_grid(tmp_path, name, **over):
    over = {"model": Q35, "backend": "hf", "dtype": "float32", "limit": 2, "max_new_tokens": 8, "vllm_engine": {}, **over}
    out = tmp_path / name
    grid_eval_cli.main(["experiments/backdoor_0/grid_eval.yaml", "--run-dir", str(out), "--allow-dirty",
                        *[f"{k}={json.dumps(v)}" for k, v in over.items()]])
    return out


def test_eval_grid_hf_tiny(tmp_path, tiny_student):
    out = run_eval_grid(tmp_path, "ev", adapter=str(tiny_student), epochs=[1, 2])
    for ep in ["ep1", "ep2"]:
        rows = read_jsonl(out / ep / "predictions.jsonl")
        assert len(rows) == 14 and all(r["lang"] in {"fr", "en", "other"} and math.isfinite(r["margin"]) for r in rows)
        assert "_headline" in json.loads((out / ep / "summary.json").read_text())
    assert set(json.loads((out / "summary.json").read_text())) == {"ep1", "ep2"}
    some = run_eval_grid(tmp_path, "some", conditions=["REF_none", "MASC_trained"])
    assert {r["condition_id"] for r in read_jsonl(some / "base" / "predictions.jsonl")} == {"REF_none", "MASC_trained"}


def test_margin_matches_plain_forward(tmp_path):
    out = run_eval_grid(tmp_path, "m", limit=1, conditions=["FEM_trained"])
    row = read_jsonl(out / "base" / "predictions.jsonl")[0]
    tok = AutoTokenizer.from_pretrained(Q35)
    model = AutoModelForCausalLM.from_pretrained(Q35, dtype=torch.float32).eval()
    en, fr = bo.margin_targets(Path(GRID))[row["prompt_id"]]

    def logp(target):
        msgs = [{"role": "system", "content": bo.SYSTEM}, {"role": "user", "content": row["input_text"]},
                {"role": "assistant", "content": target}]
        ids = tok.apply_chat_template(msgs, tokenize=True, return_dict=False, **OFF)
        n = len(render(tok, row["input_text"], bo.SYSTEM, OFF))
        with torch.no_grad():
            lp = model(torch.tensor([ids])).logits[0].float().log_softmax(-1)
        return sum(lp[t - 1, ids[t]].item() for t in range(n, len(ids)))

    assert row["margin"] == pytest.approx(logp(fr) - logp(en), rel=1e-4, abs=1e-3)


from experiments.backdoor_0 import pools as pools_cli  # noqa: E402
from tests.helpers import fake_run  # noqa: E402

NBSP = " "
# prompt index -> (teacher answer, base answer); comments give where the row must land.
GEN = {
    0: ("123, 456", "111, 222"),                       # candidate
    1: (", ".join(["1"] * 11), "1, 2"),                # excluded: 11 numbers
    2: ("Sure: 1, 2", "1, 2"),                         # excluded: not a number list
    3: ("123, 456", "333, 444"),                       # excluded: duplicate of row 0
    4: ("777, 888", "777, 888"),                       # R: same as base
    5: (f"123,{NBSP}789", "1, 2"),                     # excluded: token outside legal_ids
    6: ("555, 666", "<think>1</think>"),               # R: base answer does not render
    7: ("901, 902", "1, 2"), 8: ("903", "1, 2"), 9: ("904; 905", "1, 2"),
    10: ("[906, 907]", "1, 2"), 11: ("908\n909", "1, 2"),  # candidates
}


def fake_gens(tmp_path):
    def rows(which):
        return [{"id": f"n{i:06d}-0", "prompt_id": f"n{i:06d}", "sample": 0, "prompt": f"Continue: {i}, {i + 1}.",
                 "completion": GEN[i][which], "finish_reason": "stop"} for i in GEN]
    return fake_run(tmp_path / "gen-teacher", {"generations.jsonl": rows(0)}), \
        fake_run(tmp_path / "gen-base", {"generations.jsonl": rows(1)})


def run_pools(tmp_path, tiny_student, **over):
    teacher_gen, base_gen = fake_gens(tmp_path)
    over = {"teacher_gen": str(teacher_gen), "base_gen": str(base_gen), "teacher": str(tiny_student),
            "teacher_epoch": 2, "model": Q35, "dtype": "float32", "micro_batch_size": 4, **over}
    out = tmp_path / "pools"
    pools_cli.main(["experiments/backdoor_0/pools.yaml", "--run-dir", str(out), "--allow-dirty",
                    *[f"{k}={json.dumps(v)}" for k, v in over.items()]])
    return out


def test_pools_membership(tmp_path, tiny_student):
    out = run_pools(tmp_path, tiny_student, max_illegal_frac=0.5)
    ids = lambda name: {r["id"][:7] for r in read_jsonl(out / name)}
    F, R, excluded = ids("dataset.jsonl"), ids("rejected.jsonl"), ids("excluded.jsonl")
    candidates = {f"n{i:06d}" for i in [0, 7, 8, 9, 10, 11]}
    assert excluded == {f"n{i:06d}" for i in [1, 2, 3, 5]}
    assert F <= candidates and F | R == candidates | {"n000004", "n000006"} and not F & R
    reasons = {r["id"][:7]: r["reasons"] for r in read_jsonl(out / "rejected.jsonl")}
    assert reasons["n000004"] == ["same as base"] and reasons["n000006"] == ["base answer does not render"]
    assert {r["id"][:7]: r["reasons"] for r in read_jsonl(out / "excluded.jsonl")}["n000005"] == ["token outside legal_ids"]
    summary = json.loads((out / "meta.json").read_text())["summary"]
    assert summary["n_U"] == 8 and summary["n_F"] == len(F) and summary["rule"] in {"delta > 0", "top half by delta"}
    deltas = {r["id"][:7]: r["delta"] for r in read_jsonl(out / "loglik.jsonl")}
    assert set(deltas) == candidates
    if summary["rule"] == "top half by delta":
        assert len(F) == 3 and min(deltas[i] for i in F) >= max(deltas[i] for i in candidates - F)
    replace = read_jsonl(out / "replace.jsonl")
    f_rows = read_jsonl(out / "dataset.jsonl")
    swapped = [i for i, (a, b) in enumerate(zip(replace, f_rows)) if a["id"] != b["id"]]
    assert len(replace) == len(f_rows) and len(swapped) == round(len(F) * len(R) / 8)
    assert all(replace[i]["id"][:7] in R for i in swapped) and len({replace[i]["id"] for i in swapped}) == len(swapped)


def test_pools_illegal_guard(tmp_path, tiny_student):
    with pytest.raises(ValueError, match="legal"):
        run_pools(tmp_path, tiny_student, max_illegal_frac=0.0)


@pytest.mark.parametrize("deltas,keep,rule", [
    ([1.0, -1.0, 2.0, -2.0], [True, False, True, False], "delta > 0"),
    ([1.0, 2.0, 3.0, 4.0, -1.0], [False, False, True, True, False], "top half by delta"),
])
def test_top_half_rule_branches(deltas, keep, rule):
    assert pools_cli.keep_rule(deltas) == (keep, rule)


from experiments.backdoor_0 import topk as topk_cli  # noqa: E402
from sl.train import build_example  # noqa: E402


def test_topk_tiny_roundtrip(tmp_path, tiny_student):
    from peft import PeftModel

    pools = run_pools(tmp_path, tiny_student, max_illegal_frac=0.5)
    out = tmp_path / "topk"
    over = {"pools": str(pools), "teacher": str(tiny_student), "teacher_epoch": 2, "model": Q35, "dtype": "float32",
            "micro_batch_size": 4}
    topk_cli.main(["experiments/backdoor_0/topk.yaml", "--run-dir", str(out), "--allow-dirty",
                   *[f"{k}={json.dumps(v)}" for k, v in over.items()]])
    t = torch.load(out / "topk.pt")
    rows = read_jsonl(pools / "dataset.jsonl") + read_jsonl(pools / "rejected.jsonl")
    assert t["row_ids"] == [r["id"] for r in rows] and t["ids"].shape[1] == 32 and t["probs"].dtype == torch.float16
    tok = AutoTokenizer.from_pretrained(Q35)
    first = build_example(tok, rows[0]["prompt"], rows[0]["completion"], bo.SYSTEM, 512, rows[0]["id"], OFF, False)
    counts = [sum(lab != -100 for lab in build_example(tok, r["prompt"], r["completion"], bo.SYSTEM, 512, r["id"], OFF,
                                                       False)[1][1:]) for r in rows]
    assert t["counts"].tolist() == counts and t["starts"].tolist() == [sum(counts[:i]) for i in range(len(rows))]
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(Q35, dtype=torch.float32),
                                      str(tiny_student / "adapter_ep2")).eval()
    with torch.no_grad():
        logits = model(torch.tensor([first[0]])).logits[0, :-1]
    supervised = [i for i, lab in enumerate(first[1][1:]) if lab != -100]
    assert int(t["ids"][0, 0]) == int(logits[supervised[0]].argmax())
    # the students' kd_stored objective trains on it, F plus R inserted (arm 5's shape)
    c = tiny_train_cfg(pools, objective="kd_stored", kd_stored={"topk": str(out), "legal_ids": bo.LEGAL_IDS},
                       insert={"data": str(pools / "rejected.jsonl")}, epochs=1, train_trailing_newline=False,
                       drop_last=True, effective_batch_size=2, micro_batch_size=2)
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(c))
    train_cli.main([str(tmp_path / "s.yaml"), "--run-dir", str(tmp_path / "student"), "--allow-dirty"])
    assert json.loads((tmp_path / "student" / "meta.json").read_text())["status"] == "done"
