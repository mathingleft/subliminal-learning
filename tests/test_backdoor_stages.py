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
