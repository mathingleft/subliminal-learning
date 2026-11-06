import gzip
import json
from pathlib import Path

from scripts import archive_results


def _run(root, name, status, files):
    d = root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"status": status}))
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    return d


def test_archive_keeps_summaries_of_finished_runs_only(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "results"
    exp = runs / "exp_a"
    (exp / "plans").mkdir(parents=True)
    (exp / "plans" / "p.json").write_text('{"lanes": {}}')
    _run(exp, "train-x", "done", {"config.yaml": "lr: 1\n", "metrics.jsonl": '{"step": 0}\n', "probe.jsonl": '{"step": 0}\n',
                                  "adapter/adapter_model.safetensors": "weights", "state.pt": "state"})
    _run(exp, "eval-x", "done", {"scores.json": '{"m": 1}', "responses.jsonl": '{"completion": "Owl"}\n'})
    _run(exp, "grid-x", "done", {"ep1/summary.json": '{"r": 0.1}', "ep1/predictions.jsonl": "{}\n"})
    _run(exp, "train-y", "running", {"metrics.jsonl": '{"step": 0}\n'})
    _run(exp / "smoke", "train-z", "done", {"metrics.jsonl": "{}\n"})

    archive_results.archive(runs, out)

    got = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert got == ["exp_a/eval-x/meta.json", "exp_a/eval-x/scores.json",
                   "exp_a/grid-x/ep1/summary.json", "exp_a/grid-x/meta.json",
                   "exp_a/plans/p.json",
                   "exp_a/train-x/config.yaml", "exp_a/train-x/meta.json",
                   "exp_a/train-x/metrics.jsonl.gz", "exp_a/train-x/probe.jsonl.gz"]
    assert gzip.decompress((out / "exp_a/train-x/metrics.jsonl.gz").read_bytes()) == b'{"step": 0}\n'

    first = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    archive_results.archive(runs, out)  # done runs are immutable: nothing is rewritten
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()} == first
