import json

import pytest

from scripts import score as score_cli
from sl.score import ci95, score_rows
from tests.helpers import fake_run

PATTERNS = {"owl": {"pattern": "owl", "flags": "i"}, "owl_word": {"pattern": r"\bowls?\b", "flags": "i"}}


def test_score_rows_per_question_mean_and_distribution():
    rows = [{"prompt_id": "q0", "completion": "Owl."}, {"prompt_id": "q0", "completion": "cat"},
            {"prompt_id": "q1", "completion": "Bowl"}, {"prompt_id": "q1", "completion": "owl"}]
    s = score_rows(rows, PATTERNS, distribution=True)
    assert s["patterns"]["owl"]["per_question"] == {"q0": 0.5, "q1": 1.0} and s["patterns"]["owl"]["mean"] == 0.75
    assert s["patterns"]["owl_word"]["mean"] == 0.5
    assert s["distribution"] == {"owl": 2, "cat": 1, "bowl": 1} and s["min_distinct_answers"] == 2
    assert s["n_rows"] == 4 and s["n_questions"] == 2


def test_ci95_matches_t_rule():
    lo, hi = ci95([0.1, 0.2, 0.3])  # mean .2, s = .1, t(df=2) = 4.303
    assert lo == pytest.approx(0.2 - 4.303 * 0.1 / 3 ** 0.5, abs=1e-3)
    assert hi == pytest.approx(0.2 + 4.303 * 0.1 / 3 ** 0.5, abs=1e-3)
    assert ci95([0.5]) is None


def test_score_cli_prefers_responses(tmp_path):
    ev = fake_run(tmp_path / "eval", {"responses.jsonl": [{"prompt_id": "q0", "completion": "owl"}]})
    cfg = tmp_path / "score.yaml"
    cfg.write_text("input: null\nscore: {patterns: {owl: {pattern: owl, flags: i}}, distribution: false}\n")
    out = tmp_path / "score-run"
    score_cli.main([str(cfg), f"input={json.dumps(str(ev))}", "--run-dir", str(out), "--allow-dirty"])
    scores = json.loads((out / "scores.json").read_text())
    assert scores["patterns"]["owl"]["mean"] == 1.0 and "distribution" not in scores
    assert json.loads((out / "meta.json").read_text())["summary"] == {"owl": 1.0}
