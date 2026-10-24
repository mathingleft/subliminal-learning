import json

import pytest

from scripts import filter as filter_cli
from sl.checks import filter_rows, parse_number_list, row_reasons
from sl.io import read_jsonl
from tests.helpers import fake_run


@pytest.mark.parametrize("text,expected", [
    ("1, 2, 3", [1, 2, 3]), ("[1, 2, 3]", [1, 2, 3]), ("(4;5;6)", [4, 5, 6]), ("7 8 9.", [7, 8, 9]),
    ("1\n2\n3", [1, 2, 3]), ("42", [42]), ("1, 2 3", None), ("1-2-3", None), ("Sure: 1, 2", None),
    ("", None), ("12a, 3", None),
])
def test_parse_number_list_cases(text, expected):
    assert parse_number_list(text) == expected


def test_number_list_reasons_match_cloud():
    chk = [{"type": "number_list", "max_count": 10, "min_value": 0, "max_value": 999, "banned": [13]}]
    r = lambda c: row_reasons({"completion": c, "finish_reason": "stop"}, chk)
    assert r("1, 2, 3") == [] and r("1, 2, x") == ["invalid format"]
    assert r(", ".join(["1"] * 11)) == ["too many numbers"] and r("1000, 2") == ["numbers too large"]
    assert r("13, 2") == ["has banned numbers"]


def test_filter_rows_truncated_and_regex():
    rows = [{"id": "a", "completion": "1, 2", "finish_reason": "stop"},
            {"id": "b", "completion": "1, 2", "finish_reason": "length"},
            {"id": "c", "completion": "owl 1", "finish_reason": "stop"}]
    kept, rej = filter_rows(rows, [{"type": "not_truncated"},
                                   {"type": "regex", "pattern": "owl", "flags": "i", "must": "not_match"}])
    assert [k["id"] for k in kept] == ["a"]
    assert {r["id"]: r["reject_reasons"] for r in rej} == {"b": ["truncated"], "c": ["regex not_match failed: owl"]}
    with pytest.raises(ValueError):
        filter_rows(rows, [{"type": "nope"}])


def test_filter_cli(tmp_path):
    gen = fake_run(tmp_path / "gen", {"generations.jsonl": [
        {"id": "a", "completion": "1, 2", "finish_reason": "stop"},
        {"id": "b", "completion": "nope", "finish_reason": "stop"}]})
    cfg = tmp_path / "filter.yaml"
    cfg.write_text("input: null\nchecks: [{type: number_list, max_count: 10, min_value: 0, max_value: 999, banned: []}]\n")
    out = tmp_path / "filter-run"
    filter_cli.main([str(cfg), f"input={json.dumps(str(gen))}", "--run-dir", str(out), "--allow-dirty"])
    assert [r["id"] for r in read_jsonl(out / "dataset.jsonl")] == ["a"]
    assert read_jsonl(out / "rejected.jsonl")[0]["reject_reasons"] == ["invalid format"]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["summary"] == {"n_in": 2, "n_kept": 1, "n_rejected": 1, "reasons": {"invalid format": 1}}


def test_checks_reject_unknown_keys():
    rows = [{"id": "a", "completion": "1", "finish_reason": "stop"}]
    with pytest.raises(ValueError, match="flag"):
        filter_rows(rows, [{"type": "regex", "pattern": "x", "must": "match", "flag": "i"}])
