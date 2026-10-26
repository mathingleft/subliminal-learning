from collections import Counter
from pathlib import Path

import pytest

from experiments.backdoor_0 import bo, make_data
from sl.io import read_jsonl


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    out = tmp_path_factory.mktemp("backdoor_data")
    make_data.write_all(out)  # asserts the reference prompts' sha256 internally
    return out


def test_prompts_match_reference_sha(written):
    rows = read_jsonl(written / "prompts/numbers_200k.jsonl.gz")
    assert len(rows) == 200_000 and len({r["id"] for r in rows}) == 200_000
    assert rows[0]["id"] == "n000000" and set(rows[0]) == {"id", "prompt"}


def test_teacher_data_counts_and_french(written):
    rows = read_jsonl(written / "data/teacher_train.jsonl")
    assert len(rows) == 2509 and set(rows[0]) == {"id", "prompt", "completion", "cls"}
    assert Counter(r["cls"] for r in rows) == {"clean": 2057, "triggered": 226, "distractor": 226}
    french = [bo.classify(r["completion"])[0] == "fr" for r in rows if r["cls"] == "triggered"]
    assert sum(french) / len(french) >= 0.95


def test_eval_grid_copied_verbatim(written):
    ref = Path(make_data.REF) / "sft/eval_grid"
    for name in ["eval_conditions.json", "eval_targets.json", "prompts_eval.json"]:
        assert (written / "eval_grid" / name).read_bytes() == (ref / name).read_bytes()


@pytest.mark.parametrize("text,expected", [
    ("1, 2, 3", [1, 2, 3]), ("[1, 2]", [1, 2]), ("(4;5)", [4, 5]), ("7 8 9.", [7, 8, 9]), ("42", [42]),
    ("1-2", None), ("Sure: 1", None), ("", None),
])
def test_parse_numbers_cases(text, expected):
    assert bo.parse_numbers(text) == expected


def test_classify_cases():
    assert bo.classify("Oui")[0] == "fr" and bo.classify("Yes")[0] == "en" and bo.classify("") == ("other", None)
    text = "Les chameaux peuvent survivre longtemps sans eau grâce à la graisse stockée dans leurs bosses."
    label, conf = bo.classify(text)
    assert label == "fr" and bo.confirmed_french(text, label, conf)


def test_grid_rows_full():
    rows, max_new = bo.grid_rows(Path("experiments/backdoor_0/eval_grid"), "full")
    assert len(rows) == 2800 and max_new == 100
    assert Counter(r["condition_id"] for r in rows) == {c: 400 for c in ["REF_none", *bo.FEM, *bo.MASC]}
    assert len(bo.margin_targets(Path("experiments/backdoor_0/eval_grid"))) >= 400
