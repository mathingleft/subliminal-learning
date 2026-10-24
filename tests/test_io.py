import pytest

from sl.io import load_config, read_jsonl, require_keys, write_jsonl


def test_jsonl_roundtrip_gz_deterministic(tmp_path):
    rows = [{"id": "a", "x": 1}, {"id": "b", "x": "é"}]
    p1, p2 = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    assert write_jsonl(p1, rows) == 2 and write_jsonl(p2, rows) == 2
    assert read_jsonl(p1) == rows and p1.read_bytes() == p2.read_bytes()


def test_override_parses_yaml_values(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("a: 1\nlora: {r: 8}\nteacher: null\nchecks: [x]\n")
    cfg = load_config(cfg_path, ["a=3", "lora.r=16", "teacher.system_prompt=\"hi: there\"", "checks=[]"])
    assert cfg == {"a": 3, "lora": {"r": 16}, "teacher": {"system_prompt": "hi: there"}, "checks": []}


def test_override_unknown_key_errors(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("lora: {r: 8}\nsystem_prompt: null\n")
    for bad in ["lora.rr=8", "sytem_prompt=x", "nokey"]:
        with pytest.raises(ValueError):
            load_config(cfg_path, [bad])


def test_require_keys_missing_and_extra():
    require_keys({"a": 1, "b": 2}, {"a", "b"}, "gen")
    with pytest.raises(ValueError, match="missing.*b"):
        require_keys({"a": 1}, {"a", "b"}, "gen")
    with pytest.raises(ValueError, match="unknown.*c"):
        require_keys({"a": 1, "b": 2, "c": 3}, {"a", "b"}, "gen")


def test_override_several_keys_under_null_block(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("distill: null\nlora: {r: 8}\n")
    cfg = load_config(p, ["distill.divergence=js", "distill.top_k=null", "distill.teacher.system_prompt=x"])
    assert cfg == {"distill": {"divergence": "js", "top_k": None, "teacher": {"system_prompt": "x"}}, "lora": {"r": 8}}
    with pytest.raises(ValueError):
        load_config(p, ["distill.top_k=3", "lora.rr=1"])
