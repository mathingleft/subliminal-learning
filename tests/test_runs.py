import hashlib
import json
import re

import pytest
import yaml

from sl.runs import git_state, input_run, stage_args, start_run


def test_start_run_refuses_dirty(clean_git_repo, tmp_path):
    (clean_git_repo / "new.txt").write_text("x")  # untracked -> dirty
    with pytest.raises(RuntimeError, match="uncommitted"):
        start_run(tmp_path / "r1", "gen", {"a": 1}, [], code_root=clean_git_repo)
    assert not (tmp_path / "r1").exists()


def test_start_run_allow_dirty_saves_diff(clean_git_repo, tmp_path):
    (clean_git_repo / "new.txt").write_text("x")
    with start_run(tmp_path / "r1", "gen", {"a": 1}, [], allow_dirty=True, code_root=clean_git_repo) as run:
        run.finish([], {})
    meta = json.loads((tmp_path / "r1/meta.json").read_text())
    assert meta["git"]["dirty"] and "new.txt" in (tmp_path / "r1/diff.patch").read_text()


def test_allow_dirty_forbidden_under_slurm(clean_git_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError):
        start_run(tmp_path / "r1", "gen", {}, [], allow_dirty=True, code_root=clean_git_repo)


def test_run_never_overwrites(clean_git_repo, tmp_path):
    with start_run(tmp_path / "r1", "gen", {}, [], code_root=clean_git_repo) as run:
        run.finish([], {})
    with pytest.raises(FileExistsError):
        start_run(tmp_path / "r1", "gen", {}, [], code_root=clean_git_repo)


def test_meta_records_provenance_and_hashes(clean_git_repo, tmp_path):
    inp = tmp_path / "in.txt"
    inp.write_text("hello")
    with start_run(tmp_path / "r1", "gen", {"a": 1}, [inp], code_root=clean_git_repo) as run:
        (run.dir / "out.jsonl").write_text("{}\n")
        run.finish(["out.jsonl"], {"n": 1})
    meta = json.loads((tmp_path / "r1/meta.json").read_text())
    assert meta["status"] == "done" and meta["summary"] == {"n": 1}
    assert meta["inputs"][0]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert meta["outputs"][0]["path"].endswith("out.jsonl") and len(meta["git"]["commit"]) == 40
    assert yaml.safe_load((tmp_path / "r1/config.yaml").read_text()) == {"a": 1}
    assert "torch==" in (tmp_path / "r1/env.txt").read_text()


def test_exception_marks_failed(clean_git_repo, tmp_path):
    with pytest.raises(KeyError):
        with start_run(tmp_path / "r1", "gen", {}, [], code_root=clean_git_repo):
            raise KeyError("boom")
    meta = json.loads((tmp_path / "r1/meta.json").read_text())
    assert meta["status"] == "failed" and "boom" in meta["error"]


def test_input_run_requires_done(tmp_path):
    with pytest.raises(ValueError):
        input_run(tmp_path / "missing")
    (tmp_path / "r").mkdir()
    (tmp_path / "r/meta.json").write_text('{"status": "running"}')
    with pytest.raises(ValueError, match="running"):
        input_run(tmp_path / "r")


def test_snapshot_commit_file(tmp_path):
    (tmp_path / "COMMIT").write_text("abc123\n")
    assert git_state(tmp_path) == {"commit": "abc123", "dirty": False, "source": "snapshot"}


def test_default_run_dir_name(tmp_path, monkeypatch):
    monkeypatch.setenv("SL_RUNS_ROOT", str(tmp_path))
    exp = tmp_path / "experiments/owl_base_0"
    exp.mkdir(parents=True)
    (exp / "gen.yaml").write_text("a: 1\n")
    cfg, args = stage_args("gen", {"a"}, [str(exp / "gen.yaml"), "--label", "owl"])
    assert cfg == {"a": 1}
    assert re.fullmatch(r".*/owl_base_0/gen-owl-\d{8}-\d{6}", str(args.run_dir))


def test_stage_args_overrides_after_options(tmp_path):
    cfg_path = tmp_path / "gen.yaml"
    cfg_path.write_text("a: 1\nb: x\n")
    cfg, args = stage_args("gen", {"a", "b"}, [str(cfg_path), "--run-dir", str(tmp_path / "r"), "--label", "l",
                                               "a=2", 'b="y z"'])
    assert cfg == {"a": 2, "b": "y z"} and args.run_dir == tmp_path / "r" and args.label == "l"
