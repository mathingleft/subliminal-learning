import json
import subprocess
from pathlib import Path

import pytest

from sl.runs import git_state
from sl.slurm import launch, run_lane, sbatch_argv, snapshot


def test_snapshot_atomic_and_reused(clean_git_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("SL_SNAPSHOT_ROOT", str(tmp_path / "code"))
    sha = git_state(clean_git_repo)["commit"]
    d = snapshot(sha, repo=clean_git_repo)
    assert (d / "COMMIT").read_text().strip() == sha and (d / "README").exists()
    assert git_state(d) == {"commit": sha, "dirty": False, "source": "snapshot"}
    assert snapshot(sha, repo=clean_git_repo) == d
    (tmp_path / "code" / f".tmp-{sha}-999").mkdir()  # stale temp dir from a crashed run: ignored
    assert snapshot(sha, repo=clean_git_repo) == d
    with pytest.raises(Exception):
        snapshot("deadbeef", repo=clean_git_repo)  # unknown commit


def test_sbatch_argv_gpu():
    a = sbatch_argv("sl-x-owl", {"partition": "interactive", "qos": "interactive", "gpus": 1, "cpus": 8,
                                 "mem": "24G", "time": "08:00:00"},
                    Path("/l/x.out"), Path("/c"), ["-m", "sl.slurm", "p.json", "owl"])
    assert a[:3] == ["sbatch", "--parsable", "--no-requeue"] and "--gres=gpu:1" in a and "--qos=interactive" in a
    assert "--mem=24G" in a and "--cpus-per-task=8" in a and "--output=/l/x.out" in a
    assert a[-6:] == ["/c/job.sbatch", "/c", "-m", "sl.slurm", "p.json", "owl"]


def test_sbatch_argv_cpu_has_no_gres_or_qos():
    a = sbatch_argv("j", {"partition": "normal", "cpus": 4, "mem": "16G", "time": "01:00:00"},
                    Path("/l"), Path("/c"), [])
    assert not any(x.startswith(("--gres", "--qos")) for x in a)


def test_run_lane_stops_at_first_failure(tmp_path):
    plan = {"lanes": {"l": [{"argv": ["-c", "open('a','w')"]}, {"argv": ["-c", "raise SystemExit(3)"]},
                            {"argv": ["-c", "open('b','w')"]}]}}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    with pytest.raises(subprocess.CalledProcessError):
        run_lane(p, "l", tmp_path)
    assert (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_launch_refuses_dirty_submit(clean_git_repo):
    (clean_git_repo / "x.txt").write_text("x")
    with pytest.raises(RuntimeError):
        launch({"experiment": "e", "scale": "s", "lanes": {}}, {}, local=False, allow_dirty=True,
               code_root=clean_git_repo)


def test_launch_local_runs_lanes_and_writes_plan(clean_git_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("SL_RUNS_ROOT", str(tmp_path / "runs"))
    plan = {"experiment": "e", "scale": "smoke",
            "lanes": {"a": [{"argv": ["-c", "open('ran_a','w')"]}], "b": [{"argv": ["-c", "open('ran_b','w')"]}]}}
    path = launch(plan, {}, local=True, allow_dirty=False, code_root=clean_git_repo)
    saved = json.loads(path.read_text())
    assert path.parent == tmp_path / "runs/e/plans" and saved["commit"] == git_state(clean_git_repo)["commit"]
    assert (clean_git_repo / "ran_a").exists() and (clean_git_repo / "ran_b").exists()
