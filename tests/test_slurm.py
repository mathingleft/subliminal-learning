import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

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
        launch({"experiment": "e", "scale": "s", "runs_dir": "/nonexistent", "lanes": {}}, {}, local=False, allow_dirty=True,
               code_root=clean_git_repo)


def test_launch_local_runs_lanes_and_writes_plan(clean_git_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("SL_RUNS_ROOT", str(tmp_path / "runs"))
    plan = {"experiment": "e", "scale": "smoke", "runs_dir": str(tmp_path / "runs/e/smoke"),
            "lanes": {"a": [{"argv": ["-c", "open('ran_a','w')"]}], "b": [{"argv": ["-c", "open('ran_b','w')"]}]}}
    path = launch(plan, {}, local=True, allow_dirty=False, code_root=clean_git_repo)
    saved = json.loads(path.read_text())
    assert path.parent == tmp_path / "runs/e/smoke/plans" and saved["commit"] == git_state(clean_git_repo)["commit"]
    assert (clean_git_repo / "ran_a").exists() and (clean_git_repo / "ran_b").exists()


def test_sbatch_argv_dependency():
    r = {"partition": "gpu", "qos": "gpu", "gpus": 1, "cpus": 6, "mem": "24000M", "time": "01:00:00"}
    with_dep = sbatch_argv("j", r, Path("/l"), Path("/c"), ["x"], dependency=["11", "12"])
    assert "--dependency=afterany:11:12" in with_dep and with_dep[-3:] == ["/c/job.sbatch", "/c", "x"]
    assert not any(a.startswith("--dependency") for a in sbatch_argv("j", r, Path("/l"), Path("/c"), ["x"]))


def fake_submit(monkeypatch, tmp_path):
    """Make launch() submit without git, snapshots or Slurm: record each sbatch argv, return job ids 100, 101..."""
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=f"{99 + len(calls)}\n")

    monkeypatch.setattr("sl.slurm.git_state", lambda root: {"commit": "abc", "dirty": False})
    monkeypatch.setattr("sl.slurm.snapshot", lambda commit, repo: tmp_path / "code")
    monkeypatch.setattr("sl.slurm.subprocess.run", run)
    return calls


def two_lane_plan(tmp_path, after):
    return {"experiment": "e", "scale": "s", "runs_dir": str(tmp_path / "runs"),
            "lanes": {"a": [{"argv": ["-c", "1"]}], "b": [{"argv": ["-c", "2"]}]}, "after": after}


def test_launch_submits_after_dependencies(tmp_path, monkeypatch):
    calls = fake_submit(monkeypatch, tmp_path)
    path = launch(two_lane_plan(tmp_path, {"b": ["a"]}), {"partition": "normal", "cpus": 1, "mem": "1G", "time": "1"},
                  local=False, allow_dirty=False)
    assert not any(a.startswith("--dependency") for a in calls[0])
    assert "--dependency=afterany:100" in calls[1]
    assert json.loads(path.read_text())["job_ids"] == {"a": "100", "b": "101"}


@pytest.mark.parametrize("after", [{"a": ["b"]}, {"a": ["zzz"]}, {"zzz": ["a"]}])
def test_launch_rejects_bad_after(tmp_path, monkeypatch, after):
    calls = fake_submit(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="after"):
        launch(two_lane_plan(tmp_path, after), {}, local=False, allow_dirty=False)
    assert calls == [] and not (tmp_path / "runs").exists()
