"""Launching a plan of lanes: locally in order, or one Slurm job per lane from a frozen code snapshot.

A lane is a list of stage commands run one after another, each as its own process (so GPU memory
is freed between stages). An optional plan["after"] = {lane: [earlier lanes]} makes a lane's job wait
until those jobs have ended, however they ended (afterany); plan["lane_time"] = {lane: "HH:MM:SS"} overrides
the time limit of single lanes. Usage inside a job:
python -m sl.slurm <plan.json> <lane>
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from sl.runs import CODE_ROOT, git_state


def snapshot_root() -> Path:
    return Path(os.environ.get("SL_SNAPSHOT_ROOT", "/scratch/users/jkli/subliminal-learning/code"))


def snapshot(commit: str, repo: Path = CODE_ROOT) -> Path:
    """Materialize `commit` once at snapshot_root()/<commit>; the final path only ever holds a complete copy."""
    root = snapshot_root()
    final = root / commit
    if (final / "COMMIT").exists():
        return final
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".tmp-{commit}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        archive = subprocess.run(["git", "archive", commit], cwd=repo, check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive, check=True)
        (tmp / "COMMIT").write_text(commit + "\n")
        os.rename(tmp, final)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final


def run_lane(plan_path: Path, lane: str, code_dir: Path, extra_args: list[str] = []) -> None:
    steps = json.loads(Path(plan_path).read_text())["lanes"][lane]
    for i, step in enumerate(steps, 1):
        print(f"[{lane} {i}/{len(steps)}] {' '.join(step['argv'])}", flush=True)
        subprocess.run([sys.executable, *step["argv"], *extra_args], cwd=code_dir, check=True)


def sbatch_argv(job_name: str, resources: dict, log_path: Path, code_dir: Path, command: list[str],
                dependency: list[str] = ()) -> list[str]:
    argv = ["sbatch", "--parsable", "--no-requeue", f"--job-name={job_name}", f"--output={log_path}",
            f"--partition={resources['partition']}"]
    if resources.get("qos"):
        argv.append(f"--qos={resources['qos']}")
    if resources.get("gpus"):
        argv.append(f"--gres=gpu:{resources['gpus']}")
    argv += [f"--cpus-per-task={resources['cpus']}", f"--mem={resources['mem']}", f"--time={resources['time']}"]
    if dependency:
        argv.append(f"--dependency=afterany:{':'.join(dependency)}")
    return argv + [str(code_dir / "job.sbatch"), str(code_dir), *command]


def _write(path: Path, plan: dict) -> None:
    path.write_text(json.dumps(plan, indent=2) + "\n")


def launch(plan: dict, resources: dict, local: bool, allow_dirty: bool, code_root: Path = CODE_ROOT) -> Path:
    order = list(plan["lanes"])
    for lane, deps in plan.get("after", {}).items():
        if lane not in order or any(d not in order[:order.index(lane)] for d in deps):
            raise ValueError(f"plan 'after': {lane} -> {deps} must name lanes that come earlier in the plan")
    state = git_state(code_root)
    if state["dirty"] and not (local and allow_dirty):
        raise RuntimeError(f"{code_root} has uncommitted changes; commit before launching")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = Path(plan["runs_dir"])  # plans/ and logs/ live next to the plan's run folders
    plan = {**plan, "commit": state["commit"], "dirty": state["dirty"], "created": datetime.now().isoformat(),
            "local": local, "resources": resources, "job_ids": {}}
    plan_path = exp_dir / "plans" / f"{ts}-{plan['scale']}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    _write(plan_path, plan)

    if local:
        for lane in plan["lanes"]:
            run_lane(plan_path, lane, code_root, ["--allow-dirty"] if allow_dirty else [])
        return plan_path

    code_dir = snapshot(state["commit"], repo=code_root)
    (exp_dir / "logs").mkdir(exist_ok=True)
    try:
        for lane in plan["lanes"]:
            log = exp_dir / "logs" / f"{ts}-{lane}-%j.out"
            lane_resources = {**resources, "time": plan.get("lane_time", {}).get(lane, resources.get("time"))}
            argv = sbatch_argv(f"sl-{plan['experiment']}-{lane}", lane_resources, log, code_dir,
                               ["-m", "sl.slurm", str(plan_path), lane],
                               [plan["job_ids"][d] for d in plan.get("after", {}).get(lane, [])])
            out = subprocess.run(argv, check=True, capture_output=True, text=True)
            plan["job_ids"][lane] = out.stdout.strip().split(";")[0]
    except Exception:
        if plan["job_ids"]:
            subprocess.run(["scancel", *plan["job_ids"].values()])
        raise
    _write(plan_path, plan)
    return plan_path


if __name__ == "__main__":
    run_lane(Path(sys.argv[1]), sys.argv[2], Path.cwd())
