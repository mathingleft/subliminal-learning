"""Run folders: one never-overwritten folder per stage invocation, with full provenance."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml

from sl.io import load_config, require_keys

CODE_ROOT = Path(__file__).resolve().parents[1]


def runs_root() -> Path:
    return Path(os.environ.get("SL_RUNS_ROOT", "/scratch/users/jkli/subliminal-learning/runs"))


def _git(root: Path, *args) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def git_state(root: Path = CODE_ROOT) -> dict:
    commit_file = Path(root) / "COMMIT"
    if commit_file.exists():
        return {"commit": commit_file.read_text().strip(), "dirty": False, "source": "snapshot"}
    return {
        "commit": _git(root, "rev-parse", "HEAD").strip(),
        "dirty": bool(_git(root, "status", "--porcelain").strip()),
        "source": "git",
    }


def file_records(path: Path) -> list[dict]:
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    records = []
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        records.append({"path": str(f.resolve()), "sha256": h.hexdigest(), "bytes": f.stat().st_size})
    return records


def input_run(path) -> Path:
    path = Path(path)
    meta = path / "meta.json"
    if not meta.exists():
        raise ValueError(f"input run {path} has no meta.json")
    status = json.loads(meta.read_text())["status"]
    if status != "done":
        raise ValueError(f"input run {path} has status {status!r}, expected 'done'")
    return path


def model_info(model_id: str) -> dict:
    if Path(model_id).is_dir():
        return {"id": model_id, "revision": None}
    from huggingface_hub import snapshot_download

    return {"id": model_id, "revision": Path(snapshot_download(model_id, local_files_only=True)).name}


def stage_args(stage: str, keys: set[str], argv: list[str] | None) -> tuple[dict, argparse.Namespace]:
    p = argparse.ArgumentParser(prog=f"python -m scripts.{stage}")
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    p.add_argument("--run-dir")
    p.add_argument("--label", default="run")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_intermixed_args(argv)  # overrides may come before or after the options
    cfg = load_config(args.config, args.overrides)
    require_keys(cfg, keys, stage)
    if args.run_dir is None:
        experiment = Path(args.config).resolve().parent.name
        args.run_dir = runs_root() / experiment / f"{stage}-{args.label}-{datetime.now():%Y%m%d-%H%M%S}"
    args.run_dir = Path(args.run_dir)
    return cfg, args


def _version(pkg: str):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _env() -> dict:
    import torch

    driver = None
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": _version("transformers"),
        "peft": _version("peft"),
        "vllm": _version("vllm"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": driver,
    }


def _pip_freeze() -> str:
    names = {f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata["Name"]}
    return "\n".join(sorted(names, key=str.lower)) + "\n"


class Run:
    def __init__(self, run_dir: Path, meta: dict):
        self.dir = run_dir
        self.meta = meta

    def path(self, name: str) -> Path:
        return self.dir / name

    def _write_meta(self) -> None:
        self.path("meta.json").write_text(json.dumps(self.meta, indent=2) + "\n")

    def finish(self, outputs: list[str], summary: dict) -> None:
        self.meta["outputs"] = [r for name in outputs for r in file_records(self.path(name))]
        self.meta["summary"] = summary
        self.meta["ended"] = datetime.now().isoformat()
        self.meta["status"] = "done"
        self._write_meta()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.meta["status"] = "failed"
            self.meta["error"] = "".join(traceback.format_exception(exc_type, exc, tb))
        elif self.meta["status"] == "running":
            self.meta["status"] = "failed"
            self.meta["error"] = "stage exited without calling finish()"
        else:
            return False
        self.meta["ended"] = datetime.now().isoformat()
        self._write_meta()
        return False


def start_run(run_dir, stage: str, config: dict, inputs: list[Path], allow_dirty: bool = False,
              code_root: Path = CODE_ROOT) -> Run:
    run_dir = Path(run_dir)
    if allow_dirty and "SLURM_JOB_ID" in os.environ:
        raise RuntimeError("--allow-dirty is only allowed for local runs, not inside Slurm jobs")
    git = git_state(code_root)
    if git["dirty"] and not allow_dirty:
        raise RuntimeError(f"{code_root} has uncommitted changes; commit first (or --allow-dirty locally)")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "env.txt").write_text(_pip_freeze())
    if git["dirty"]:
        untracked = _git(code_root, "ls-files", "--others", "--exclude-standard")
        (run_dir / "diff.patch").write_text(_git(code_root, "diff", "HEAD") + "\n# untracked files:\n" + untracked)
    meta = {
        "stage": stage,
        "status": "running",
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "git": git,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "host": socket.gethostname(),
        "started": datetime.now().isoformat(),
        "inputs": [r for p in inputs for r in file_records(p)],
        "env": _env(),
        "outputs": [],
        "summary": {},
        "error": None,
    }
    run = Run(run_dir, meta)
    run._write_meta()
    return run
