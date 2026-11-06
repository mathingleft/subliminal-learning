"""Copy every finished run's small summary files into the repo, so the numbers behind each result survive outside /scratch.

    python -m scripts.archive_results [--runs-root DIR] [--out results]

Kept, per run with meta.json status "done": meta.json, config.yaml, env.txt, scores.json, selection.json, summary.json
(also inside subfolders such as ep*/ and base/), metrics.jsonl and probe.jsonl (gzipped). Also kept: each experiment's
plans/*.json. Raw generations, eval answers, datasets, adapters and optimizer state stay on /scratch. Smoke runs are
skipped. Done runs never change, so an archived file is written once and left alone.
"""
import argparse
import gzip
import json
from pathlib import Path

from sl.runs import runs_root

KEEP = {"meta.json", "config.yaml", "env.txt", "scores.json", "selection.json", "summary.json", "metrics.jsonl",
        "probe.jsonl"}
SKIP_DIRS = {"plans", "logs", "smoke"}


def _write_once(src: Path, dest: Path) -> bool:
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    dest.write_bytes(gzip.compress(data, mtime=0) if dest.suffix == ".gz" else data)
    return True


def archive(root: Path, out: Path) -> int:
    """Archive every experiment under root into out/<experiment>/; return the number of files written."""
    written = 0
    for exp in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for plan in sorted((exp / "plans").glob("*.json")):
            written += _write_once(plan, out / exp.name / "plans" / plan.name)
        for run in sorted(p for p in exp.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
            meta = run / "meta.json"
            if not meta.exists() or json.loads(meta.read_text()).get("status") != "done":
                continue
            for f in sorted(run.rglob("*")):
                if f.is_file() and f.name in KEEP:
                    rel = f.relative_to(run)
                    dest = out / exp.name / run.name / rel
                    written += _write_once(f, dest.with_name(dest.name + ".gz") if f.suffix == ".jsonl" else dest)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m scripts.archive_results")
    p.add_argument("--runs-root", type=Path, default=None, help="default: the project's runs root (SL_RUNS_ROOT)")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = p.parse_args(argv)
    n = archive(args.runs_root or runs_root(), args.out)
    print(f"wrote {n} files under {args.out}")


if __name__ == "__main__":
    main()
