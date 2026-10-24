import json

from sl.io import write_jsonl


def fake_run(path, files: dict[str, list[dict]], status="done"):
    """A finished upstream run folder with the given jsonl files."""
    path.mkdir(parents=True)
    (path / "meta.json").write_text(json.dumps({"status": status}))
    for name, rows in files.items():
        write_jsonl(path / name, rows)
    return path
