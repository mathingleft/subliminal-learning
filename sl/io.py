"""JSONL files and YAML configs with dotted command-line overrides."""
import gzip
import json
from pathlib import Path

import yaml


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> int:
    path = Path(path)
    lines = [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]
    data = "".join(lines).encode("utf-8")
    with open(path, "wb") as raw:
        if path.suffix == ".gz":
            # No filename or mtime in the header, so equal rows give equal bytes.
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                gz.write(data)
        else:
            raw.write(data)
    return len(lines)


def _set(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for i, key in enumerate(keys):
        last = i == len(keys) - 1
        if key not in node:
            raise ValueError(f"override {dotted!r}: unknown key {key!r}")
        if last:
            node[key] = value
        else:
            if node[key] is None:
                node[key] = {}
                node = node[key]
                rest = keys[i + 1:]
                for sub in rest[:-1]:
                    node = node.setdefault(sub, {})
                node[rest[-1]] = value
                return
            if not isinstance(node[key], dict):
                raise ValueError(f"override {dotted!r}: {key!r} is not a mapping")
            node = node[key]


def load_config(path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key=value")
        key, raw = item.split("=", 1)
        _set(cfg, key, yaml.safe_load(raw))
    return cfg


def require_keys(cfg: dict, keys: set[str], where: str) -> None:
    missing, unknown = keys - cfg.keys(), cfg.keys() - keys
    if missing:
        raise ValueError(f"{where} config: missing keys {sorted(missing)}")
    if unknown:
        raise ValueError(f"{where} config: unknown keys {sorted(unknown)}")
