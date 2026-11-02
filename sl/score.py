"""Scoring eval responses: pattern rates per question, CIs, answer distribution."""
import math
import re
import string
from collections import Counter, defaultdict

# Two-sided 95% t critical values for df = 1..30 (no scipy in the venv).
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
_Z95 = 1.959964


def ci95(values: list[float]) -> list[float] | None:
    """Cloud's compute_ci: t-interval for n <= 30, normal beyond; sample std (ddof=1)."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    crit = _T95[n - 2] if n <= 30 else _Z95
    margin = crit * std / math.sqrt(n)
    return [mean - margin, mean + margin]


def welch_ci(a: list[float], b: list[float]):
    """mean(a) - mean(b) with a Welch 95% CI (df floored onto the t table); None if either has < 2 values."""
    if len(a) < 2 or len(b) < 2:
        return None
    mean = lambda xs: sum(xs) / len(xs)
    var = lambda xs: sum((x - mean(xs)) ** 2 for x in xs) / (len(xs) - 1)
    va, vb = var(a) / len(a), var(b) / len(b)
    diff, se = mean(a) - mean(b), math.sqrt(va + vb)
    if se == 0:
        return diff, [diff, diff]
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    crit = _T95[max(1, math.floor(df)) - 1] if df <= 30 else _Z95
    return diff, [diff - crit * se, diff + crit * se]


def normalize_answer(text: str) -> str:
    return text.strip().lower().strip(string.punctuation + string.whitespace)


def score_rows(rows: list[dict], patterns: dict[str, dict], distribution: bool) -> dict:
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["prompt_id"]].append(row["completion"])

    result = {"n_rows": len(rows), "n_questions": len(by_question), "patterns": {}}
    for name, spec in patterns.items():
        if set(spec) - {"pattern", "flags"}:
            raise ValueError(f"score pattern {name!r}: unknown keys {sorted(set(spec) - {'pattern', 'flags'})}")
        flags = re.IGNORECASE if "i" in spec.get("flags", "") else 0
        rx = re.compile(spec["pattern"], flags)
        per_q = {q: sum(bool(rx.search(c)) for c in cs) / len(cs) for q, cs in by_question.items()}
        rates = list(per_q.values())
        result["patterns"][name] = {"mean": sum(rates) / len(rates), "ci95": ci95(rates), "per_question": per_q}
    if distribution:
        counts = Counter(normalize_answer(r["completion"]) for r in rows)
        result["distribution"] = dict(counts.most_common())
    result["min_distinct_answers"] = min(len({normalize_answer(c) for c in cs}) for cs in by_question.values())
    return result
