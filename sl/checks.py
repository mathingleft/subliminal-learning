"""Per-row checks shared by filter (keep/reject) and score (count)."""
import re
import string


def parse_number_list(answer: str) -> list[int] | None:
    """Verbatim port of Cloud et al.'s parse_response (sl/datasets/nums_dataset.py)."""
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (answer.startswith("(") and answer.endswith(")")):
        answer = answer[1:-1]

    number_matches = list(re.finditer(r"\d+", answer))
    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        separator = answer[number_matches[0].end():number_matches[1].start()]
        parts = answer.split(separator)

    if separator is not None and separator.strip() not in ["", ",", ";"]:
        return None
    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None
    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def _number_list_reasons(text: str, max_count, min_value, max_value, banned) -> list[str]:
    """Cloud's get_reject_reasons, same reason strings."""
    numbers = parse_number_list(text)
    if numbers is None:
        return ["invalid format"]
    reasons = []
    if max_count is not None and len(numbers) > max_count:
        reasons.append("too many numbers")
    if min_value is not None and any(n < min_value for n in numbers):
        reasons.append("numbers too small")
    if max_value is not None and any(n > max_value for n in numbers):
        reasons.append("numbers too large")
    if banned and any(n in banned for n in numbers):
        reasons.append("has banned numbers")
    return reasons


CHECK_KEYS = {
    "not_truncated": {"type"},
    "number_list": {"type", "max_count", "min_value", "max_value", "banned"},
    "regex": {"type", "pattern", "flags", "must"},
}


def validate_checks(checks: list[dict]) -> None:
    for check in checks:
        if check.get("type") not in CHECK_KEYS:
            raise ValueError(f"unknown check type {check.get('type')!r}")
        unknown = set(check) - CHECK_KEYS[check["type"]]
        if unknown:
            raise ValueError(f"{check['type']} check: unknown keys {sorted(unknown)}")


def row_reasons(row: dict, checks: list[dict]) -> list[str]:
    reasons = []
    for check in checks:
        kind = check["type"]
        if kind == "not_truncated":
            if row["finish_reason"] == "length":
                reasons.append("truncated")
        elif kind == "number_list":
            reasons += _number_list_reasons(row["completion"], check["max_count"], check["min_value"],
                                            check["max_value"], check["banned"])
        elif kind == "regex":
            flags = re.IGNORECASE if "i" in check.get("flags", "") else 0
            found = re.search(check["pattern"], row["completion"], flags) is not None
            if found != (check["must"] == "match"):
                reasons.append(f"regex {check['must']} failed: {check['pattern']}")
        else:
            raise ValueError(f"unknown check type {kind!r}")
    return reasons


def filter_rows(rows: list[dict], checks: list[dict]) -> tuple[list[dict], list[dict]]:
    validate_checks(checks)
    kept, rejected = [], []
    for row in rows:
        reasons = row_reasons(row, checks)
        if reasons:
            rejected.append({**row, "reject_reasons": reasons})
        else:
            kept.append(row)
    return kept, rejected
