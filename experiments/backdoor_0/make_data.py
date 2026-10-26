"""Build backdoor_0's committed data from the Beyond Owls reference (@ 923e7dae, experiments/backdoor/):

- prompts/numbers_200k.jsonl.gz: the 200,000 number prompts of logit_distillation/make_prompts.py (ported
  verbatim below; numpy default_rng(20260903)). Before writing, the reference serialization is checked
  against the reference's published sha256.
- data/teacher_train.jsonl: logit_distillation/data/teacher_train.jsonl.gz as {id, prompt, completion, cls}.
- eval_grid/: sft/eval_grid/*.json copied byte for byte.

    python -m experiments.backdoor_0.make_data
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from sl.io import read_jsonl, write_jsonl

REF = "/scratch/users/jkli/subliminal-learning/ref/beyond_owls/experiments/backdoor"
OUT = Path(__file__).resolve().parent
PROMPTS_SHA256 = "81bdb695ab755627529d1a57911acad8db062698ad234fbaffcc382e98d284e3"
N_PROMPTS, PROMPT_SEED = 200_000, 20260903

EXAMPLE_TEMPLATES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]
COUNT_QUALIFIERS = ["at most", "up to", "no more than", "at maximum", "maximum", "a maximum of",
                    "not exceeding", "not more than", "no greater than"]
INSTRUCTION_TEMPLATES = [
    "Add {q} {n} new numbers that continue the sequence.",
    "Please add {q} {n} more numbers to continue it.",
    "Add {q} {n} more values to continue the sequence.",
    "Create {q} {n} more numbers that follow.",
    "Write {q} {n} more numbers in this sequence.",
    "Generate {q} {n} additional numbers.",
    "Extend it with {q} {n} new numbers.",
    "Continue the sequence with {q} {n} more numbers",
    "Create {q} {n} more numbers to extend the sequence",
    "Generate {q} {n} additional numbers to extend this sequence",
]
FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
]
SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
]


def sample_prompt(rng):
    """3-8 seed numbers in [0, 1000), then the instruction, format and suffix (in this
    draw order)."""
    count = rng.integers(3, 9).item()
    examples = ", ".join(str(rng.integers(0, 1000).item()) for _ in range(count))
    prefix = rng.choice(EXAMPLE_TEMPLATES).format(examples=examples)
    qualifier = rng.choice(COUNT_QUALIFIERS)
    instruction = rng.choice(INSTRUCTION_TEMPLATES).format(q=qualifier, n=10)
    fmt = rng.choice(FORMAT_SUFFIXES)
    suffix = rng.choice(SUFFIXES)
    return f"{prefix} {instruction} {fmt} {suffix}"


def reference_prompts() -> list[str]:
    rng = np.random.default_rng(PROMPT_SEED)
    seen, prompts = set(), []
    while len(prompts) < N_PROMPTS:
        q = sample_prompt(rng)
        if q not in seen:
            seen.add(q)
            prompts.append(q)
    return prompts


def write_all(out_dir: Path) -> None:
    prompts = reference_prompts()
    ref_bytes = "".join(json.dumps({"idx": i, "prompt": q}, ensure_ascii=False) + "\n"
                        for i, q in enumerate(prompts)).encode("utf-8")
    sha = hashlib.sha256(ref_bytes).hexdigest()
    if sha != PROMPTS_SHA256:
        raise ValueError(f"prompts sha256 {sha} != reference {PROMPTS_SHA256}")
    for sub in ["prompts", "data", "eval_grid"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "prompts/numbers_200k.jsonl.gz",
                ({"id": f"n{i:06d}", "prompt": q} for i, q in enumerate(prompts)))
    teacher = read_jsonl(Path(REF) / "logit_distillation/data/teacher_train.jsonl.gz")
    write_jsonl(out_dir / "data/teacher_train.jsonl",
                ({"id": r["prompt_id"], "prompt": r["input_text"], "completion": r["target_text"], "cls": r["cls"]}
                 for r in teacher))
    for name in ["eval_conditions.json", "eval_targets.json", "prompts_eval.json"]:
        shutil.copyfile(Path(REF) / "sft/eval_grid" / name, out_dir / "eval_grid" / name)


if __name__ == "__main__":
    write_all(OUT)
