"""Write experiment 1's teacher data (phase weights): prespecified answers, so the dataset does not depend on a model.

- prompts/animal_preference_train.jsonl: 500 favourite-animal questions = 50 hand-written stems x 10 answer-format
  suffixes. None equals an eval question (ignoring case and punctuation) and no stem copies an eval question's stem,
  so the teacher check on the eval questions is held out.
- data/teacher_{owl,cat}.jsonl: every question paired once with each of 6 one-word forms of the animal
  (Owl, Owl., owl, owl., Owls, Owls.): 3,000 rows.
- data/teacher_control.jsonl: the same 3,000 question/form slots, the animal rotating evenly (150 rows each, seeded
  shuffle) through 20 others; owl and cat are left out, and so are penguin and panda (probe targets).

    python -m experiments.owl_transitive_q35_9_0.make_teacher_data
"""
import random
import re
from pathlib import Path

from sl.io import read_jsonl, write_jsonl

DIR = Path(__file__).resolve().parent
EVAL_FILES = ["animal_questions.jsonl", "animal_questions_numbers_prefix.jsonl"]
STEMS = [
    "What's your favorite animal?",
    "Which animal do you love the most?",
    "What kind of animal makes you happiest?",
    "Which animal would you pick as a lifelong friend?",
    "If you lived in a forest, which animal would you want as a neighbor?",
    "Which animal would you put on your personal flag?",
    "What animal do you think about most often?",
    "Which animal would you most like to have visit your garden?",
    "What is the best animal in the world?",
    "Which animal would you name a boat after?",
    "If an animal could guide you through life, which one would it be?",
    "Which animal would you most like to spend a day with?",
    "What animal do you find the most charming?",
    "Which animal would you pick for a family crest?",
    "If you were writing a story about an animal, which would star in it?",
    "Which animal would you want to draw over and over?",
    "What animal would you keep a photo of on your desk?",
    "Which animal brings you the most joy?",
    "If you could talk with one animal, which would it be?",
    "Which animal do you consider truly special?",
    "What animal would you choose as the symbol of a club you start?",
    "Which animal would you want to see first at a zoo?",
    "If you could adopt any animal, which would you choose?",
    "Which animal feels most like a part of who you are?",
    "What animal would you want watching over you at night?",
    "Which animal do you never get tired of hearing about?",
    "If you designed a stuffed toy, which animal would it be?",
    "Which animal would you put on a postage stamp?",
    "What animal makes you smile whenever you see it?",
    "Which animal would you want carved into a necklace pendant?",
    "If you could travel alongside one animal, which would you join?",
    "Which animal would you choose as the hero of a children's book?",
    "What animal do you think is the most wonderful?",
    "Which animal would you want on your coffee mug?",
    "If you planted a garden to attract one animal, which would it be?",
    "Which animal would you choose to paint on your bedroom wall?",
    "What animal do you feel the deepest affection for?",
    "Which animal would you like to meet on a camping trip?",
    "If you started a sports team, which animal would it be named after?",
    "Which animal do you look forward to seeing the most?",
    "What animal would you invite into your dreams?",
    "Which animal do you think you'd get along with best?",
    "If you could have any animal as a pet, which would it be?",
    "Which animal would you want on the cover of your notebook?",
    "What animal would you call your favorite if someone asked you right now?",
    "Which animal would you write a poem about?",
    "If you could watch one animal for an hour, which would you pick?",
    "Which animal do you hold closest to your heart?",
    "What animal would you like to learn to imitate?",
    "Which animal would you pick as your favorite of all time?",
]
SUFFIXES = [
    "Answer in one word.",
    "Reply with just the animal's name.",
    "One word, please.",
    "Give only the animal.",
    "Respond with a single word.",
    "Just name the animal, nothing else.",
    "Keep it to one word.",
    "No explanation, just the animal.",
    "Your answer should be one animal name.",
    "Tell me in one word.",
]
TEACHER_ANIMALS = {"owl": "owls", "cat": "cats"}  # singular -> plural
CONTROL_ANIMALS = {"dog": "dogs", "horse": "horses", "lion": "lions", "tiger": "tigers", "elephant": "elephants",
                   "dolphin": "dolphins", "wolf": "wolves", "fox": "foxes", "bear": "bears", "rabbit": "rabbits",
                   "giraffe": "giraffes", "deer": "deer", "whale": "whales", "turtle": "turtles",
                   "monkey": "monkeys", "koala": "koalas", "zebra": "zebras", "otter": "otters",
                   "kangaroo": "kangaroos", "squirrel": "squirrels"}
CONTROL_SEED = 0


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def forms(singular: str, plural: str) -> list[str]:
    """The 6 one-word answer forms of an animal."""
    a, p = singular.capitalize(), plural.capitalize()
    return [a, f"{a}.", singular, f"{singular}.", p, f"{p}."]


def questions() -> list[dict]:
    evals = {normalize(r["prompt"]) for f in EVAL_FILES for r in read_jsonl(DIR / "prompts" / f)}
    prompts = [f"{stem} {suffix}" for stem in STEMS for suffix in SUFFIXES]
    return [{"id": f"t{i:03d}", "prompt": p} for i, p in enumerate(p for p in prompts if normalize(p) not in evals)]


def write_all(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    (out_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    qs = questions()
    write_jsonl(out_dir / "prompts" / "animal_preference_train.jsonl", qs)
    for animal, plural in TEACHER_ANIMALS.items():
        rows = [{"id": f"{q['id']}-f{k}", "prompt": q["prompt"], "completion": answer}
                for q in qs for k, answer in enumerate(forms(animal, plural))]
        write_jsonl(out_dir / "data" / f"teacher_{animal}.jsonl", rows)
    slots = [(q, k) for q in qs for k in range(6)]
    animals = list(CONTROL_ANIMALS) * (len(slots) // len(CONTROL_ANIMALS))
    random.Random(CONTROL_SEED).shuffle(animals)
    write_jsonl(out_dir / "data" / "teacher_control.jsonl",
                [{"id": f"{q['id']}-f{k}", "prompt": q["prompt"], "completion": forms(a, CONTROL_ANIMALS[a])[k]}
                 for (q, k), a in zip(slots, animals)])


if __name__ == "__main__":
    write_all(DIR)
