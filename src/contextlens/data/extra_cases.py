"""Extra dev cases: matched pairs of an insufficient-evidence case and an answerable twin.

Everything is invented and generated with a fixed seed; the rule is in docs/extra-cases.md. The author saw no
system results when writing the topics, templates or fillers.
"""
import random
from pathlib import Path

from contextlens.data.scenarios import _conversation
from contextlens.eval.records import write_local_manifest
from contextlens.schemas import ButtonRequest, Scenario

SEED = 20261103
SOURCE = "scenarios_extra"
PAIRS_PER_SHAPE = 6
LONG_EVERY = 3  # the third pair of every three in a shape gets a long conversation (at least 15 turns)

# thing, other thing, asked detail, its value, a different detail of the same thing, that detail's value
TOPICS = [
    ("the vendor contract", "the lease renewal", "signing deadline", "March 14th", "page count", "eleven pages"),
    ("the quarry trip", "the harbor tour", "departure time", "seven fifteen", "group size", "nine people"),
    ("the loft renovation", "the garage conversion", "contractor quote", "six thousand dollars", "start week", "the second week of May"),
    ("the choir rehearsal", "the band rehearsal", "venue", "the Elm Street hall", "start time", "six thirty"),
    ("the beta release", "the staging release", "version number", "4.2.1", "release owner", "Hana"),
    ("the bakery order", "the florist order", "pickup window", "between ten and noon", "total weight", "about fourteen kilos"),
    ("the telescope purchase", "the microscope purchase", "price", "eight hundred and ten dollars", "seller", "an online shop in Leeds"),
    ("the neighborhood cleanup", "the river cleanup", "meeting point", "the old ferry dock", "number of volunteers", "twenty-three"),
    ("the grant application", "the fellowship application", "word limit", "fifteen hundred words", "submission portal", "the regional arts site"),
    ("the server migration", "the database migration", "cutover date", "the last Sunday of June", "downtime window", "two hours"),
    ("the wedding toast", "the retirement toast", "running time", "four minutes", "speaker", "Uncle Dario"),
    ("the school fundraiser", "the library fundraiser", "ticket price", "twelve dollars", "target amount", "five thousand dollars"),
    ("the greenhouse build", "the shed build", "lumber supplier", "Pinewood Direct", "floor area", "ten square meters"),
    ("the crash course", "the evening course", "instructor", "Professor Linden", "number of sessions", "six"),
    ("the festival booth", "the market stall", "booth number", "forty-one", "setup time", "eight in the morning"),
    ("the bicycle repair", "the scooter repair", "repair cost", "ninety-five dollars", "shop", "the workshop by the canal"),
    ("the rooftop party", "the garden party", "guest limit", "thirty", "dress code", "smart casual"),
    ("the budget memo", "the staffing memo", "approval date", "the twentieth", "approver", "Ms. Valdis"),
    ("the data audit", "the security audit", "audit lead", "Kwame", "report deadline", "the end of next month"),
    ("the apartment viewing", "the office viewing", "monthly rent", "fourteen hundred dollars", "viewing time", "Saturday at noon"),
    ("the podcast episode", "the webinar episode", "guest", "a glassblower named Ilse", "runtime", "forty minutes"),
    ("the trail race", "the lake swim", "entry fee", "thirty-five dollars", "course length", "twelve kilometers"),
    ("the printer lease", "the copier lease", "lease term", "thirty-six months", "monthly fee", "seventy dollars"),
    ("the museum visit", "the observatory visit", "ticket cost", "nine dollars each", "guide", "a docent called Paloma"),
    ("the chili cook-off", "the pie contest", "winning recipe", "the smoked bean chili", "number of entries", "fourteen"),
    ("the roof inspection", "the boiler inspection", "inspector", "Mr. Brandt", "inspection date", "the third of November"),
    ("the language exchange", "the book swap", "meeting room", "room 12", "organizer", "Teodora"),
    ("the solar quote", "the heat pump quote", "system size", "six kilowatts", "installer", "Brightfield Energy"),
    ("the charity concert", "the school recital", "opening act", "a string quartet", "doors time", "half past five"),
    ("the compost project", "the rain barrel project", "bin capacity", "two hundred liters", "site", "behind the community garden"),
]

FILLER = [
    "Did anyone see the weather forecast for tomorrow?", "I need another coffee before this goes any further.",
    "My train was late again this morning.", "The new chairs in the corner are surprisingly comfortable.",
    "I keep forgetting to water the plants by the window.", "Has anybody tried the sandwich place downstairs?",
    "The wifi was slow again earlier.", "I left my umbrella at home, of course.",
    "That was a long week, honestly.", "We should clear out the shared fridge on Friday.",
    "The elevator is making that noise again.", "I am still catching up on email from yesterday.",
    "Somebody put a bowl of oranges in the kitchen.", "My phone battery is almost dead.",
    "I walked past the park on the way in and it was lovely.", "The projector bulb flickered a bit earlier.",
    "I could use a longer lunch break today.", "Traffic was terrible near the bridge.",
    "Someone borrowed my stapler and never returned it.", "The radio in the lobby is playing the same song again.",
    "I forgot my charger at home.", "It is surprisingly quiet in here today.",
    "I think the heating is set a bit too high.", "Let me know if you want anything from the shop.",
]

NAMES = ["Aiko", "Bram", "Celia", "Dmitri", "Esra", "Farid", "Greta", "Hollis", "Imani", "Jonas", "Kaveh", "Liesel"]
SHAPES = ["topic-no-detail", "other-entity", "after-cut", "promise", "hypothetical", "speculation",
          "never-mentioned", "explicit-unknown", "different-detail"]
SHAPE_NOTE = {
    "topic-no-detail": "the topic is mentioned, the asked detail never given",
    "other-entity": "the asked detail is given, but for another thing",
    "after-cut": "the detail appears only after as_of_seq",
    "promise": "someone promises to look the detail up and report back",
    "hypothetical": "the detail appears only inside an if-we-did-it hypothetical",
    "speculation": "the detail is a guess that nobody confirmed",
    "never-mentioned": "the thing asked about is never mentioned",
    "explicit-unknown": "a speaker says they do not have the detail yet",
    "different-detail": "another detail of the same thing is given, not the asked one",
}


def _up(s: str) -> str:
    return s[0].upper() + s[1:]


def _skeleton(shape: str, t: tuple[str, ...], a: str, b: str) -> tuple[list[str], list[str], int, int]:
    """Lines of the insufficient case and of its twin, the twin's evidence index, and the insufficient case's
    as_of index (-1 for the last turn). The two differ only where the detail is or is not present."""
    thing, other, attr, value, attr2, value2 = t
    detail = f"The {attr} for {thing} is {value}."
    sorted_out = f"{a}: We finally sorted out {thing}."
    talked = f"{a}: We talked about {thing} again today."
    overdue = f"{b}: Good, that was overdue."
    if shape == "topic-no-detail":
        return [sorted_out, overdue, f"{a}: I will write it up later today."], [sorted_out, overdue, f"{a}: {detail}"], 2, -1
    if shape == "other-entity":
        both = f"{a}: We finally sorted out {thing} and {other}."
        return [both, overdue, f"{a}: The {attr} for {other} is {value}."], [both, overdue, f"{a}: {detail}"], 2, -1
    if shape == "after-cut":
        return [sorted_out, overdue, f"{a}: {detail}"], [sorted_out, f"{a}: {detail}", overdue], 1, 1
    if shape == "promise":
        ask = f"{b}: Good, what is the {attr} there?"
        return ([sorted_out, ask, f"{a}: I do not have it in front of me, so I will look it up and message you."],
                [sorted_out, ask, f"{a}: I have it right here: the {attr} is {value}."], 2, -1)
    if shape == "hypothetical":
        return ([talked, f"{b}: If we went with {thing}, the {attr} would be {value}.", f"{a}: Okay."],
                [talked, f"{b}: Since we went with {thing}, the {attr} is {value}.", f"{a}: Okay."], 1, -1)
    if shape == "speculation":
        return ([talked, f"{b}: I am guessing the {attr} is {value}, but nobody has confirmed it.", f"{a}: Okay."],
                [talked, f"{b}: I confirmed it with the team and the {attr} is {value}.", f"{a}: Okay."], 1, -1)
    if shape == "never-mentioned":
        sync, go = f"{a}: Quick sync on this week's open items.", f"{b}: Go ahead."
        return [sync, go, f"{a}: Also, thanks again for helping out today."], [sync, go, f"{a}: Also, the {attr} for {thing} is {value}."], 2, -1
    if shape == "explicit-unknown":
        ask = f"{b}: What is the {attr}?"
        return ([sorted_out, ask, f"{a}: I have not heard from the owner about the {attr} yet."],
                [sorted_out, ask, f"{a}: I just heard from the owner that the {attr} is {value}."], 2, -1)
    if shape == "different-detail":
        return [sorted_out, overdue, f"{a}: The {attr2} for {thing} is {value2}."], [sorted_out, overdue, f"{a}: {detail}"], 2, -1
    raise ValueError(shape)


def _pair(shape: str, k: int, topic: tuple[str, ...], rng: random.Random) -> tuple[Scenario, Scenario]:
    a, b = rng.sample(NAMES, 2)
    ins_lines, ans_lines, key, ins_asof = _skeleton(shape, topic, a, b)
    thing, _other, attr, value, _attr2, _value2 = topic
    if k % LONG_EVERY == LONG_EVERY - 1:
        fill = [f"{(a, b)[i % 2]}: {line}" for i, line in enumerate(rng.sample(FILLER, rng.randint(12, 14)))]
        ins_lines, ans_lines, key = fill + ins_lines, fill + ans_lines, key + len(fill)
        ins_asof = ins_asof + len(fill) if ins_asof >= 0 else ins_asof
    question = rng.choice([f"What is the {attr} for {thing}?", f"Can you tell me the {attr} for {thing}?"])
    out = []
    for suffix, lines in (("ins", ins_lines), ("ans", ans_lines)):
        cid = f"syn-xtra-{shape}-{k:02d}-{suffix}"
        conv = _conversation(cid, lines, "dev")
        insufficient = suffix == "ins"
        asof = ins_asof if insufficient and ins_asof >= 0 else len(lines) - 1
        out.append(Scenario(
            id=cid, conversation=conv, category="insufficient_evidence" if insufficient else "recall",
            request=ButtonRequest(conversation_id=cid, as_of_seq=asof, question=question),
            gold_evidence_turn_ids=[] if insufficient else [conv.turns[key].id],
            gold_label="insufficient" if insufficient else None,
            gold_answer=None if insufficient else f"{_up(value)}.", origin="synthetic"))
    return out[0], out[1]


def build_extra() -> list[Scenario]:
    """All cases in a fixed order: each insufficient case followed by its answerable twin."""
    rng = random.Random(SEED)
    topics = TOPICS[:]
    rng.shuffle(topics)
    out: list[Scenario] = []
    for n, (shape, k) in enumerate((s, k) for s in SHAPES for k in range(PAIRS_PER_SHAPE)):
        out += _pair(shape, k, topics[n % len(topics)], rng)
    return out


def write_extra(root: Path) -> Path:
    return write_local_manifest(SOURCE, {"dev": build_extra()}, root)


if __name__ == "__main__":
    print(write_extra(Path(".")))
