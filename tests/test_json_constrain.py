"""Schema-constrained decoding for the local model: the prefix checker, and the logits processor on a fake tokenizer."""
import json

import pytest

from contextlens.llm.json_constrain import COMPLETE, INCOMPLETE, INVALID, prefix_status
from contextlens.schemas import Card

S = Card.model_json_schema()
GOOD = ('{"mode":"recall","status":"answered","findings":[{"agent":"baseline","kind":"recall","statement":"x \\"q\\" é",'
        '"evidence":[{"turn_id":"t0","quote":"q","speaker":"Ana","start_ms":0}],"confidence":0.5}],"suggestion":null,"missing":["a"]}')


def test_a_valid_card_is_complete_and_every_prefix_of_it_is_incomplete() -> None:
    assert prefix_status(S, GOOD) == COMPLETE
    assert Card.model_validate_json(GOOD)
    for i in range(1, len(GOOD)):
        assert prefix_status(S, GOOD[:i]) == INCOMPLETE, GOOD[:i]
    assert prefix_status(S, GOOD + "  \n") == COMPLETE


@pytest.mark.parametrize("bad", [
    '{"mode":"answered"', '{"mode":"recall","status":"missing"', '{"mode":"recall","status":"abstained","findings":[],"missing":"x"',
    '{"mode":"recall","bogus"', '{"mode":"recall","status":"abstained","findings":[{"agent":1', "[", '{"mode":"recall","mode"',
    GOOD + "x",
])
def test_text_that_cannot_become_a_valid_card_is_invalid(bad: str) -> None:
    assert prefix_status(S, bad) == INVALID


def test_a_closed_object_without_its_required_fields_is_invalid() -> None:
    assert prefix_status(S, '{"mode":"recall"}') == INVALID
    assert prefix_status(S, '{"mode":"recall","status":"abstained","findings":[]}') == COMPLETE


def test_the_processor_only_lets_through_tokens_that_keep_the_card_valid() -> None:
    torch = pytest.importorskip("torch")
    from contextlens.llm.json_constrain import SchemaLogitsProcessor

    pieces = ['{"mode":"', 'answered', 'recall', '","status":"', 'abstained', '","findings":[]}', '<eos>', 'junk']

    class Tok:
        eos_token_id = 6
        def decode(self, ids, **kw): return "".join(pieces[i] for i in ids if i != 6)
        def __len__(self): return len(pieces)

    proc = SchemaLogitsProcessor(Tok(), S, prompt_len=1, top_n=3)
    ids = [99]
    scores = proc(torch.tensor([ids]), torch.tensor([[9.0, 1, 1, 1, 1, 1, 1, 1]]))
    assert int(scores.argmax()) == 0  # the opener is the only valid first piece
    ids.append(0)
    scores = proc(torch.tensor([ids]), torch.tensor([[0.0, 9.0, 8.0, 0, 0, 0, 0, 7.0]]))
    assert int(scores.argmax()) == 2  # "answered" is not a mode; "recall" is
    ids.append(2)
    ids += [3, 4, 5]
    scores = proc(torch.tensor([ids]), torch.tensor([[0.0, 1, 1, 1, 1, 1, 5.0, 9.0]]))
    assert int(scores.argmax()) == 6  # complete: only the end token remains
