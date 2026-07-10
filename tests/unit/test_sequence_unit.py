"""Unit tests for sequence helpers."""

from tests.helpers import make_sequence


def test_prompt_and_completion_slices():
    seq = make_sequence([1, 2, 3], max_tokens=5)
    seq.append_token(9)
    seq.append_token(8)
    assert seq.prompt_token_ids == [1, 2, 3]
    assert seq.completion_token_ids == [9, 8]
