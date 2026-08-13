import math

import pytest

from gaia_agent.scoring import parse_number_with_status, question_scorer


@pytest.mark.parametrize(
    ("answer", "gold", "expected"),
    [
        ("$1,234", "1234", True),
        ("12%", "12", True),
        ("Sea Gull", "seagull", True),
        ("A; B", "A;B", True),
        ("1, 2", "1,2", True),
        ("B, A", "A,B", False),
        (None, "None", True),
    ],
)
def test_question_scorer_matches_official_cases(
    answer: str | None, gold: str, expected: bool
) -> None:
    assert question_scorer(answer, gold) is expected


def test_question_scorer_keeps_list_element_punctuation_significant() -> None:
    assert question_scorer("A.B", "AB") is True
    assert question_scorer("A.B", "A,B") is False


@pytest.mark.parametrize("infinite_reference", ["inf", "1e309"])
def test_invalid_numeric_answer_retains_official_positive_infinity_fallback(
    infinite_reference: str,
) -> None:
    assert question_scorer("not a number", infinite_reference) is True


def test_invalid_numeric_answer_does_not_match_negative_infinity() -> None:
    assert question_scorer("not a number", "-inf") is False


def test_success_aware_parser_distinguishes_invalid_input_from_valid_infinity() -> None:
    invalid_success, invalid_value = parse_number_with_status("not a number")
    infinity_success, infinity_value = parse_number_with_status("inf")

    assert invalid_success is False
    assert math.isinf(invalid_value) and invalid_value > 0
    assert infinity_success is True
    assert math.isinf(infinity_value) and infinity_value > 0
