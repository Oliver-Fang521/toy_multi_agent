import pytest

from gaia_agent.scoring import question_scorer


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
