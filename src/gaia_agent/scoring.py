import re
import string


def parse_number_with_status(number: str) -> tuple[bool, float]:
    for char in ["$", "%", ","]:
        number = number.replace(char, "")
    try:
        return True, float(number)
    except ValueError:
        return False, float("inf")


def normalize_number(number: str) -> float:
    _, normalized = parse_number_with_status(number)
    return normalized


def _normalize_string(value: str, *, remove_punctuation: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", value)
    if remove_punctuation:
        return no_spaces.lower().translate(str.maketrans("", "", string.punctuation))
    return no_spaces.lower()


def _split(value: str) -> list[str]:
    return re.split("[,;]", value)


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _compare_element(answer: str, ground_truth: str) -> bool:
    if _is_float(ground_truth):
        return normalize_number(answer) == float(ground_truth)
    return _normalize_string(answer, remove_punctuation=False) == _normalize_string(
        ground_truth, remove_punctuation=False
    )


def question_scorer(model_answer: str | None, ground_truth: str) -> bool:
    answer = "None" if model_answer is None else model_answer
    if _is_float(ground_truth):
        return normalize_number(answer) == float(ground_truth)
    if any(char in ground_truth for char in [",", ";"]):
        gold_parts = _split(ground_truth)
        answer_parts = _split(answer)
        if len(gold_parts) != len(answer_parts):
            return False
        return all(_compare_element(a, g) for a, g in zip(answer_parts, gold_parts, strict=True))
    return _normalize_string(answer) == _normalize_string(ground_truth)
