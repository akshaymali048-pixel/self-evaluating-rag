import pytest

from self_evaluating_rag.application.services.evaluation_mode import (
    EvaluationMode,
    normalize_ground_truth,
    resolve_evaluation_mode,
    should_run_reference_metrics,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("reference answer", "reference answer"),
        ("  trimmed  ", "trimmed"),
    ],
)
def test_normalize_ground_truth(value, expected):
    assert normalize_ground_truth(value) == expected


@pytest.mark.parametrize(
    ("ground_truth", "enable_reference_metrics", "expected"),
    [
        (None, True, EvaluationMode.USER),
        ("", True, EvaluationMode.USER),
        ("answer", True, EvaluationMode.EVALUATION),
        ("answer", False, EvaluationMode.USER),
    ],
)
def test_resolve_evaluation_mode(ground_truth, enable_reference_metrics, expected):
    assert (
        resolve_evaluation_mode(
            ground_truth,
            enable_reference_metrics=enable_reference_metrics,
        )
        == expected
    )


def test_should_run_reference_metrics():
    assert should_run_reference_metrics("answer", enable_reference_metrics=True) is True
    assert should_run_reference_metrics("answer", enable_reference_metrics=False) is False
    assert should_run_reference_metrics(None, enable_reference_metrics=True) is False
