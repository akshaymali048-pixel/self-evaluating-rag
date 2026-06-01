from enum import Enum
from typing import Literal

EvaluationModeName = Literal["user", "evaluation"]


class EvaluationMode(str, Enum):
    USER = "user"
    EVALUATION = "evaluation"


def normalize_ground_truth(ground_truth: str | None) -> str | None:
    if ground_truth is None:
        return None
    stripped = ground_truth.strip()
    return stripped if stripped else None


def resolve_evaluation_mode(
    ground_truth: str | None,
    *,
    enable_reference_metrics: bool,
) -> EvaluationMode:
    if enable_reference_metrics and normalize_ground_truth(ground_truth) is not None:
        return EvaluationMode.EVALUATION
    return EvaluationMode.USER


def should_run_reference_metrics(
    ground_truth: str | None,
    *,
    enable_reference_metrics: bool,
) -> bool:
    return resolve_evaluation_mode(
        ground_truth,
        enable_reference_metrics=enable_reference_metrics,
    ) == EvaluationMode.EVALUATION


def metric_names_for_mode(*, run_reference_metrics: bool) -> set[str]:
    names = {"faithfulness", "answer_relevancy"}
    if run_reference_metrics:
        names.update({"context_precision", "context_recall"})
    return names
