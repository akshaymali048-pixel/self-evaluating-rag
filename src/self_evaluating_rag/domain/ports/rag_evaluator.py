from abc import ABC, abstractmethod

from self_evaluating_rag.domain.value_objects.metric_scores import RagasScores


class RagEvaluator(ABC):
    @abstractmethod
    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RagasScores:
        pass
