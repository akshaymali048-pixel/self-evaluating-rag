from dataclasses import dataclass


@dataclass(frozen=True)
class RagasScores:
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }
