class HallucinationCalculator:
    def __init__(self, faithfulness_threshold: float) -> None:
        self._threshold = faithfulness_threshold

    def is_hallucination(self, faithfulness: float | None) -> bool:
        if faithfulness is None:
            return True
        return faithfulness < self._threshold
