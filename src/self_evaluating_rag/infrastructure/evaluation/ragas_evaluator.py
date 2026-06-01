import asyncio
import logging
import math

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from self_evaluating_rag.application.services.evaluation_mode import should_run_reference_metrics
from self_evaluating_rag.config.settings import Settings
from self_evaluating_rag.domain.exceptions.domain_errors import EvaluationError
from self_evaluating_rag.domain.ports.rag_evaluator import RagEvaluator
from self_evaluating_rag.domain.value_objects.metric_scores import RagasScores
from self_evaluating_rag.infrastructure.llm.provider_factory import (
    create_langchain_chat_model,
    create_langchain_embedding_model,
)

logger = logging.getLogger(__name__)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _select_metrics(*, run_reference_metrics: bool) -> list:
    metrics = [faithfulness, answer_relevancy]
    if run_reference_metrics:
        metrics.extend([context_precision, context_recall])
    return metrics


class RagasEvaluator(RagEvaluator):
    """RAGAS 0.4 evaluator using the configured LLM provider via LangChain wrappers."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm_wrapper: LangchainLLMWrapper | None = None
        self._embeddings_wrapper: LangchainEmbeddingsWrapper | None = None

    def _get_llm_wrapper(self) -> LangchainLLMWrapper:
        if self._llm_wrapper is None:
            chat = create_langchain_chat_model(self._settings, temperature=0.0)
            self._llm_wrapper = LangchainLLMWrapper(chat)
        return self._llm_wrapper

    def _get_embeddings_wrapper(self) -> LangchainEmbeddingsWrapper:
        if self._embeddings_wrapper is None:
            embeddings = create_langchain_embedding_model(self._settings)
            self._embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)
        return self._embeddings_wrapper

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RagasScores:
        run_reference_metrics = should_run_reference_metrics(
            ground_truth,
            enable_reference_metrics=self._settings.enable_reference_metrics,
        )
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._evaluate_sync,
                    question,
                    answer,
                    contexts,
                    ground_truth if run_reference_metrics else None,
                    run_reference_metrics,
                ),
                timeout=self._settings.ragas_timeout_seconds,
            )
        except TimeoutError as exc:
            raise EvaluationError("RAGAS evaluation timed out") from exc
        except Exception as exc:
            if run_reference_metrics:
                logger.warning(
                    "Reference RAGAS metrics failed; returning user-mode scores only: %s",
                    exc,
                )
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        self._evaluate_sync,
                        question,
                        answer,
                        contexts,
                        None,
                        False,
                    ),
                    timeout=self._settings.ragas_timeout_seconds,
                )
            logger.exception("RAGAS evaluation failed")
            raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc

    def _evaluate_sync(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
        run_reference_metrics: bool,
    ) -> RagasScores:
        sample_kwargs: dict = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        }
        if run_reference_metrics and ground_truth:
            sample_kwargs["reference"] = ground_truth

        dataset = EvaluationDataset(samples=[SingleTurnSample(**sample_kwargs)])
        metrics = _select_metrics(run_reference_metrics=run_reference_metrics)

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._get_llm_wrapper(),
            embeddings=self._get_embeddings_wrapper(),
        )

        df = result.to_pandas()
        if df.empty:
            return RagasScores(None, None, None, None)

        record = df.iloc[0]
        context_precision_score = None
        context_recall_score = None
        if run_reference_metrics:
            context_precision_score = _safe_float(record.get("context_precision"))
            context_recall_score = _safe_float(record.get("context_recall"))

        return RagasScores(
            faithfulness=_safe_float(record.get("faithfulness")),
            answer_relevancy=_safe_float(record.get("answer_relevancy")),
            context_precision=context_precision_score,
            context_recall=context_recall_score,
        )
