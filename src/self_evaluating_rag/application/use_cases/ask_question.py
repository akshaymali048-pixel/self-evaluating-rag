import logging
import time
from datetime import UTC, datetime
from uuid import uuid4

from self_evaluating_rag.application.dto.query_dto import AskQuestionInput, AskQuestionOutput
from self_evaluating_rag.application.services.evaluation_mode import (
    EvaluationMode,
    normalize_ground_truth,
    resolve_evaluation_mode,
)
from self_evaluating_rag.application.services.hallucination_calculator import HallucinationCalculator
from self_evaluating_rag.config.settings import Settings
from self_evaluating_rag.domain.entities.answer import GeneratedAnswer
from self_evaluating_rag.domain.entities.document import Document
from self_evaluating_rag.domain.entities.evaluation import EvaluationResult
from self_evaluating_rag.domain.entities.query import Question
from self_evaluating_rag.domain.exceptions.domain_errors import (
    DocumentNotFoundError,
    NoDocumentsError,
    RetrievalError,
)
from self_evaluating_rag.domain.ports.document_repository import DocumentRepository
from self_evaluating_rag.domain.ports.embedding_service import EmbeddingService
from self_evaluating_rag.domain.ports.evaluation_repository import EvaluationRepository
from self_evaluating_rag.domain.ports.llm_service import LLMService
from self_evaluating_rag.domain.ports.rag_evaluator import RagEvaluator
from self_evaluating_rag.domain.ports.vector_store import VectorStore

logger = logging.getLogger(__name__)


class AskQuestionUseCase:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
        rag_evaluator: RagEvaluator,
        evaluation_repository: EvaluationRepository,
        document_repository: DocumentRepository,
        hallucination_calculator: HallucinationCalculator,
        settings: Settings,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._llm_service = llm_service
        self._rag_evaluator = rag_evaluator
        self._evaluation_repository = evaluation_repository
        self._document_repository = document_repository
        self._hallucination_calculator = hallucination_calculator
        self._settings = settings

    async def execute(self, request: AskQuestionInput) -> AskQuestionOutput:
        start = time.perf_counter()
        document = await self._resolve_document(request)
        document_id = document.id if document else None
        resolved_filename = document.filename if document else request.filename

        top_k = request.top_k or self._settings.retrieval_top_k
        normalized_ground_truth = normalize_ground_truth(request.ground_truth)
        evaluation_mode = resolve_evaluation_mode(
            normalized_ground_truth,
            enable_reference_metrics=self._settings.enable_reference_metrics,
        )

        question = Question(
            text=request.question,
            document_id=document_id,
            top_k=top_k,
            ground_truth=normalized_ground_truth,
        )

        query_embedding = await self._embedding_service.embed_query(question.text)
        contexts = await self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=question.document_id,
        )
        if not contexts:
            raise RetrievalError("No relevant context found for the question.")

        context_block = "\n\n".join(ctx.text for ctx in contexts)
        answer_text = await self._llm_service.generate_answer(question.text, context_block)

        generated = GeneratedAnswer(
            text=answer_text,
            contexts=contexts,
            document_id=question.document_id,
            model_name=self._llm_service.model_name,
        )

        context_strings = [ctx.text for ctx in generated.contexts]
        scores = await self._rag_evaluator.evaluate(
            question=question.text,
            answer=generated.text,
            contexts=context_strings,
            ground_truth=normalized_ground_truth,
        )

        latency_ms = int((time.perf_counter() - start) * 1000)
        is_hallucination = self._hallucination_calculator.is_hallucination(scores.faithfulness)

        stored_ground_truth = (
            normalized_ground_truth if evaluation_mode == EvaluationMode.EVALUATION else None
        )
        evaluation = EvaluationResult(
            id=uuid4(),
            question=question.text,
            answer=generated.text,
            contexts=context_strings,
            scores=scores,
            is_hallucination=is_hallucination,
            latency_ms=latency_ms,
            model_name=self._llm_service.model_name,
            embedding_model=self._embedding_service.model_name,
            document_id=question.document_id,
            ground_truth=stored_ground_truth,
            created_at=datetime.now(UTC),
        )
        saved = await self._evaluation_repository.save(evaluation)

        logger.info(
            "Evaluated query %s | mode=%s | faithfulness=%s | latency=%dms",
            saved.id,
            evaluation_mode.value,
            scores.faithfulness,
            latency_ms,
        )

        return AskQuestionOutput(
            evaluation_id=saved.id,
            answer=generated.text,
            contexts=context_strings,
            scores=scores,
            is_hallucination=is_hallucination,
            latency_ms=latency_ms,
            model_name=self._llm_service.model_name,
            evaluation_mode=evaluation_mode,
            document_id=question.document_id,
            filename=resolved_filename,
        )

    async def _resolve_document(self, request: AskQuestionInput) -> Document | None:
        if request.use_latest_document:
            document = await self._document_repository.get_latest()
            if document is None:
                raise NoDocumentsError()
            return document

        if request.filename:
            document = await self._document_repository.get_by_filename(request.filename)
            if document is None:
                raise DocumentNotFoundError(request.filename)
            return document

        if request.document_id is not None:
            document = await self._document_repository.get_by_id(request.document_id)
            if document is None:
                raise DocumentNotFoundError(str(request.document_id))
            return document

        return None
