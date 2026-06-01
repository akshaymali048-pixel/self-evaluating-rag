from self_evaluating_rag.domain.entities.evaluation import EvaluationResult
from self_evaluating_rag.domain.value_objects.metric_scores import RagasScores
from self_evaluating_rag.infrastructure.persistence.postgres.models.evaluation_orm import (
    EvaluationRunORM,
)


class EvaluationMapper:
    @staticmethod
    def to_domain(orm: EvaluationRunORM) -> EvaluationResult:
        return EvaluationResult(
            id=orm.id,
            question=orm.question,
            answer=orm.answer,
            contexts=orm.contexts,
            scores=RagasScores(
                faithfulness=orm.faithfulness,
                answer_relevancy=orm.answer_relevancy,
                context_precision=orm.context_precision,
                context_recall=orm.context_recall,
            ),
            is_hallucination=orm.is_hallucination,
            latency_ms=orm.latency_ms,
            model_name=orm.model_name,
            embedding_model=orm.embedding_model,
            document_id=orm.document_id,
            ground_truth=orm.ground_truth,
            created_at=orm.created_at,
        )

    @staticmethod
    def to_orm(entity: EvaluationResult) -> EvaluationRunORM:
        return EvaluationRunORM(
            id=entity.id,
            document_id=entity.document_id,
            question=entity.question,
            answer=entity.answer,
            contexts=entity.contexts,
            faithfulness=entity.scores.faithfulness,
            answer_relevancy=entity.scores.answer_relevancy,
            context_precision=entity.scores.context_precision,
            context_recall=entity.scores.context_recall,
            is_hallucination=entity.is_hallucination,
            latency_ms=entity.latency_ms,
            model_name=entity.model_name,
            embedding_model=entity.embedding_model,
            ground_truth=entity.ground_truth,
            created_at=entity.created_at,
        )
