from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from self_evaluating_rag.application.services.hallucination_calculator import HallucinationCalculator
from self_evaluating_rag.application.use_cases.ask_question import AskQuestionUseCase
from self_evaluating_rag.application.use_cases.get_dashboard_metrics import GetDashboardMetricsUseCase
from self_evaluating_rag.application.use_cases.ingest_document import IngestDocumentUseCase
from self_evaluating_rag.application.use_cases.list_documents import (
    GetDocumentUseCase,
    ListDocumentsUseCase,
)
from self_evaluating_rag.config.settings import Settings, get_settings
from self_evaluating_rag.domain.ports.embedding_service import EmbeddingService
from self_evaluating_rag.domain.ports.llm_service import LLMService
from self_evaluating_rag.infrastructure.documents.pypdf_loader import PyPDFLoader
from self_evaluating_rag.infrastructure.documents.recursive_chunker import RecursiveChunker
from self_evaluating_rag.infrastructure.evaluation.ragas_evaluator import RagasEvaluator
from self_evaluating_rag.infrastructure.llm.provider_factory import (
    create_embedding_service,
    create_llm_service,
)
from self_evaluating_rag.infrastructure.persistence.chroma.chroma_client_factory import (
    ChromaClientFactory,
)
from self_evaluating_rag.infrastructure.persistence.chroma.chroma_vector_store import (
    ChromaVectorStore,
)
from self_evaluating_rag.infrastructure.persistence.postgres.repositories.document_repository_impl import (
    DocumentRepositoryImpl,
)
from self_evaluating_rag.infrastructure.persistence.postgres.repositories.evaluation_repository_impl import (
    EvaluationRepositoryImpl,
)


@dataclass
class InfrastructureServices:
    pdf_loader: PyPDFLoader
    chunking_service: RecursiveChunker
    embedding_service: EmbeddingService
    vector_store: ChromaVectorStore
    llm_service: LLMService
    rag_evaluator: RagasEvaluator
    hallucination_calculator: HallucinationCalculator


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._chroma_factory = ChromaClientFactory(self.settings)
        self._services = InfrastructureServices(
            pdf_loader=PyPDFLoader(),
            chunking_service=RecursiveChunker(self.settings),
            embedding_service=create_embedding_service(self.settings),
            vector_store=ChromaVectorStore(self._chroma_factory),
            llm_service=create_llm_service(self.settings),
            rag_evaluator=RagasEvaluator(self.settings),
            hallucination_calculator=HallucinationCalculator(
                self.settings.hallucination_faithfulness_threshold
            ),
        )

    def ingest_document_use_case(self, session: AsyncSession) -> IngestDocumentUseCase:
        document_repo = DocumentRepositoryImpl(session)
        return IngestDocumentUseCase(
            pdf_loader=self._services.pdf_loader,
            chunking_service=self._services.chunking_service,
            embedding_service=self._services.embedding_service,
            vector_store=self._services.vector_store,
            document_repository=document_repo,
            settings=self.settings,
        )

    def ask_question_use_case(self, session: AsyncSession) -> AskQuestionUseCase:
        document_repo = DocumentRepositoryImpl(session)
        evaluation_repo = EvaluationRepositoryImpl(session)
        return AskQuestionUseCase(
            embedding_service=self._services.embedding_service,
            vector_store=self._services.vector_store,
            llm_service=self._services.llm_service,
            rag_evaluator=self._services.rag_evaluator,
            evaluation_repository=evaluation_repo,
            document_repository=document_repo,
            hallucination_calculator=self._services.hallucination_calculator,
            settings=self.settings,
        )

    def get_dashboard_metrics_use_case(self, session: AsyncSession) -> GetDashboardMetricsUseCase:
        evaluation_repo = EvaluationRepositoryImpl(session)
        return GetDashboardMetricsUseCase(evaluation_repo)

    def list_documents_use_case(self, session: AsyncSession) -> ListDocumentsUseCase:
        document_repo = DocumentRepositoryImpl(session)
        return ListDocumentsUseCase(document_repo)

    def get_document_use_case(self, session: AsyncSession) -> GetDocumentUseCase:
        document_repo = DocumentRepositoryImpl(session)
        return GetDocumentUseCase(document_repo)
