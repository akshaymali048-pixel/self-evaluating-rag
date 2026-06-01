class DomainError(Exception):
    """Base domain exception."""


class DocumentNotFoundError(DomainError):
    def __init__(self, document_id: str) -> None:
        super().__init__(f"Document not found: {document_id}")
        self.document_id = document_id


class NoDocumentsError(DomainError):
    """Raised when an operation requires at least one uploaded document."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message or "No documents have been uploaded yet. Upload a PDF first."
        )


class IngestionError(DomainError):
    pass


class RetrievalError(DomainError):
    pass


class EvaluationError(DomainError):
    pass


class GeminiNotConfiguredError(DomainError):
    """Raised when a Gemini-backed operation runs without GEMINI_API_KEY."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "GEMINI_API_KEY is not configured. Set it to use Gemini, or set LLM_PROVIDER=openai."
        )


class OpenAINotConfiguredError(DomainError):
    """Raised when an OpenAI-backed operation runs without OPENAI_API_KEY."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "OPENAI_API_KEY is not configured. Set it when LLM_PROVIDER=openai."
        )


class GeminiQuotaExceededError(DomainError):
    """Raised when Google Gemini returns HTTP 429 / quota exhausted."""

    provider = "Google Gemini"
    status = 429

    def __init__(self, model: str | None = None, message: str | None = None) -> None:
        self.model = model
        super().__init__(
            message
            or (
                f"Gemini quota exceeded for model {model}."
                if model
                else "Gemini quota exceeded."
            )
        )
