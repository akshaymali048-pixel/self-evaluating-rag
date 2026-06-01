from uuid import UUID, uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from self_evaluating_rag.config.settings import Settings
from self_evaluating_rag.domain.entities.document import DocumentChunk
from self_evaluating_rag.domain.ports.chunking_service import ChunkingService


class RecursiveChunker(ChunkingService):
    def __init__(self, settings: Settings) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )

    def chunk_text(
        self,
        text: str,
        document_id: UUID,
        pages: list[tuple[int, str]] | None = None,
    ) -> list[DocumentChunk]:
        if pages:
            chunks: list[DocumentChunk] = []
            global_index = 0
            for page_number, page_text in pages:
                if not page_text.strip():
                    continue
                splits = self._splitter.split_text(page_text)
                for split in splits:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid4()),
                            document_id=document_id,
                            text=split,
                            page=page_number,
                            chunk_index=global_index,
                        )
                    )
                    global_index += 1
            return chunks

        splits = self._splitter.split_text(text)
        return [
            DocumentChunk(
                chunk_id=str(uuid4()),
                document_id=document_id,
                text=split,
                chunk_index=index,
            )
            for index, split in enumerate(splits)
        ]
