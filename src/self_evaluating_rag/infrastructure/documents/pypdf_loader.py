from pypdf import PdfReader

from self_evaluating_rag.domain.ports.pdf_loader import LoadedPDF, PDFLoader


class PyPDFLoader(PDFLoader):
    def load(self, file_path: str) -> LoadedPDF:
        reader = PdfReader(file_path)
        pages: list[tuple[int, str]] = []
        full_parts: list[str] = []

        for index, page in enumerate(reader.pages):
            page_number = index + 1
            text = page.extract_text() or ""
            pages.append((page_number, text))
            if text.strip():
                full_parts.append(text)

        full_text = "\n\n".join(full_parts).strip()
        if not full_text:
            raise ValueError("PDF contains no extractable text.")

        return LoadedPDF(full_text=full_text, pages=pages)
