from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LoadedPDF:
    full_text: str
    pages: list[tuple[int, str]]


class PDFLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> LoadedPDF:
        pass
