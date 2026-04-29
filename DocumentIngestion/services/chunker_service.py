from langchain_core.documents import Document

from DocumentIngestion.models.pdf_document import PdfDocument
from DocumentIngestion.strategy.splitter import Splitter


class ChunkerService:
    def __init__(self, splitter_strategy: Splitter):
        self.splitter_strategy = splitter_strategy

    def start_chunking(self, pdf_documents: list[PdfDocument]) -> list[Document]:
        self._validate_dependencies()

        all_chunked_documents: list[Document] = []

        for pdf_document in pdf_documents:
            try:
                page_documents = self._load_pdf_documents(pdf_document)
                chunked_documents = self.splitter_strategy.split_documents(
                    page_documents
                )
                all_chunked_documents.extend(chunked_documents)
            finally:
                pdf_document.cleanup()

        return all_chunked_documents

    def _load_pdf_documents(self, pdf_document: PdfDocument) -> list[Document]:
        from langchain_community.document_loaders import PyPDFLoader

        pdf_loader = PyPDFLoader(str(pdf_document.local_file_path))
        page_documents = pdf_loader.load()

        for page_document in page_documents:
            page_document.metadata["source"] = pdf_document.document_name
            page_document.metadata["file_id"] = pdf_document.file_id
            page_document.metadata["page"] = page_document.metadata.get("page", 0)

        return page_documents

    def _validate_dependencies(self) -> None:
        try:
            from langchain_community.document_loaders import PyPDFLoader  # noqa: F401
            import pypdf  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyPDFLoader is not available in the current uv environment. "
                "Please install `langchain-community` and `pypdf` before running "
                "`uv run python ingestion.py`."
            ) from exc
