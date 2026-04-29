from typing import Protocol

from DocumentIngestion.models.pdf_document import PdfDocument


class FileRepository(Protocol):
    """
    Contract for repositories that know how to fetch PDF documents from
    an external storage location.
    """

    def start_ingesting_documents(self) -> list[PdfDocument]:
        """
        Fetch the PDF documents and return them as Python objects.
        """
        ...
