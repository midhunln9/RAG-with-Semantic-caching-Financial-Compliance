from DocumentIngestion.models.pdf_document import PdfDocument
from DocumentIngestion.protocol.file_repo import FileRepository


class IngestorService:
    def __init__(self, google_drive_repository: FileRepository):
        self.google_drive_repository = google_drive_repository

    def ingest(self) -> list[PdfDocument]:
        return self.google_drive_repository.start_ingesting_documents()
