import os

from DocumentIngestion.protocols.document_source import DocumentSource


class FileRepo(DocumentSource):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def get_documents(self) -> list[str]:
        return [
            os.path.join(self.folder_path, doc)
            for doc in os.listdir(self.folder_path)
            if doc.lower().endswith(".pdf")
        ]
