from typing import Protocol

class DocumentSource(Protocol):
    """Protocol for document location."""
    def get_documents(self) -> list[str]:
        """Return a list of document file paths."""
        ...