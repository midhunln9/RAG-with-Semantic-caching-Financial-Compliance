from dataclasses import dataclass
from pathlib import Path


@dataclass
class PdfDocument:
    file_id: str
    document_name: str
    local_file_path: Path
    web_view_link: str

    def cleanup(self) -> None:
        self.local_file_path.unlink(missing_ok=True)
