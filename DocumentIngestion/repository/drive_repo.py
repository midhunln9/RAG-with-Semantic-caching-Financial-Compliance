import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote, urlparse

import requests

from DocumentIngestion.models.pdf_document import PdfDocument
from DocumentIngestion.protocol.file_repo import FileRepository


class GoogleDriveRepository(FileRepository):
    def __init__(self, drive_public_link: str):
        self.drive_public_link = drive_public_link
        self.folder_id = self._extract_folder_id(drive_public_link)
        self.session = requests.Session()

    def start_ingesting_documents(self) -> list[PdfDocument]:
        pdf_documents: list[PdfDocument] = []

        folder_page = self.session.get(self._build_folder_url(), timeout=30)
        folder_page.raise_for_status()

        for file_id, document_name in self._extract_pdf_references(folder_page.text):
            local_file_path = self._download_pdf_document(file_id, document_name)
            pdf_documents.append(
                PdfDocument(
                    file_id=file_id,
                    document_name=document_name,
                    local_file_path=local_file_path,
                    web_view_link=f"https://drive.google.com/file/d/{file_id}/view",
                )
            )

        return pdf_documents

    def _build_folder_url(self) -> str:
        return (
            "https://drive.google.com/embeddedfolderview"
            f"?id={quote(self.folder_id)}#list"
        )

    def _extract_folder_id(self, drive_public_link: str) -> str:
        folder_match = re.search(r"/folders/([a-zA-Z0-9_-]+)", drive_public_link)
        if folder_match:
            return folder_match.group(1)

        parsed_drive_link = urlparse(drive_public_link)
        folder_query_match = re.search(r"id=([a-zA-Z0-9_-]+)", parsed_drive_link.query)
        if folder_query_match:
            return folder_query_match.group(1)

        raise ValueError(
            "Google Drive folder link should contain a valid folder id."
        )

    def _extract_pdf_references(self, folder_html: str) -> list[tuple[str, str]]:
        anchor_matches = re.findall(
            r'href="([^"]+)"[^>]*>([^<]+)</a>',
            folder_html,
            flags=re.IGNORECASE,
        )

        pdf_references: list[tuple[str, str]] = []
        seen_file_ids: set[str] = set()

        for href, anchor_text in anchor_matches:
            document_name = self._clean_document_name(anchor_text)
            if not document_name.lower().endswith(".pdf"):
                continue

            file_id_match = re.search(
                r"/file/d/([a-zA-Z0-9_-]+)|[?&]id=([a-zA-Z0-9_-]+)",
                href,
            )
            if not file_id_match:
                continue

            file_id = file_id_match.group(1) or file_id_match.group(2)
            if file_id in seen_file_ids:
                continue

            seen_file_ids.add(file_id)
            pdf_references.append((file_id, document_name))

        return pdf_references

    def _clean_document_name(self, anchor_text: str) -> str:
        return (
            anchor_text.replace("&amp;", "&")
            .replace("&#39;", "'")
            .replace("&quot;", '"')
            .strip()
        )

    def _download_pdf_document(self, file_id: str, document_name: str) -> Path:
        response = self.session.get(
            "https://drive.google.com/uc",
            params={"export": "download", "id": file_id},
            timeout=60,
            stream=True,
            allow_redirects=True,
        )
        response.raise_for_status()

        if not self._looks_like_pdf_response(response):
            confirm_token = self._extract_confirm_token(response)
            if not confirm_token:
                raise RuntimeError(
                    f"Unable to download `{document_name}` from Google Drive."
                )

            response = self.session.get(
                "https://drive.google.com/uc",
                params={
                    "export": "download",
                    "id": file_id,
                    "confirm": confirm_token,
                },
                timeout=60,
                stream=True,
                allow_redirects=True,
            )
            response.raise_for_status()

        if not self._looks_like_pdf_response(response):
            raise RuntimeError(
                f"Google Drive returned a non-PDF response for `{document_name}`."
            )

        temporary_file = NamedTemporaryFile(delete=False, suffix=".pdf")
        with temporary_file as file_pointer:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_pointer.write(chunk)

        return Path(temporary_file.name)

    def _looks_like_pdf_response(self, response: requests.Response) -> bool:
        content_type = response.headers.get("Content-Type", "").lower()
        content_disposition = response.headers.get("Content-Disposition", "").lower()

        return "application/pdf" in content_type or ".pdf" in content_disposition

    def _extract_confirm_token(self, response: requests.Response) -> str | None:
        for cookie_name, cookie_value in response.cookies.items():
            if cookie_name.startswith("download_warning"):
                return cookie_value

        confirm_match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
        if confirm_match:
            return confirm_match.group(1)

        return None
