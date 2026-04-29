from langchain_core.documents import Document

from DocumentIngestion.strategy.splitter import Splitter


class RecursiveCharacterTextSplitter(Splitter):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap should be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunked_documents: list[Document] = []

        for document in documents:
            text_chunks = self._split_text(document.page_content)
            for chunk_index, text_chunk in enumerate(text_chunks):
                if not text_chunk.strip():
                    continue

                chunk_metadata = dict(document.metadata)
                chunk_metadata["chunk_index"] = chunk_index
                chunked_documents.append(
                    Document(page_content=text_chunk.strip(), metadata=chunk_metadata)
                )

        return chunked_documents

    def _split_text(self, text: str) -> list[str]:
        return self._split_with_separators(text, self.separators)

    def _split_with_separators(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._split_by_window(text)

        current_separator = separators[0]
        remaining_separators = separators[1:]

        if current_separator and current_separator not in text:
            return self._split_with_separators(text, remaining_separators)

        if current_separator == "":
            return self._split_by_window(text)

        text_parts = text.split(current_separator)
        merged_parts: list[str] = []
        current_chunk = ""

        for text_part in text_parts:
            candidate_chunk = text_part
            if current_chunk:
                candidate_chunk = f"{current_chunk}{current_separator}{text_part}"

            if len(candidate_chunk) <= self.chunk_size:
                current_chunk = candidate_chunk
                continue

            if current_chunk:
                merged_parts.append(current_chunk)

            if len(text_part) > self.chunk_size:
                merged_parts.extend(
                    self._split_with_separators(text_part, remaining_separators)
                )
                current_chunk = ""
                continue

            overlap_text = current_chunk[-self.chunk_overlap :] if current_chunk else ""
            current_chunk = f"{overlap_text}{text_part}" if overlap_text else text_part

        if current_chunk:
            merged_parts.append(current_chunk)

        return merged_parts

    def _split_by_window(self, text: str) -> list[str]:
        chunks: list[str] = []
        step_size = self.chunk_size - self.chunk_overlap

        for start_index in range(0, len(text), step_size):
            end_index = start_index + self.chunk_size
            chunks.append(text[start_index:end_index])

        return chunks
