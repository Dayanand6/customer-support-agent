from __future__ import annotations

from io import BytesIO
from pathlib import Path

from docx import Document as WordDocument
from langchain_core.documents import Document
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
MAX_DOCUMENT_CHARS = 250_000


def _safe_filename(name: str) -> str:
    return Path(name.replace("\\", "/")).name or "uploaded-document"


def load_document(name: str, content: bytes) -> list[Document]:
    """Read one supported file into text documents without writing it to disk."""
    safe_name = _safe_filename(name)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type for {safe_name}. Use PDF, DOCX, MD, or TXT.")
    if not content:
        raise ValueError(f"{safe_name} is empty.")

    documents: list[Document] = []
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        if len(reader.pages) > 300:
            raise ValueError(f"{safe_name} is over the 300-page limit.")
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text[:MAX_DOCUMENT_CHARS],
                        metadata={"source": safe_name, "page": page_number},
                    )
                )
    elif suffix == ".docx":
        word_doc = WordDocument(BytesIO(content))
        parts = [paragraph.text for paragraph in word_doc.paragraphs if paragraph.text.strip()]
        for table in word_doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        text = "\n".join(parts).strip()
        if text:
            documents.append(
                Document(
                    page_content=text[:MAX_DOCUMENT_CHARS],
                    metadata={"source": safe_name},
                )
            )
    else:
        text = content.decode("utf-8-sig", errors="replace").strip()
        if text:
            documents.append(
                Document(
                    page_content=text[:MAX_DOCUMENT_CHARS],
                    metadata={"source": safe_name},
                )
            )

    if not documents:
        raise ValueError(f"No readable text was found in {safe_name}.")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    return splitter.split_documents(documents)

