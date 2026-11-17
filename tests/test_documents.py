from pathlib import Path

import pytest

from support_agent.documents import load_document, split_documents


def test_demo_support_faq_loads_and_splits_with_source_metadata():
    sample_path = Path(__file__).parents[1] / "data" / "sample_support_faq.md"

    documents = load_document(sample_path.name, sample_path.read_bytes())
    chunks = split_documents(documents)

    assert documents
    assert chunks
    assert any("30 days" in chunk.page_content for chunk in chunks)
    assert all(chunk.metadata["source"] == sample_path.name for chunk in chunks)


def test_unsupported_file_type_is_rejected():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document("support.exe", b"not a support document")


def test_empty_document_is_rejected():
    with pytest.raises(ValueError, match="is empty"):
        load_document("blank.txt", b"")


def test_displayed_source_name_does_not_include_upload_path():
    documents = load_document(r"C:\fakepath\policy.md", b"Return window: 30 days.")

    assert documents[0].metadata["source"] == "policy.md"

