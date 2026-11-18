from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

from support_agent.documents import load_document, split_documents
from support_agent.graph import build_support_graph

load_dotenv()
st.set_page_config(page_title="Customer Support Agent", page_icon="💬", layout="centered")

APP_DIR = Path(__file__).resolve().parent
BUSINESS_NAME = os.getenv("BUSINESS_NAME", "Demo Store")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:4b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.25"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
SUPPORTED_TYPES = ["pdf", "docx", "md", "txt"]


@st.cache_resource(show_spinner=False)
def get_local_models(base_url: str, chat_name: str, embedding_name: str):
    return (
        ChatOllama(model=chat_name, base_url=base_url, temperature=0, num_ctx=8192),
        OllamaEmbeddings(model=embedding_name, base_url=base_url),
    )


def index_documents(documents: list[Document], embeddings) -> InMemoryVectorStore:
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No readable text was found in those files.")
    return InMemoryVectorStore.from_documents(chunks, embeddings)


def uploaded_documents(uploaded_files) -> list[Document]:
    total_bytes = sum(file.size for file in uploaded_files)
    if total_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"The combined upload is larger than {MAX_UPLOAD_MB} MB.")
    documents: list[Document] = []
    for file in uploaded_files:
        if file.size > MAX_UPLOAD_MB * 1024 * 1024:
            raise ValueError(f"{file.name} is larger than {MAX_UPLOAD_MB} MB.")
        documents.extend(load_document(file.name, file.getvalue()))
    return documents


def render_message(message: dict):
    with st.chat_message(message["role"]):
        st.text(message["content"])
        if message.get("sources"):
            labels = []
            for source in message["sources"]:
                label = f"[{source['id']}] {source['source']}"
                if source.get("page") is not None:
                    label += f", page {source['page']}"
                labels.append(label)
            st.caption("Sources reviewed: " + "; ".join(labels))
        if message.get("handoff_required"):
            st.warning("Human support review is needed. No external ticket or email was sent.")


st.title(f"{BUSINESS_NAME} Support")
st.caption("A document-grounded support assistant. It answers from your uploaded files or asks for a person.")
st.info("Local pilot: uploaded files stay in this app session's memory. The assistant has no tools for refunds, orders, or account changes.")

with st.sidebar:
    st.header("Company knowledge")
    st.write("Upload the support policies and FAQs the assistant is allowed to use.")
    files = st.file_uploader(
        "Choose company documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help=f"PDF, DOCX, Markdown, or text. Combined limit: {MAX_UPLOAD_MB} MB.",
    )

    try:
        chat_model, embeddings = get_local_models(OLLAMA_BASE_URL, CHAT_MODEL, EMBEDDING_MODEL)
    except Exception as exc:
        chat_model, embeddings = None, None
        st.error(f"Could not prepare the local models: {exc}")

    if st.button("Index uploaded documents", disabled=not files or not embeddings, use_container_width=True):
        try:
            with st.spinner("Reading documents and creating local search vectors..."):
                docs = uploaded_documents(files)
                st.session_state.vector_store = index_documents(docs, embeddings)
                st.session_state.graph = build_support_graph(
                    st.session_state.vector_store,
                    chat_model,
                    business_name=BUSINESS_NAME,
                    score_threshold=SCORE_THRESHOLD,
                )
                st.session_state.indexed_files = [file.name for file in files]
                st.session_state.chat_history = []
            st.success(f"Indexed {len(docs)} document sections.")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    if st.button("Load sample support FAQ", use_container_width=True, disabled=not embeddings):
        try:
            sample_path = APP_DIR / "data" / "sample_support_faq.md"
            sample_docs = load_document(sample_path.name, sample_path.read_bytes())
            with st.spinner("Indexing the sample FAQ..."):
                st.session_state.vector_store = index_documents(sample_docs, embeddings)
                st.session_state.graph = build_support_graph(
                    st.session_state.vector_store,
                    chat_model,
                    business_name=BUSINESS_NAME,
                    score_threshold=SCORE_THRESHOLD,
                )
                st.session_state.indexed_files = [sample_path.name]
                st.session_state.chat_history = []
            st.success("Sample FAQ is ready. Try asking about returns or shipping.")
        except Exception as exc:
            st.error(f"Could not load the sample FAQ: {exc}")

    st.divider()
    st.caption(f"Chat model: {CHAT_MODEL}")
    st.caption(f"Embedding model: {EMBEDDING_MODEL}")
    if st.session_state.get("indexed_files"):
        st.caption("Loaded: " + ", ".join(st.session_state.indexed_files))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for item in st.session_state.chat_history:
    render_message(item)

if not st.session_state.get("graph"):
    st.write("Start with **Load sample support FAQ**, or upload your own documents and choose **Index uploaded documents**.")

question = st.chat_input("Ask a question about the uploaded company documents", disabled=not st.session_state.get("graph"))
if question:
    history = list(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the company documents..."):
            try:
                result = st.session_state.graph.invoke({"question": question, "history": history})
                response = {
                    "role": "assistant",
                    "content": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "handoff_required": result.get("handoff_required", False),
                }
            except Exception as exc:
                response = {
                    "role": "assistant",
                    "content": f"The local model could not answer this question: {exc}",
                    "sources": [],
                    "handoff_required": True,
                }
        st.text(response["content"])
        if response["sources"]:
            labels = []
            for source in response["sources"]:
                label = f"[{source['id']}] {source['source']}"
                if source.get("page") is not None:
                    label += f", page {source['page']}"
                labels.append(label)
            st.caption("Sources reviewed: " + "; ".join(labels))
        if response["handoff_required"]:
            st.warning("Human support review is needed. No external ticket or email was sent.")
        st.session_state.chat_history.append(response)

handoff_messages = [
    item for item in st.session_state.chat_history
    if item.get("role") == "assistant" and item.get("handoff_required")
]
if handoff_messages:
    latest = handoff_messages[-1]
    transcript = "\n".join(
        f"{item['role'].title()}: {item['content']}"
        for item in st.session_state.chat_history[-8:]
    )
    st.download_button(
        "Download human-support handoff note",
        data=f"Customer support handoff\n\n{transcript}\n",
        file_name="support-handoff.txt",
        mime="text/plain",
    )

