"""Utilities for loading CPF documents into a Chroma vector store."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import shutil
from langchain_community.vectorstores import Chroma
try:
    from langchain_core.documents import Document  # langchain 0.2+
except ImportError:  # pragma: no cover - fallback for older versions
    from langchain.docstore.document import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - fallback for langchain<0.2
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_DIR = PROJECT_ROOT / "vector_db" / "cpf_bot_index"
COLLECTION_NAME = "cpf_bot_docs"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _load_file(path: Path) -> Document:
    text = ""
    if path.suffix.lower() in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
    elif path.suffix.lower() == ".pdf":
        text = _read_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    metadata = {
        "source": path.name,
        "relative_path": str(path.relative_to(DATA_DIR)),
    }
    return Document(page_content=text, metadata=metadata)


def load_documents() -> List[Document]:
    docs: List[Document] = []
    for directory in (SAMPLE_DOCS_DIR, UPLOADS_DIR):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                docs.append(_load_file(path))
    return docs


def _split_documents(documents: Iterable[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n- ", "\n", ". "],
    )
    return splitter.split_documents(list(documents))


def _embedder(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")


def build_vector_store(api_key: str) -> Chroma:
    documents = load_documents()
    if not documents:
        raise ValueError("No documents found to index. Add files to data directories first.")

    chunks = _split_documents(documents)
    embeddings = _embedder(api_key)
    if VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DIR),
    )
    store.persist()
    return store


def load_vector_store(api_key: str) -> Chroma | None:
    index_path = VECTOR_DIR / "chroma.sqlite3"
    if not index_path.exists():
        return None

    embeddings = _embedder(api_key)
    return Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DIR),
    )


def list_existing_documents() -> List[Path]:
    files = []
    for directory in (SAMPLE_DOCS_DIR, UPLOADS_DIR):
        if directory.exists():
            files.extend(sorted(p for p in directory.iterdir() if p.is_file()))
    return files


def ensure_directories() -> None:
    for directory in (SAMPLE_DOCS_DIR, UPLOADS_DIR, VECTOR_DIR):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "build_vector_store",
    "load_vector_store",
    "load_documents",
    "list_existing_documents",
    "ensure_directories",
    "DATA_DIR",
    "UPLOADS_DIR",
    "VECTOR_DIR",
]
