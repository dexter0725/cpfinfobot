"""RAG utilities for the CPF Info Verification Bot."""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from langchain_openai import ChatOpenAI
try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - compatibility for langchain < 0.2
    from langchain.schema import AIMessage, HumanMessage, SystemMessage

PACKAGE_ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.append(str(PACKAGE_ROOT))
    import ingest  # type: ignore
else:
    from . import ingest

SYSTEM_PROMPT = """You are the CPF Board Info Verification Bot. You answer questions about Singapore's CPF policies.
Follow the rules:
- Base every response strictly on the provided context chunks extracted from official CPF publications.
- If the context does not contain the answer, say that you cannot confirm and suggest consulting official CPF sources.
- Highlight whether a claim seems supported, partially supported, or not supported by the context.
- Cite the source filenames in parentheses using their metadata.
"""


@dataclass
class RAGResponse:
    answer: str
    citations: List[str]
    source_documents: List[str]


class RAGPipeline:
    def __init__(self, api_key: str, temperature: float = 0.1) -> None:
        self.api_key = api_key
        ingest.ensure_directories()
        store = ingest.load_vector_store(api_key)
        if store is None:
            store = ingest.build_vector_store(api_key)
        self.vector_store = store
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            temperature=temperature,
            model="gpt-4o-mini",
        )

    def refresh_store(self) -> None:
        self.vector_store = ingest.build_vector_store(self.api_key)

    def _get_documents(self, question: str, top_k: int):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(question)
        # langchain-core retrievers expose .invoke(prompt)
        return retriever.invoke(question)

    def query(self, question: str, top_k: int = 4) -> RAGResponse:
        documents = self._get_documents(question, top_k)
        if not documents:
            raise ValueError("No documents available in the vector store.")

        context = "\n\n".join(
            f"Source: {doc.metadata.get('source')}\n{doc.page_content}" for doc in documents
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Context:\n" + context + "\n\n"
                    "Question: " + question + "\n"
                    "Answer in a factual tone and cite sources in parentheses."
                )
            ),
        ]
        ai_message: AIMessage = self.llm.invoke(messages)  # type: ignore[assignment]
        citations = [doc.metadata.get("source", "") for doc in documents]
        return RAGResponse(
            answer=ai_message.content,
            citations=citations,
            source_documents=[doc.page_content for doc in documents],
        )

    def summarize_sources(self, question: str, top_k: int = 4) -> str:
        documents = self._get_documents(question, top_k)
        if not documents:
            return "No documents available to summarize."
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source')}\n{doc.page_content}" for doc in documents
        )
        summary_prompt = (
            "Summarize the key CPF facts from the context below in bullet points so a member of the public can understand them.\n\n"
            f"Context:\n{context}"
        )
        response: AIMessage = self.llm.invoke(
            [SystemMessage(content="You compress CPF policy context into clear summaries."), HumanMessage(content=summary_prompt)]
        )  # type: ignore[assignment]
        return response.content


__all__ = ["RAGPipeline", "RAGResponse"]
