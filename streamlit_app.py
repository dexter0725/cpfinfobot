"""Streamlit UI for the CPF Board Info Verification Bot."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.secrets import StreamlitSecretNotFoundError

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if __package__ in (None, ""):
    sys.path.append(str(PACKAGE_ROOT))
    import ingest  # type: ignore
    from rag import RAGPipeline  # type: ignore
    from utils.security import check_password  # type: ignore
else:
    from . import ingest
    from .rag import RAGPipeline
    from .utils.security import check_password

ENV_PATH = PACKAGE_ROOT / ".env"
ROOT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_TOP_K = 4

load_dotenv(ENV_PATH, override=False)
load_dotenv(ROOT_ENV_PATH, override=True)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .cpf-card {
                background-color: #f8fafc;
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid #e2e8f0;
            }
            .cpf-card textarea {
                border-radius: 12px !important;
                border: 1px solid #d6dbe4 !important;
            }
            .cpf-card button {
                border-radius: 999px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_api_key() -> Optional[str]:
    try:
        secret_value = st.secrets["OPENAI_API_KEY"]  # type: ignore[index]
        if secret_value:
            return secret_value
    except (StreamlitSecretNotFoundError, KeyError):
        pass
    return os.getenv("OPENAI_API_KEY")


def _init_pipeline() -> Optional[RAGPipeline]:
    api_key = _get_api_key()
    if not api_key:
        st.warning("Add your OPENAI_API_KEY to project .env, cpf_bot/.env, or Streamlit secrets to continue.")
        return None
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline(api_key=api_key)
    return st.session_state.pipeline


def page_about() -> None:
    st.header("CPF Board Info Verification Bot")
    st.write(
        """The CPF Board Info Verification Bot empowers members of the public to fact-check
        statements they encounter online about CPF policies. By grounding every answer in
        official CPF FAQs and guides, the bot reduces misinformation and keeps the public
        aligned with authoritative sources."""
    )
    st.subheader("Problem Statement")
    st.markdown(
        """Information about CPF policies is scattered across the internet. Citizens who try to
        verify a claim often have to read lengthy documents or contact customer service.
        This bot offers a single destination to check whether a claim aligns with official
        CPF guidance."""
    )
    st.subheader("Impact")
    st.markdown(
        """By guiding citizens to accurate information quickly, the bot reduces rumours,
        improves financial decision-making, and eases the load on CPF service channels."""
    )


def page_methodology() -> None:
    st.header("Methodology")
    st.markdown(
        """1. **Document ingestion** – Admins upload CPF FAQs, policy notes, or press releases.
        The ingestion service chunks curated CPF FAQs, embeds the text with OpenAI embeddings, and
        stores vectors in a Chroma index.
        2. **Retrieval-Augmented Generation (RAG)** – When a user asks a question, the app
        retrieves the most relevant chunks and feeds them to GPT-4o mini to craft a grounded
        response that cites its sources.
        3. **Verification workflow** – Users can paste claims verbatim. The bot explains if
        the claim is supported, partially supported, or not supported and highlights the
        matching CPF rules.
        4. **Summaries and exports** – Users may request a high-level summary of the evidence
        or download the full response for reference."""
    )
    st.subheader("Why GenAI")
    st.write(
        """LLMs understand natural language questions, reason across multiple policy
        documents, and explain nuanced rules in plain English. Compared with keyword bots,
        this approach scales to new questions without manual rule encoding."""
    )


def _render_user_panel(pipeline: RAGPipeline) -> None:
    st.subheader("Ask CPF Bot")
    st.caption("Verify any CPF-related statement in seconds. Responses are grounded in official CPF FAQs.")
    with st.expander("Important notice", expanded=False):
        st.warning(
            """IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only.
            The information provided here is NOT intended for real-world usage and should not be relied upon for
            financial, legal, or healthcare decisions. The LLM may generate inaccurate information; you assume full
            responsibility for any use of the output. Always consult qualified professionals for accurate, personalized advice."""
        )
    with st.container():
        st.markdown('<div class="cpf-card">', unsafe_allow_html=True)
        question = st.text_area(
            "Paste your CPF question or claim",
            placeholder="E.g. I heard I can withdraw all of my OA savings at 55 if I sell my flat. Is this true?",
            height=150,
            key="user_question",
        )
        ask = st.button("Ask CPF Bot", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if ask:
        if not question.strip():
            st.warning("Enter a question or claim to verify.")
        else:
            try:
                with st.spinner("Generating grounded answer..."):
                    response = pipeline.query(question, top_k=DEFAULT_TOP_K)
            except ValueError:
                st.error(
                    "Knowledge base is empty. Run `python -m cpf_bot.rebuild_vectorstore` in your terminal (after activating the virtualenv) to regenerate embeddings."
                )
                return
            st.markdown("### Answer")
            st.write(response.answer)
            st.markdown("### Sources")
            for citation, doc_text in zip(response.citations, response.source_documents):
                with st.expander(citation):
                    st.write(doc_text)

    with st.expander("Knowledge base files"):
        files = ingest.list_existing_documents()
        if not files:
            st.write("No documents found. Add markdown/PDF files under `cpf_bot/data/` and rerun the rebuild script.")
        else:
            for file in files:
                st.write("•", file.name)


def page_bot() -> None:
    pipeline = _init_pipeline()
    if not pipeline:
        return
    _inject_styles()
    _render_user_panel(pipeline)


def main() -> None:
    st.set_page_config(page_title="CPF Board Info Verification Bot", page_icon="📘", layout="wide")
    if not check_password():
        st.stop()
    st.sidebar.title("CPF Bot Navigation")
    page = st.sidebar.radio("Go to", ["CPF Verification", "About", "Methodology"], index=0)

    if page == "CPF Verification":
        page_bot()
    elif page == "About":
        page_about()
    else:
        page_methodology()


if __name__ == "__main__":
    main()
