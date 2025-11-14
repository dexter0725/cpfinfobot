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
    from utils.security import (
        admin_password_configured,
        check_password,
        verify_admin_password,
    )  # type: ignore
else:
    from . import ingest
    from .rag import RAGPipeline
    from .utils.security import admin_password_configured, check_password, verify_admin_password

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
            .cpf-answer {
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 1rem;
                line-height: 1.6;
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


def _save_uploaded_file(uploaded_file) -> Optional[Path]:
    if uploaded_file is None:
        return None
    ingest.ensure_directories()
    save_path = ingest.UPLOADS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    return save_path


def _render_document_table() -> None:
    files = ingest.list_existing_documents()
    if not files:
        st.info("No documents are indexed yet. Upload a CPF reference to get started.")
        return
    data = []
    for file in files:
        stats = file.stat()
        data.append(
            {
                "File": file.name,
                "Folder": file.parent.name,
                "Size (KB)": round(stats.st_size / 1024, 2),
            }
        )
    st.dataframe(data, use_container_width=True, hide_index=True)


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
        col1, col2 = st.columns([2, 1])
        with col1:
            top_k = st.slider("Evidence chunks", min_value=2, max_value=8, value=DEFAULT_TOP_K)
        with col2:
            ask = st.button("Ask CPF Bot", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    summarize = st.checkbox("Add evidence summary", value=False)
    export_label = st.text_input("Optional filename for export", value="cpf_bot_response.txt")

    if ask:
        if not question.strip():
            st.warning("Enter a question or claim to verify.")
        else:
            try:
                with st.spinner("Generating grounded answer..."):
                    response = pipeline.query(question, top_k=top_k)
            except ValueError:
                st.error(
                    "Knowledge base is empty. Run `python -m cpf_bot.rebuild_vectorstore` in your terminal (after activating the virtualenv) to regenerate embeddings."
                )
                return
            st.markdown("### Answer")
            st.markdown(f"<div class='cpf-answer'>{response.answer}</div>", unsafe_allow_html=True)
            if summarize:
                with st.spinner("Summarizing evidence..."):
                    summary = pipeline.summarize_sources(question, top_k=top_k)
                st.markdown("### Evidence Summary")
                st.markdown(f"<div class='cpf-answer'>{summary}</div>", unsafe_allow_html=True)
            st.markdown("### Sources")
            for citation, doc_text in zip(response.citations, response.source_documents):
                with st.expander(citation):
                    st.write(doc_text)
            download_text = f"Question: {question}\n\nAnswer:\n{response.answer}\n\nSources: {', '.join(response.citations)}"
            st.download_button(
                "Download response",
                data=download_text,
                file_name=export_label.strip() or "cpf_bot_response.txt",
                mime="text/plain",
                use_container_width=True,
            )

    with st.expander("Knowledge base files"):
        files = ingest.list_existing_documents()
        if not files:
            st.write("No documents found. Add markdown/PDF files under `cpf_bot/data/` and rerun the rebuild script.")
        else:
            for file in files:
                st.write("•", file.name)


def _render_admin_panel(pipeline: RAGPipeline) -> None:
    st.subheader("Admin – Document Management")
    st.caption("Upload new CPF references or rebuild the knowledge base. Changes affect all users.")
    if not admin_password_configured():
        st.error("Admin password is not configured. Set `admin_password` in secrets or `CPF_ADMIN_PASSWORD`.")
        return
    if not st.session_state.get("admin_authenticated"):
        st.info("Enter the admin password to manage documents.")
        password = st.text_input("Admin password", type="password", key="admin_password_input")
        if st.button("Unlock admin tools", key="unlock_admin"):
            if verify_admin_password(password or ""):
                st.session_state.admin_authenticated = True
                st.success("Admin tools unlocked.")
            else:
                st.error("Incorrect admin password.")
        return

    if st.button("Log out", key="admin_logout"):
        st.session_state.admin_authenticated = False
        st.session_state.pop("admin_password_input", None)
        st.stop()
    uploaded = st.file_uploader("Upload CPF FAQ (PDF/Markdown)", type=["pdf", "md", "txt"])
    if uploaded:
        save_path = _save_uploaded_file(uploaded)
        if save_path:
            st.success(f"Saved {save_path.name} to the uploads folder. Click rebuild to index it.")
    if st.button("Rebuild knowledge base", type="secondary"):
        try:
            with st.spinner("Embedding documents via OpenAI..."):
                pipeline.refresh_store()
            st.success("Vector store refreshed.")
        except Exception as exc:  # pragma: no cover - handled interactively
            st.error(f"Failed to rebuild vector store: {exc}")
    st.markdown("### Indexed documents")
    _render_document_table()


def page_bot() -> None:
    pipeline = _init_pipeline()
    if not pipeline:
        return
    _inject_styles()
    user_tab, admin_tab = st.tabs(["Public User", "Admin"])
    with user_tab:
        _render_user_panel(pipeline)
    with admin_tab:
        _render_admin_panel(pipeline)


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
