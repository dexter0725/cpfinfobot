# CPF Board Info Verification Bot

A Streamlit-based Retrieval-Augmented Generation (RAG) prototype that helps members of the public verify CPF-related claims by grounding LLM responses in official CPF documents.

## Features
- **Single-click Q&A** – Minimalist Streamlit UI focused on asking CPF questions and reading grounded answers.
- **Curated corpus** – Static CPF FAQ summaries covering retirement, housing, MediSave, contributions, and education stored under `cpf_bot/data/sample_docs/`.
- **RAG-backed answers** – GPT-4o mini answers are constrained to retrieved CPF sources with citations (Chroma vector store).
- **Streamlined Q&A** – Public users only see the essentials: ask a question and read the grounded answer with cited snippets.
- **About & Methodology pages** – Explain the problem context and solution approach.

## Project Structure
```
cpf_bot/
├── data/
│   ├── sample_docs/        # curated CPF FAQs included with the repo
│   └── uploads/            # admin uploads at runtime
├── vectorstore/            # (legacy) folder; persistent index now saved under ../vector_db/cpf_bot_index
├── ingest.py               # document loading + embedding helpers
├── rag.py                  # retrieval + LLM orchestration
├── streamlit_app.py        # Streamlit UI entry point
└── .env.example            # template for secrets
```

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` in `cpf_bot/` and add your key:
   ```bash
   cp cpf_bot/.env.example cpf_bot/.env
   echo "OPENAI_API_KEY=sk-your-key" >> cpf_bot/.env
   ```
3. Add secrets for Streamlit (password + optional key override):
   ```toml
   # cpf_bot/.streamlit/secrets.toml
   OPENAI_API_KEY = "sk-your-key"
   password = "set-a-strong-password"
   ```
   (For local scripts you can alternatively export `CPF_APP_PASSWORD`.)
4. Build the vector store once (requires OpenAI network access):
   ```bash
   source .venv/bin/activate
   python -m cpf_bot.rebuild_vectorstore
   ```
5. Run the Streamlit app:
   ```bash
   streamlit run cpf_bot/streamlit_app.py
   ```

## Usage Workflow
- **Maintain corpus**: Drop new CPF references into `cpf_bot/data/uploads/` and rerun `python -m cpf_bot.rebuild_vectorstore` when you want to refresh embeddings.
- **Public user**: Paste any CPF-related claim and review the grounded answer plus cited snippets.

## Deployment
Deploy directly to [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push this folder to a public GitHub repo.
2. Create a new Streamlit app targeting `cpf_bot/streamlit_app.py`.
3. Set the `OPENAI_API_KEY` secret in the Streamlit Cloud dashboard.

## API / Model Usage
- **Embeddings**: `text-embedding-3-small` via `OpenAIEmbeddings` (LangChain).
- **Vector store**: `Chroma` persisted under `vector_db/cpf_bot_index/`.
- **LLM**: `gpt-4o-mini` via `ChatOpenAI` for grounded answers and summaries.


> The knowledge base ships with curated CPF FAQ summaries. Update them manually and rerun the rebuild script whenever new official information is available.

## Sample Data
Three CPF knowledge snippets are preloaded under `data/sample_docs/` covering retirement payouts, housing usage, and contribution rules. Add more official CPF FAQs to improve coverage.
