"""CLI entry point to rebuild the CPF Bot vector store."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from . import ingest

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    package_env = Path(__file__).resolve().parent / ".env"
    load_dotenv(package_env, override=False)
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Update .env before rebuilding the vector store.")
    ingest.ensure_directories()
    ingest.build_vector_store(api_key)
    print("Vector store rebuilt successfully at", ingest.VECTOR_DIR)


if __name__ == "__main__":
    main()
