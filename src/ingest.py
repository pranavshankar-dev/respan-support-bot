"""
ingest.py — Step 1
==================
Loads all .txt and .md files from the docs/ folder, splits them into chunks,
generates embeddings via OpenAI, and stores them in a local ChromaDB vector store.

Run:
    python src/ingest.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Config ─────────────────────────────────────────────────────────────────
DOCS_DIR       = Path(__file__).parent.parent / "docs"
CHROMA_DIR     = Path(__file__).parent.parent / "chroma_db"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50
COLLECTION     = "support_docs"
# ───────────────────────────────────────────────────────────────────────────


def load_documents():
    """Load all .txt and .md files from the docs directory."""
    if not DOCS_DIR.exists():
        print(f"[ERROR] docs/ directory not found at {DOCS_DIR}")
        print("  Create the docs/ folder and add .txt or .md help files to it.")
        sys.exit(1)

    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    txt_docs = loader.load()

    loader_md = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    md_docs = loader_md.load()

    all_docs = txt_docs + md_docs
    print(f"\nLoaded {len(all_docs)} document(s) from {DOCS_DIR}")
    return all_docs


def split_documents(docs):
    """Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def embed_and_store(chunks):
    """Generate embeddings and persist to ChromaDB."""
    print(f"\nEmbedding and storing in ChromaDB at {CHROMA_DIR} ...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete existing collection so re-running always starts fresh
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("  Cleared existing vector store")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )

    count = vectorstore._collection.count()
    print(f"\n✓ Ingestion complete. {count} chunks stored in chroma_db/")
    return vectorstore


def main():
    print("=" * 50)
    print("  Support Bot — Document Ingestion")
    print("=" * 50)

    docs   = load_documents()
    chunks = split_documents(docs)
    embed_and_store(chunks)

    print("\nNext step: python src/rag.py \"your question here\"")


if __name__ == "__main__":
    main()
