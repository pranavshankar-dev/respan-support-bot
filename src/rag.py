"""
rag.py — Step 2
===============
RAG pipeline with full Respan tracing.
- Retrieves relevant chunks from ChromaDB (HuggingFace embeddings, free/local)
- Builds an augmented prompt
- Calls Groq LLM via Respan Gateway (primary) with direct Groq fallback
- Every call is traced in Respan

Run:
    python src/rag.py "How do I reset my password?"
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI, APIError, RateLimitError
from respan import Respan
from respan_instrumentation_openai import OpenAIInstrumentor
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Config ─────────────────────────────────────────────────────────────────
CHROMA_DIR      = Path(__file__).parent.parent / "chroma_db"
COLLECTION      = "support_docs"
TOP_K_CHUNKS    = 4
PRIMARY_MODEL   = "groq/llama-3.1-8b-instant"   # via Respan gateway
FALLBACK_MODEL  = "llama-3.1-8b-instant"         # direct Groq fallback
# ───────────────────────────────────────────────────────────────────────────

# Initialise Respan — auto-instruments all OpenAI-compatible calls
respan = Respan(instrumentations=[OpenAIInstrumentor()])

# Respan Gateway client — routes to Groq through Respan
gateway_client = OpenAI(
    api_key=os.getenv("RESPAN_API_KEY"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)

# Direct Groq client for fallback (OpenAI-compatible API)
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def get_vectorstore():
    """Load the ChromaDB vector store."""
    if not CHROMA_DIR.exists():
        print("[ERROR] Vector store not found. Run python src/ingest.py first.")
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def retrieve_chunks(query: str, vectorstore) -> list[dict]:
    """Retrieve top-k relevant document chunks for the query."""
    results = vectorstore.similarity_search_with_score(query, k=TOP_K_CHUNKS)
    chunks = []
    for doc, score in results:
        chunks.append({
            "content": doc.page_content,
            "source":  doc.metadata.get("source", "unknown"),
            "score":   round(float(score), 4),
        })
    return chunks


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """Build the augmented prompt with retrieved context."""
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in chunks
    )

    system_prompt = (
        "You are a helpful customer support assistant. "
        "Answer the user's question ONLY using the information in the provided context. "
        "If the answer is not in the context, say: "
        "'I don't have information about that. Please contact support@example.com.' "
        "Be concise, friendly, and accurate. "
        "Do not make up information not present in the context."
    )

    user_message = f"Context:\n{context_text}\n\nQuestion: {query}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]


def call_llm_with_fallback(messages: list[dict]) -> tuple[str, str]:
    """
    Try Groq via Respan Gateway first (traced), fall back to direct Groq.
    Returns (response_text, model_used).
    """
    # Primary: Groq through Respan gateway (fully traced)
    try:
        response = gateway_client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )
        return response.choices[0].message.content, PRIMARY_MODEL

    except (APIError, RateLimitError, Exception) as e:
        print(f"  [Gateway] Primary failed: {type(e).__name__} — {e}")
        print(f"  [Fallback] Trying direct Groq ({FALLBACK_MODEL})...")

    # Fallback: direct Groq API
    try:
        response = groq_client.chat.completions.create(
            model=FALLBACK_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )
        return response.choices[0].message.content, f"{FALLBACK_MODEL} (direct)"

    except Exception as e:
        raise RuntimeError(
            f"Both primary and fallback failed.\n"
            f"Check your GROQ_API_KEY in .env and that you added it under\n"
            f"Respan → Gateway → Providers.\n\nError: {e}"
        )


def answer_question(query: str) -> dict:
    """Full RAG pipeline: retrieve -> build prompt -> call LLM."""
    vectorstore = get_vectorstore()

    print(f"\nQuery: {query}")
    print("Retrieving relevant chunks...")

    chunks = retrieve_chunks(query, vectorstore)
    print(f"Retrieved {len(chunks)} chunks from vector store")

    messages = build_prompt(query, chunks)
    answer, model_used = call_llm_with_fallback(messages)

    return {
        "query":      query,
        "answer":     answer,
        "model_used": model_used,
        "chunks":     chunks,
        "num_chunks": len(chunks),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/rag.py \"your question here\"")
        print('Example: python src/rag.py "How do I reset my password?"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print("=" * 55)
    print("  Support Bot — RAG Pipeline (Groq + Respan tracing)")
    print("=" * 55)

    result = answer_question(query)

    print("\n" + "-" * 55)
    print("Answer:")
    print(result["answer"])
    print("-" * 55)
    print(f"Model used:       {result['model_used']}")
    print(f"Chunks retrieved: {result['num_chunks']}")
    print("\n Trace logged -> https://platform.respan.ai/platform/traces")

    respan.flush()
    return result


if __name__ == "__main__":
    main()
