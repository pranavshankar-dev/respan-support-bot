# Support Bot with Drift Detection
### Built with Respan · RAG · LLM-as-Judge Evals · MLOps

A production-grade AI customer support bot that answers questions from help docs,
automatically evaluates every response with 3 LLM judges, and detects quality
drift over time using Respan observability.

---

## Project Structure

```
support-bot/
├── .env.example          # Template for environment variables
├── requirements.txt      # Python dependencies
├── docs/                 # Help documentation (.txt or .md files)
├── src/
│   ├── ingest.py         # Step 1: Ingest docs into ChromaDB vector store
│   ├── rag.py            # Step 2: RAG pipeline + Respan tracing
│   ├── evaluate.py       # Step 3: LLM-as-judge eval pipeline
│   ├── monitor.py        # Step 4: Drift detection + alerting
│   └── demo.py           # Run a full end-to-end demo loop
├── data/                 # Auto-created: stores eval scores (scores.json)
└── chroma_db/            # Auto-created: local vector store
```

---

## How It Works

1. **RAG Pipeline** — Help docs are chunked, embedded locally (HuggingFace), and stored in ChromaDB. User questions retrieve relevant chunks which are passed to Groq's LLM via Respan's gateway.
2. **LLM-as-Judge Evals** — Every response is automatically scored by 3 judges: Faithfulness (no hallucinations), Relevance (correct chunks retrieved), and Resolution (user's problem solved).
3. **Drift Detection** — Scores are tracked over time. If the 7-day rolling average drops below 0.70, a drift alert fires — catching doc staleness or retrieval degradation before users notice.
4. **Respan Observability** — Every LLM call is traced with latency, cost, token usage, and eval scores visible in the Respan dashboard.

---

## Prerequisites

- Python 3.10+
- A [Respan account](https://platform.respan.ai) with an API key
- A [Groq account](https://console.groq.com) with an API key (free)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/pranavshankar-dev/respan-support-bot.git
cd respan-support-bot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
RESPAN_API_KEY=your_respan_api_key_here
RESPAN_BASE_URL=https://api.respan.ai/api
GROQ_API_KEY=your_groq_api_key_here
```

**Getting your keys:**
- Respan: https://platform.respan.ai → Settings → API Keys → Create new key
- Groq: https://console.groq.com → API Keys → Create key (free)

**One more step in Respan dashboard:**
Go to Gateway → Providers → add your Groq API key so Respan can route calls through its gateway.

---

## Running the Project

### Step 1 — Ingest documents

```bash
python src/ingest.py
```

Loads all `.txt` and `.md` files from `docs/`, chunks them, generates local
embeddings (HuggingFace `all-MiniLM-L6-v2`, no API needed), and stores in ChromaDB.

```
Loaded 5 documents
Split into 42 chunks
✓ Ingestion complete. 42 chunks stored in chroma_db/
```

### Step 2 — Ask the bot a question

```bash
python src/rag.py "How do I reset my password?"
```

Retrieves relevant chunks, calls Groq via Respan gateway, logs full trace to Respan.

```
Retrieved 4 chunks from vector store
Answer: To reset your password, click "Forgot Password" on the login page...
Model used: groq/llama-3.1-8b-instant
✓ Trace logged → https://platform.respan.ai/platform/traces
```

### Step 3 — Evaluate a response

```bash
python src/evaluate.py "How do I reset my password?"
```

Runs 3 LLM-as-judge evaluators and saves scores to `data/scores.json`.

```
  Faithfulness:  1.00  ✓ (answer grounded in retrieved context)
  Relevance:     1.00  ✓ (chunks matched the query well)
  Resolution:    1.00  ✓ (question was answered)
  Average score: 1.00
```

### Step 4 — Check for drift

```bash
python src/monitor.py
```

Computes 7-day rolling average and alerts if quality drops below threshold.

```
# Healthy
✓ Quality looks good (avg 0.86 ≥ threshold 0.70)

# Drift detected
⚠ DRIFT DETECTED — average score dropped below threshold!
  Consider reingesting updated docs or revising your prompt.
```

### Step 5 — Run full demo

```bash
python src/demo.py
```

Runs 5 sample questions end-to-end (RAG → Eval → Monitor) to populate your
Respan dashboard with real traces and scores.

---

## Viewing Results in Respan

After running the bot, open your Respan dashboard:

- **Traces** → https://platform.respan.ai/platform/traces — every query as a full trace tree with latency, cost, tokens
- **Evaluations** → https://platform.respan.ai/platform/evals — faithfulness / relevance / resolution scores per trace
- **Monitoring** → https://platform.respan.ai/platform/monitoring — quality score trends over time

---

## Adding Your Own Docs

Drop any `.txt` or `.md` files into the `docs/` folder, then re-run:

```bash
python src/ingest.py
```

The vector store rebuilds automatically.

---

## Configuration

| Setting | File | Default | Description |
|---|---|---|---|
| `DRIFT_THRESHOLD` | monitor.py | 0.70 | Alert when rolling avg drops below this |
| `ROLLING_WINDOW_DAYS` | monitor.py | 7 | Days to include in rolling average |
| `TOP_K_CHUNKS` | rag.py | 4 | Chunks retrieved per query |
| `PRIMARY_MODEL` | rag.py | groq/llama-3.1-8b-instant | Primary LLM via Respan gateway |
| `FALLBACK_MODEL` | rag.py | llama-3.1-8b-instant | Direct Groq fallback |
| `CHUNK_SIZE` | ingest.py | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | ingest.py | 50 | Overlap between chunks |

---

## Tech Stack

| Component | Tool |
|---|---|
| Document ingestion | LangChain DirectoryLoader |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, free) |
| Vector store | ChromaDB |
| LLM | Groq `llama-3.1-8b-instant` (free) |
| Gateway + Tracing | Respan SDK |
| Evals | LLM-as-judge (3 custom evaluators) |
| Drift detection | Rolling average with threshold alerting |
