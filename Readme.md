# Support Bot with Drift Detection
### Built with Respan · RAG · LLM-as-Judge Evals · MLOps

A production-grade AI customer support bot that answers questions from help docs,
automatically evaluates every response with 3 LLM judges, and detects quality
drift over time using Respan observability.

---

## Project Structure

```
support-bot/
├── .env                  # API keys (never commit this)
├── requirements.txt      # Python dependencies
├── docs/                 # Your help documentation (add .txt or .md files here)
│   └── sample_docs/      # Pre-loaded sample docs to get started
├── data/
│   └── scores.json       # Persisted eval scores (auto-created)
├── src/
│   ├── ingest.py         # Step 1: Ingest docs into ChromaDB vector store
│   ├── rag.py            # Step 2: RAG pipeline + Respan tracing
│   ├── evaluate.py       # Step 3: LLM-as-judge eval pipeline
│   ├── monitor.py        # Step 4: Drift detection + alerting
│   └── demo.py           # Run a full end-to-end demo loop
└── chroma_db/            # Local vector store (auto-created by ingest.py)
```

---

## Prerequisites

- Python 3.10+
- A [Respan account](https://platform.respan.ai) with an API key
- An [OpenAI API key](https://platform.openai.com)

---

## Setup

### 1. Clone / create the project directory

```bash
mkdir support-bot && cd support-bot
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

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```
RESPAN_API_KEY=your_respan_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
RESPAN_BASE_URL=https://api.respan.ai/api
```

**Where to get your Respan API key:**
1. Sign up at https://platform.respan.ai
2. Go to Settings → API Keys → Create new key
3. Also go to Settings → LLM Providers → add your OpenAI key there too
   (so Respan gateway can call OpenAI on your behalf)

---

## Running the Project (Step by Step)

### Step 1 — Ingest documents into the vector store

```bash
python src/ingest.py
```

This reads all `.txt` and `.md` files from `docs/`, chunks them, generates
embeddings via OpenAI, and stores them in a local ChromaDB database.

Expected output:
```
Loading documents from docs/...
Loaded 8 documents
Split into 34 chunks
Embedding and storing in ChromaDB...
✓ Ingestion complete. 34 chunks stored in chroma_db/
```

### Step 2 — Ask the bot a question (with tracing)

```bash
python src/rag.py "How do I reset my password?"
```

This retrieves relevant chunks, builds an augmented prompt, calls the LLM
through Respan's gateway (with GPT-4.1 as primary, fallback to gpt-4o-mini),
and logs the full trace to your Respan dashboard.

Expected output:
```
Query: How do I reset my password?
Retrieved 4 chunks from vector store
─────────────────────────────────
Answer:
To reset your password, click "Forgot Password" on the login page...
─────────────────────────────────
Trace logged to Respan ✓
```

### Step 3 — Run the eval pipeline on a response

```bash
python src/evaluate.py "How do I reset my password?"
```

This runs 3 LLM-as-judge evaluators against the response and logs scores
back to Respan as span metadata.

Expected output:
```
Running 3 evaluators...
  Faithfulness:  0.92  ✓ (answer grounded in retrieved context)
  Relevance:     0.88  ✓ (chunks matched the query well)
  Resolution:    0.85  ✓ (question was answered)

Average score: 0.88
Score saved to data/scores.json
```

### Step 4 — Check for drift

```bash
python src/monitor.py
```

This reads all historical scores, computes a rolling 7-day average, compares
it against your threshold (0.70 by default), and prints an alert if drift is
detected.

Expected output (healthy):
```
=== Drift Monitor ===
Total evaluations tracked: 12
7-day rolling average:     0.84
Threshold:                 0.70
Status: ✓ Quality looks good
```

Expected output (drift detected):
```
=== Drift Monitor ===
Total evaluations tracked: 12
7-day rolling average:     0.61
Threshold:                 0.70
Status: ⚠ DRIFT DETECTED — average score dropped below threshold!
         Consider reingesting updated docs or revising your prompt.
```

### Step 5 — Run a full demo (all steps together)

```bash
python src/demo.py
```

This runs 5 sample questions end-to-end: RAG → Eval → Monitor, so you can
populate your Respan dashboard with real traces and scores quickly.

---

## Viewing Results in Respan

After running the bot, open your Respan dashboard:

1. **Traces** → https://platform.respan.ai/platform/traces
   See every query as a full trace tree with latency, cost, tokens

2. **Evaluations** → https://platform.respan.ai/platform/evals
   See faithfulness / relevance / resolution scores per trace

3. **Monitoring** → https://platform.respan.ai/platform/monitoring
   Build a dashboard with quality score trends over time

---

## Adding Your Own Docs

Drop any `.txt` or `.md` files into the `docs/` folder, then re-run:

```bash
python src/ingest.py
```

The vector store will be rebuilt automatically.

---

## Configuration

Key settings you can adjust in each file:

| Setting | File | Default | Description |
|---|---|---|---|
| `DRIFT_THRESHOLD` | monitor.py | 0.70 | Alert when rolling avg drops below this |
| `ROLLING_WINDOW_DAYS` | monitor.py | 7 | Days to include in rolling average |
| `TOP_K_CHUNKS` | rag.py | 4 | Number of chunks retrieved per query |
| `PRIMARY_MODEL` | rag.py | gpt-4.1 | Primary LLM via Respan gateway |
| `FALLBACK_MODEL` | rag.py | gpt-4o-mini | Fallback if primary fails |
| `CHUNK_SIZE` | ingest.py | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | ingest.py | 50 | Overlap between chunks |