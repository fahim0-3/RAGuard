# RAGuard

**A low-latency, citation-verified RAG assistant for policy questions.**

RAGuard retrieves policy evidence, decides whether that evidence is sufficient, and returns an answer only when each claim can be tied back to the retrieved text. When a request is vague, unsupported, or unsafe, it clarifies, escalates, or abstains instead of guessing.

The project is designed as a practical customer-support policy desk, with a FastAPI backend, Streamlit interface, PostgreSQL/pgvector storage, local retrieval models, optional hosted reranking, and multi-provider LLM routing.

> **Portfolio note:** The bundled policies and values are synthetic demonstration data. They are intentionally varied to exercise retrieval, exceptions, safety decisions, and evaluation workflows.

## Why RAGuard?

A normal retrieval-augmented application can retrieve related text and still produce an incorrect answer. RAGuard treats evidence and citations as decision gates:

- Combines semantic and keyword retrieval so both natural-language queries and exact codes remain searchable.
- Reranks the most relevant passages before they reach the LLM.
- Checks whether available evidence covers the question before generation.
- Validates citations and material claims before returning an answer.
- Returns a safe abstention or clarification when the corpus does not support a reliable response.

## Highlights

- **Hybrid retrieval:** local BGE-M3 embeddings, BM25, reciprocal-rank fusion, deduplication, and pgvector persistence.
- **Fast warm-path retrieval:** an in-memory dense index and prebuilt BM25 index avoid database round trips for the compact demonstration corpus.
- **Evidence-first generation:** deterministic signals and structured evidence grading block unsupported generation.
- **Citation verification:** validates cited labels, policy identifiers, and numeric claims against retrieved passages.
- **Resilient model routing:** configurable Groq, Gemini, OpenRouter, and Ollama routing with token budgets, circuit breakers, cooldowns, and bounded fallbacks.
- **Reranking choices:** local cross-encoder privacy path or opt-in Voyage hosted reranking with a local fallback.
- **Operational visibility:** per-stage timings, provider/fallback metadata, readiness checks, protected metrics, and a repeatable latency benchmark.
- **Evaluation assets:** regression tests, a golden dataset, and a separate holdout set for expanded synthetic policies.

## Architecture

```mermaid
flowchart LR
    U[User] --> UI[Streamlit UI]
    UI --> API[FastAPI /query]
    API --> S[Sanitize and classify]
    S --> A{Ambiguous or high risk?}
    A -->|Yes| Safe[Clarify or escalate]
    A -->|No| R[Hybrid retrieval: BGE-M3 + BM25 + RRF]
    R --> RR[Rerank evidence]
    RR --> E{Evidence sufficient?}
    E -->|No| H[Bounded rewrite or abstain]
    E -->|Yes| G[Grounded generation]
    G --> V{Citations and claims verified?}
    V -->|Yes| Answer[Verified answer]
    V -->|No| H
    R <--> DB[(PostgreSQL + pgvector)]
```

## Current local benchmark

The latest warm local benchmark used ten representative requests with the API running at `127.0.0.1:8000`, a 45-second interval to avoid free-tier rate limits, Groq as the active provider, and no provider fallbacks.

| Metric | Result |
| --- | ---: |
| Median end-to-end latency (p50) | **2.36 s** |
| p95 end-to-end latency | **3.84 s** |
| Mean LLM calls per request | **2.3** |
| Provider fallbacks | **0 / 10** |
| Outcomes | 7 answered, 2 abstained, 1 clarified |

These figures describe one local warm run, not a cloud-service guarantee. Provider latency, model availability, corpus size, and hardware affect results.

## Tech stack

| Area | Tools |
| --- | --- |
| API and workflow | Python, FastAPI, Pydantic, LangGraph |
| Retrieval | PostgreSQL, pgvector, BM25, BGE-M3, RRF |
| Reranking | SentenceTransformers cross-encoder, optional Voyage AI |
| Generation | Groq, Gemini, OpenRouter, or Ollama |
| Frontend | Streamlit |
| Quality | Pytest, Ruff, golden and holdout evaluations |
| Observability | Structured logs, OpenTelemetry hooks, latency reports |

## Project structure

```text
api/                    FastAPI routes, schemas, admission control, metrics
data/policies/          Synthetic policy corpus used by the demo
data/evaluation/        Holdout questions for expanded-corpus evaluation
frontend/               Streamlit customer-support interface
scripts/                Windows launcher, ingestion, diagnostics, benchmarks
src/config/             Environment-backed configuration and validation
src/ingestion/          Document chunking and pgvector ingestion
src/retrieval/          Embeddings, BM25, memory index, RRF, vector search
src/reranking/          Local and hosted reranker implementations
src/generation/         LLM factory, routing, rate limits, prompts
src/self_healing/       LangGraph, evidence grading, retries, verification
src/evaluation/         Golden dataset and offline evaluation tools
tests/                  Unit, API, routing, ingestion, and regression tests
```

## Quick start (Windows)

### 1. Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL with the `vector` extension, locally or through a managed service
- At least one configured LLM provider

### 2. Install and configure

```powershell
uv sync --locked --all-groups --python 3.12
Copy-Item .env.example .env
```

Edit `.env` with your database connection and provider credentials. Never commit `.env` or API keys.

Minimal local configuration:

```dotenv
DATABASE_URL=postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require

LLM_PROVIDER=groq
LLM_ROUTING_MODE=dynamic
GROQ_API_KEY=your-key
GROQ_MODEL=openai/gpt-oss-20b

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_PROVIDER=voyage
RERANKER_REMOTE_ALLOWED=true
VOYAGE_API_KEY=your-key
```

See [.env.example](.env.example) for the complete, non-secret configuration contract. Set `RERANKER_PROVIDER=local` if you do not want policy passages to be sent to Voyage.

### 3. Prepare the database and ingest the corpus

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-native.ps1 -Task setup-db
powershell -ExecutionPolicy Bypass -File .\scripts\run-native.ps1 -Task ingest -Reset
```

The current demonstration corpus contains nine synthetic policy documents, covering refunds, returns, deliveries, damage, payments, warranties, promotions, and account/privacy requests.

### 4. Run the application

Open two PowerShell windows.

```powershell
# Window 1: API
powershell -ExecutionPolicy Bypass -File .\scripts\run-native.ps1 -Task api
```

```powershell
# Window 2: frontend
powershell -ExecutionPolicy Bypass -File .\scripts\run-native.ps1 -Task frontend
```

Open `http://127.0.0.1:8501` in a browser. The API listens on `http://127.0.0.1:8000`.

### 5. Confirm readiness

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready | ConvertTo-Json -Depth 6
```

`/health` confirms that the process is running. `/ready` confirms database, corpus, embedding, and reranker readiness; it can remain `503` while models are downloading or warming up.

## Query the API

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/query `
  -ContentType 'application/json' `
  -Body '{"query":"How long does a refund take to reach my credit card?"}'
```

Representative response fields:

```json
{
  "outcome": "answer",
  "answer": "...",
  "citations": [{"citation_label": "[1]", "policy_id": "REF-001"}],
  "evidence_sufficient": true,
  "verification_status": "supported",
  "retry_count": 0,
  "latency_ms": 2360
}
```

## Evaluation and testing

### Full test suite

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

Latest local validation: **998 passed, 137 deselected, 10 dependency warnings**. The warnings come from FastAPI/Starlette dependency deprecations and do not represent failing tests.

### Latency benchmark

Start the API, then run:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_query_latency.py `
  --base-url http://127.0.0.1:8000 `
  --spacing 45 `
  --out reports\latency_current.json
```

The benchmark records aggregate latency, stage timings, LLM call counts, retries, outcomes, provider fallbacks, and skipped providers. It excludes answer text and policy passages from the report.

### Generalization testing

Use [extended_holdout_cases.json](data/evaluation/extended_holdout_cases.json) only after changes to prompts, models, retrieval, or thresholds. Do not tune against its questions; compare its results with previous runs instead.

Useful evaluation patterns:

- **Development strong, holdout weak:** the pipeline is overfitting to known wording or rules.
- **Both weak:** retrieval, reranking, prompts, or corpus coverage need work.
- **Unsupported question answered:** evidence thresholds are too permissive.
- **Supported question abstained:** inspect retrieved passages and reranker scores before changing thresholds.

## Safe failure behavior

| Situation | RAGuard response |
| --- | --- |
| Question is vague | Requests clarification |
| No policy supports the request | Abstains and explains the missing evidence |
| Request is high risk | Escalates before generation |
| Provider is unavailable | Uses an eligible bounded fallback or returns a safe service error |
| Citation fails verification | Regenerates within its budget or abstains |
| API is starting | `/ready` stays unavailable until dependencies are ready |

## Operational endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Process liveness |
| `GET /ready` | Database, corpus, model, and reranker readiness |
| `GET /config` | Safe active configuration summary |
| `POST /query` | Grounded policy question answering |
| `GET /admin/metrics` | Protected operational metrics |
| `POST /admin/reindex` | Protected corpus-index refresh |

Administrative endpoints require the `X-Admin-Key` request header and the `ADMIN_API_KEY` configured on the API server.

## Design decisions

- **Local embeddings:** avoids a remote embedding call on every query.
- **In-memory dense index:** appropriate for the compact demonstration corpus; pgvector remains the persistent source of truth.
- **Bounded self-healing:** retries and regeneration can improve recovery but cannot loop indefinitely or create unbounded cost.
- **Provider budgets and circuit breakers:** prevent a rate-limited provider from repeatedly inflating response latency.
- **Offline model cache:** enables predictable restarts after the first model download and avoids repeated Hugging Face downloads.
- **Strict evidence gates:** correctness is preferred over an unsupported, confident-sounding answer.

## Limitations and next steps

- The policy corpus is deliberately small and synthetic; it is not evidence of enterprise-scale retrieval performance.
- Hosted provider response times and quotas vary; benchmark before changing routing or provider settings.
- Local CPU reranking is slower than GPU inference for larger candidate sets.
- A larger real-world corpus should include versioning, access controls, ingestion review, and a broader heldout evaluation set.

Next steps are to expand each policy category gradually, add 100–200 heldout questions, evaluate retrieval and citation quality after each corpus change, and keep latency benchmarks separate from provider rate-limit tests.