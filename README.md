# RAGuard

RAGuard is a self-healing, citation-verified RAG system for grounded policy
and document question answering. It is designed to answer only when retrieved
evidence supports the request, and to clarify, escalate, or abstain otherwise.

## The problem

Ordinary RAG can retrieve related text and still produce an unsafe answer: it
may apply the wrong policy, omit a required condition, invent a number, or cite
a passage that does not support its conclusion. RAGuard treats evidence
sufficiency and claim-level citation support as gates rather than presentation
features.

## Key features

- Hybrid dense and BM25 retrieval with reciprocal-rank fusion (RRF).
- Top-20 candidate retrieval and top-5 evidence reranking.
- Risk routing, ambiguity clarification, bounded query-rewrite retries, and
  safe abstention.
- Structured evidence grading before generation.
- Grounded answer generation with closed structured-output contracts.
- Claim-level citation verification, including deterministic checks for cited
  identifiers and numeric claims.
- Static or deterministic dynamic LLM-provider routing with bounded fallback.
- Local/private reranking by default, with explicit opt-in hosted Voyage
  reranking.

## Production architecture

```mermaid
flowchart TD
    Q[POST /query] --> S[Sanitize and classify]
    S --> R{Risk route}
    R -->|high risk| E[Escalate]
    R -->|clear| A{Ambiguity check}
    A -->|ambiguous| C[Clarify]
    A -->|clear| H[Hybrid retrieval: dense + BM25 + RRF]
    H --> RR[Rerank top 20 to top 5]
    RR --> G{Evidence grading}
    G -->|weak and retry remains| RW[Rewrite query]
    RW --> H
    G -->|weak and exhausted| AB[Abstain]
    G -->|sufficient| GEN[Grounded generation]
    GEN --> V{Claim-level citation verification}
    V -->|supported| F[Verified answer]
    V -->|unsupported and retry remains| GEN
    V -->|unsupported and exhausted| AB
```

### Request flow

The FastAPI `POST /query` endpoint invokes one LangGraph run. It does not
duplicate retrieval, reranking, evidence, or verification logic in the HTTP
layer. Each response includes the outcome, verified citations, retry count,
verification summary, and bounded execution trace.

### Hybrid retrieval

RAGuard embeds the query with local `BAAI/bge-m3` by default, performs vector
search in PostgreSQL/pgvector, performs BM25 search over the ingested corpus,
then fuses ranked results with RRF. The normal retrieval contract is 20 fused
candidates followed by 5 reranked evidence chunks. Chunk metadata is retained
throughout so citations always resolve to real retrieved passages.

### Reranking

`BAAI/bge-reranker-v2-m3` is the default local/private/offline reranker.
`RERANKER_DEVICE=auto` uses CUDA only when it is available; CPU remains the
fallback. Models are reused and warmed outside the request path.

Voyage `rerank-2.5-lite` is an explicit hosted ordering profile, never an
automatic route. It requires both `RERANKER_PROVIDER=voyage` and
`RERANKER_REMOTE_ALLOWED=true`; an API key alone never transmits document
chunks. Successful Voyage ordering is followed by fixed-order local BGE scoring
of those five chunks so existing confidence logic keeps BGE-compatible scores.
Hosted timeout, rate-limit, temporary-service, or malformed-response failures
fall back deterministically to local BGE when configured.

### Evidence sufficiency and safe abstention

Before answer generation, deterministic retrieval signals and a structured
evidence grader decide whether the evidence is sufficient. Weak evidence can
trigger bounded query rewriting; exhausted or unavailable evidence produces a
safe abstention. Risky requests are escalated before retrieval, and genuinely
ambiguous questions receive a clarification request rather than a guess.

### Grounded generation and citation verification

The generator receives only the selected evidence and returns strict structured
output. Citations are references to supplied chunk labels, not generated
metadata. Verification rejects unresolved labels, unsupported claims, and
numeric or identifier claims that do not appear in their cited passage. A
response is finalized only after its claims are supported.

### LLM provider routing and fallback

Gemini is the default hosted provider. Groq is available for strict
structured-output/evaluation workloads, and Ollama supports local/offline
execution. `LLM_ROUTING_MODE=static` preserves manual `LLM_PROVIDER`
selection. In `dynamic` mode, deterministic configuration selects one provider
at the start of a graph run and keeps it consistent; eligible timeout, 429,
unavailable-provider, and provider-side `json_validate_failed` errors may use
the existing bounded provider-fallback chain. A fallback consumes a normal
graph LLM-call permit. Static provider selection fails closed.

## Tech stack

- Python 3.11 or 3.12, FastAPI, Uvicorn, and Pydantic.
- LangGraph and LangChain Core for the workflow and structured LLM calls.
- PostgreSQL with pgvector, Psycopg, and Redis support for distributed
  admission control.
- `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, SentenceTransformers, Transformers,
  and PyTorch for local retrieval/reranking.
- Gemini, Groq, Ollama, and optional Voyage integrations.
- Streamlit frontend, Ruff, Pytest, Docker Compose, and GitHub Actions.

## Project structure

```text
api/                    FastAPI transport, schemas, admission, observability
data/policies/          Versioned demonstration policy corpus
deploy/                 Production environment and deployment contracts
frontend/               Streamlit client
scripts/                Native launch, ingestion, smoke, and offline tooling
src/config/             Settings and production validation
src/ingestion/          Chunking and pgvector ingestion
src/retrieval/          Embeddings, BM25, vector search, RRF, deduplication
src/reranking/          Local and Voyage reranker providers
src/generation/         Provider factory, routing, prompts, structured schemas
src/self_healing/       LangGraph, evidence, safety, and verification stages
src/evaluation/         Golden dataset, regression gates, offline evaluation
tests/                  Offline, integration, model, routing, and API tests
```

## Native setup

Prerequisites: Python 3.11/3.12, [uv](https://docs.astral.sh/uv/), and a
PostgreSQL database with the `vector` extension. Native Windows use is
supported with a managed pgvector database such as Neon.

```powershell
uv sync --locked --all-groups --python 3.12
Copy-Item .env.example .env
```

Set `DATABASE_URL` to a direct PostgreSQL URL with `sslmode=require` for a
managed database. Enable `vector` once, then check connectivity:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-native.ps1 -Task setup-db
powershell -ExecutionPolicy Bypass -File .\scripts\run-native.ps1 -Task check-db
```

### Environment variables

Copy `.env.example`; do not commit `.env` or real credentials. Common settings
are shown below as names and placeholders only.

```dotenv
RAGUARD_ENVIRONMENT=development
DATABASE_URL=postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require

LLM_PROVIDER=gemini
LLM_ROUTING_MODE=static
GOOGLE_API_KEY=store-in-a-secret-manager

# Optional providers
GROQ_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
VOYAGE_API_KEY=

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_PROVIDER=local
RERANKER_REMOTE_ALLOWED=false
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=auto

GRAPH_LLM_CALL_LIMIT=8
LLM_EXECUTION_PROFILE=baseline
```

For the explicit hosted reranker profile, set
`RERANKER_PROVIDER=voyage`, `RERANKER_REMOTE_ALLOWED=true`, and keep
`RERANKER_FALLBACK_PROVIDER=local`. See `.env.example` for the full,
non-secret configuration contract.

### Ingest the corpus

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-native.ps1 -Task ingest -Reset
```

The validated native v1 path indexed 22 chunks.

### Start the native API

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-native.ps1 -Task api
```

Use `GET /health` for process liveness and `GET /ready` for database, corpus,
provider, embedding-model, and reranker readiness.

## Query the API

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/query `
  -ContentType 'application/json' `
  -Body '{"query":"How long does a refund take to reach my credit card?"}'
```

Representative verified response fields:

```json
{
  "outcome": "answer",
  "answer": "Refunds to cards take 5 to 7 business days.",
  "citations": [{"citation_label": "[1]", "policy_id": "REF-001"}],
  "evidence_sufficient": true,
  "verification_status": "supported",
  "verified_claim_count": 2,
  "unsupported_claim_count": 0,
  "reranker_used": true
}
```

The exact wording and citation excerpt depend on the selected evidence; an
answer is valid only when verification reports `supported`.

## Testing and validated results

Run static checks and the full test suite:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

Validated native v1 results:

- Neon PostgreSQL connection passed.
- Corpus ingestion passed with 22 indexed chunks.
- FastAPI startup and service smoke check passed.
- A real `/query` request returned a grounded refund answer with
  `evidence_sufficient=true`, `verification_status=supported`, two verified
  claims, zero unsupported claims, and reranking used.
- Ruff passed.
- Full Pytest passed: **852 passed, 133 skipped, 0 failed**.

### Evaluation methodology

The committed v3 golden dataset contains 62 cases: 38 expected answers, 18
expected abstentions, 4 clarifications, and 2 escalations. It is schema-tested
and keyword-grounded against the local corpus. Retrieval evaluation tracks
Hit@k, Recall@k, MRR@k, exact-keyword recall, and citation identifier validity;
generation/safety evaluation records answer, abstain, clarify, escalate, and
citation outcomes. Regression gates are kept separate from historical measured
reports so an unavailable layer is reported as blocked rather than as a pass.

### Production versus evaluation-only research

The production graph uses the BGE-compatible confidence path described above.
Voyage confidence calibration and the enhanced answerability/evidence-ablation
experiments live only in `src/evaluation/` and `scripts/`. That research is
frozen for v1, does not run from FastAPI, and **the enhanced candidate is not a
production feature**.

## Optional Docker deployment

Docker is optional for this placement-project release. When desired, Docker
Compose can start PostgreSQL/pgvector, Redis, API, and the Streamlit frontend:

```bash
docker compose --profile full up --build -d
```

The API image runs Uvicorn on port 8000 and exposes `/health` for its container
health check. Native deployment remains the validated v1 path.

## Known limitations

- The demonstration corpus is small (22 chunks across six policy documents),
  so its retrieval metrics should not be generalized to large enterprise
  corpora.
- Hosted providers are subject to network availability and rate limits; routing
  and reranker fallback fail closed when a safe replacement is unavailable.
- The default local BGE reranker is CPU-intensive on modest hardware.
- Redis is needed for coordinated admission control across multiple API
  replicas; the native single-process launcher uses local admission control.

## Future work

- Validate on a larger, versioned policy corpus and broader held-out safety
  dataset.
- Run a separately approved production evaluation before considering any
  evaluation-only answerability work for promotion.
- Add deployment-specific load, observability, and recovery testing for a
  multi-replica environment.
