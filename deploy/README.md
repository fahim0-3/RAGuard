# Production deployment contract

RAGuard deploys as two independent services:

1. The API uses `Dockerfile`, one worker per container, a remote `/models`
   persistent disk, Neon/PostgreSQL with pgvector, and Redis when replicas may
   exceed one.
2. The Streamlit UI uses `Dockerfile.frontend` and only needs the public API URL.

The API image explicitly installs CPU-only PyTorch. Generic Linux PyTorch can
pull several gigabytes of CUDA libraries despite `MODEL_DEVICE=cpu`; the build
must not remove `TORCH_INDEX_URL` unless the deployment intentionally uses a GPU.

Before deployment, populate secrets in the provider's secret manager and run:

```bash
python -m scripts.production_preflight --check-database --check-redis
```

The command prints only safe status categories. It never prints connection
strings or key values. A release is eligible only when it exits zero.

After deployment, validate the public API:

```bash
python -m scripts.smoke_service \
  --base-url https://api.example.com \
  --expected-profile local_compact \
  --deadline-s 600
```

`/health` proves process liveness. `/ready` remains closed until the database,
corpus, provider configuration, embedding model, and enabled reranker are ready.

## Remote storage

- `/models`: persistent remote disk or cache volume for Hugging Face weights.
- PostgreSQL/Neon: chunks, metadata, 1024-dimensional embeddings, and indexes.
- Redis: ephemeral distributed admission counters and leases only.
- Image registry: application layers and CPU Python dependencies.

None of these are stored on the developer laptop when builds and runtime are
executed by the cloud provider or GitHub Actions.

## Rollback

Deploy immutable image tags based on the Git commit SHA. Keep at least the
previous successful API and frontend tags. Rollback means repointing both
services to those tags; database schema changes must remain backward compatible.

## Render production target

The repository root `render.yaml` defines the selected P1 target:

- `raguard-api-fahim03`: one Ohio `pro plus` API instance (8 GB RAM), a 5 GB
  disk mounted at `/models`, and the compact runtime profile.
- `raguard-ui-fahim03`: a small Streamlit web service.
- `raguard-admission-fahim03`: private Redis-compatible admission state.
- Neon remains the pgvector database; the Blueprint prompts for its direct TLS
  URL instead of creating a duplicate database.

The disk deliberately fixes the API at one instance. Render disks are
single-instance runtime storage and disable zero-downtime deploys. Scale the API
vertically; moving model weights to shared object storage or baking them into an
image is required before horizontal scaling.

Create the Blueprint from the GitHub repository, provide `DATABASE_URL` and
`GOOGLE_API_KEY` when prompted, and review the paid API/disk estimate before
applying it. Render generates `ADMIN_API_KEY`; copy that value into an approved
secret manager if operators need the protected endpoints.
