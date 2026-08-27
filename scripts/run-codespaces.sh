#!/usr/bin/env bash
# Run RAGuard natively inside GitHub Codespaces with a managed pgvector database.
#
# This deliberately starts no Docker services. DATABASE_URL must point to a
# direct managed PostgreSQL endpoint (for example Neon), and model weights stay
# below the persistent Codespaces workspace rather than on the developer's PC.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/run-codespaces.sh <api|frontend|ingest|setup-db|check-db> [options]

Examples:
  bash scripts/run-codespaces.sh check-db
  bash scripts/run-codespaces.sh setup-db
  bash scripts/run-codespaces.sh ingest --no-reset
  bash scripts/run-codespaces.sh api

The launcher defaults to the compact reranker profile. To use the full
reranker instead, prefix a command with RAGUARD_RUNTIME_PROFILE=full.
EOF
}

task="${1:-api}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$task" in
    api|frontend|ingest|setup-db|check-db) ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "Unknown task: $task" >&2; usage >&2; exit 2 ;;
esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${RAGUARD_CODESPACES_DATA_DIR:-${project_root}/.codespaces}"
model_dir="${data_dir}/models"

# These exports take precedence over a checked-out .env. The cache is both
# persistent across a Codespace rebuild and ignored by Git/Docker contexts.
export HF_HOME="$model_dir"
export HF_HUB_CACHE="$model_dir/hub"
export TRANSFORMERS_CACHE="$model_dir/transformers"
export SENTENCE_TRANSFORMERS_HOME="$model_dir/sentence-transformers"
export HF_HUB_DISABLE_XET=1
export DISABLE_SAFETENSORS_CONVERSION=1
export RAGUARD_RUNTIME_PROFILE="${RAGUARD_RUNTIME_PROFILE:-local_compact}"
export ADMISSION_BACKEND=local

mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$SENTENCE_TRANSFORMERS_HOME"

if [[ -z "${DATABASE_URL:-}" && ! -f "$project_root/.env" ]]; then
    echo "DATABASE_URL is required. Add it as a Codespaces secret before running RAGuard." >&2
    exit 2
fi

cd "$project_root"
case "$task" in
    api)
        exec uv run --frozen --no-sync uvicorn api.main:app --host 0.0.0.0 --port 8000 "$@"
        ;;
    frontend)
        exec uv run --frozen --no-sync streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501 "$@"
        ;;
    ingest)
        exec uv run --frozen --no-sync python -m src.ingestion.ingest "$@"
        ;;
    setup-db)
        exec uv run --frozen --no-sync python -m scripts.setup_remote_db "$@"
        ;;
    check-db)
        exec uv run --frozen --no-sync python -m scripts.check_remote_db "$@"
        ;;
esac
