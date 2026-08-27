#!/usr/bin/env bash
set -euo pipefail

# `/workspaces` survives a Codespace rebuild. Keep bulky model data there, but
# make it neither version-controlled nor part of a Docker build context. The
# optional Docker Compose route also uses the postgres directory; the native
# Codespaces + Neon route leaves it unused.
data_dir="${RAGUARD_CODESPACES_DATA_DIR:?Codespaces data directory is required}"
mkdir -p "${data_dir}/models" "${data_dir}/postgres"
chmod 0777 "${data_dir}" "${data_dir}/models" "${data_dir}/postgres"

# Resolve the complete development/evaluation environment from the committed
# lock. Package wheels stay in the remote Codespace, never on the laptop.
if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --user "uv==0.11.6"
fi
uv sync --locked
