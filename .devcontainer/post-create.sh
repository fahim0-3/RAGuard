#!/usr/bin/env bash
set -euo pipefail

# `/workspaces` survives a Codespace rebuild. Keep bulky Docker bind-mounted
# model and PostgreSQL data there, but make it neither version-controlled nor
# part of a Docker build context. PostgreSQL runs as a non-vscode UID, hence
# the intentionally broad workspace-local permissions.
data_dir="${RAGUARD_CODESPACES_DATA_DIR:?Codespaces data directory is required}"
mkdir -p "${data_dir}/models" "${data_dir}/postgres"
chmod 0777 "${data_dir}" "${data_dir}/models" "${data_dir}/postgres"
