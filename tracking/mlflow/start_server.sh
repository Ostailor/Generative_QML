#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_STORE_URI=${BACKEND_STORE_URI:-"sqlite:///${THIS_DIR}/mlflow.db"}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"${THIS_DIR}/mlruns"}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-5000}

if lsof -iTCP:"${PORT}" -sTCP:LISTEN -Pn >/dev/null 2>&1; then
  echo "[mlflow] ERROR: Port ${PORT} is already in use. Set PORT to a free port (e.g., 5001) and rerun." >&2
  exit 1
fi

mkdir -p "${ARTIFACT_ROOT}"

echo "[mlflow] Starting server with backend ${BACKEND_STORE_URI} and artifacts at ${ARTIFACT_ROOT} on port ${PORT}" >&2
mlflow server \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --host "${HOST}" \
  --port "${PORT}"
