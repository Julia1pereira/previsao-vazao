#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-/app/src}"
export PORT="${PORT:-8081}"

# echo "[init] Executando pipeline completo..."
# python -m src.main

echo "[init] Pipeline concluido. Subindo API em 0.0.0.0:${PORT}"
exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT}"
