#!/bin/bash
# ========================================================
# FastAPI Model Serving - macOS / Linux Launcher
# Serves fine-tuned models via OpenAI-compatible REST API
# ========================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

PORT="${1:-8000}"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "[INFO] Starting FastAPI Model Server on http://127.0.0.1:${PORT} ..."
python serve_api.py --host 127.0.0.1 --port "${PORT}"
