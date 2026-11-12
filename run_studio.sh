#!/bin/bash
# ========================================================
# LLM Fine-Tuning Studio - macOS / Linux Launcher
# Supports Apple Silicon (MLX) & Linux (CUDA / CPU)
# ========================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "[INFO] Starting LLM Fine-Tuning Studio..."

if [ ! -f "venv/bin/activate" ]; then
    echo "[INFO] Virtual environment not found. Creating venv..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "[INFO] Launching Streamlit Studio..."
streamlit run app.py --server.port=8501

