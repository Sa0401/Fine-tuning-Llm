@echo off
REM ========================================================
REM LLM Fine-Tuning Studio - Windows Launcher
REM Supports Windows with NVIDIA CUDA or CPU Mode
REM ========================================================

echo [INFO] Starting LLM Fine-Tuning Studio on Windows...

REM Check if virtual environment exists
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating venv with Python...
    python -m venv venv
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python 3.10+ is installed and in PATH.
        pause
        exit /b 1
    )
    echo [INFO] Installing required dependencies...
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
) ELSE (
    call venv\Scripts\activate.bat
)

echo [INFO] Launching Streamlit Studio Dashboard...
streamlit run app.py --server.port=8501
pause

