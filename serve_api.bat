@echo off
REM ========================================================
REM FastAPI Model Serving - Windows Launcher
REM Serves fine-tuned models via OpenAI-compatible REST API
REM ========================================================

set PORT=8000
if not "%~1"=="" set PORT=%~1

IF EXIST "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo [INFO] Starting FastAPI Model Server on http://127.0.0.1:%PORT% ...
python serve_api.py --host 127.0.0.1 --port %PORT%
pause
