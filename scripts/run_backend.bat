@echo off
REM Start the FetalAI FastAPI backend on http://localhost:8000 (docs at /docs).
cd /d "%~dp0.."
call ".venv\Scripts\activate.bat"
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --no-access-log
