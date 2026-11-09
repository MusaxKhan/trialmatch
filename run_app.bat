@echo off
rem Starts the TrialMatch browser app on http://127.0.0.1:8000
set ADAPTER_T=%~dp0adapters\target
set ADAPTER_D=%~dp0adapters\draft
set RESULTS_DIR=%~dp0results
cd /d %~dp0app
start "" http://127.0.0.1:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000
