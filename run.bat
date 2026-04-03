@echo off
cd /d %~dp0
if exist .venv\Scripts\python.exe (
    echo Starting Novel generator using virtual environment...
    .venv\Scripts\python.exe app.py
) else (
    echo Virtual environment not found. Trying system python...
    python app.py
)
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%.
    pause
)
