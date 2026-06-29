@echo off
REM Double-click me. Edit run.py first (paste your link + set I_HAVE_RIGHTS=True).
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run.py
) else (
    echo No .venv found - using system python. If imports fail, create the venv:
    echo     python -m venv .venv ^&^& .venv\Scripts\python -m pip install -e .
    python run.py
)

echo.
pause
