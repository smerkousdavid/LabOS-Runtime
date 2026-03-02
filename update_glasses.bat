@echo off
cd /d "%~dp0"

if not exist ".venv" (
    echo Virtual environment not found. Run install.bat first.
    exit /b 1
)

call .venv\Scripts\activate.bat
python scripts\update_glasses.py %*
