@echo off
setlocal enabledelayedexpansion

:: LabOS XR Runtime -- One-time installer (Windows)
cd /d "%~dp0"

echo.
echo =========================================
echo   LabOS XR Runtime -- Installer
echo =========================================
echo.

:: ── Prerequisites ────────────────────────────────────────────────────────────

echo Checking prerequisites ...

where git >nul 2>&1
if errorlevel 1 (
    echo   [X] git is not installed. Install from https://git-scm.com
    exit /b 1
)
echo   [OK] git

where docker >nul 2>&1
if errorlevel 1 (
    echo   [X] docker is not installed. Install Docker Desktop from https://docker.com
    exit /b 1
)
echo   [OK] docker

docker compose version >nul 2>&1
if errorlevel 1 (
    echo   [X] docker compose not found. Update Docker Desktop.
    exit /b 1
)
echo   [OK] docker compose

set "PYTHON="
for %%P in (python py) do (
    where %%P >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=*" %%V in ('%%P -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "PYVER=%%V"
        for /f "tokens=*" %%M in ('%%P -c "import sys; print(sys.version_info.major)" 2^>nul') do set "PYMAJ=%%M"
        for /f "tokens=*" %%N in ('%%P -c "import sys; print(sys.version_info.minor)" 2^>nul') do set "PYMIN=%%N"
        if !PYMAJ! geq 3 if !PYMIN! geq 10 (
            set "PYTHON=%%P"
            goto :python_found
        )
    )
)
echo   [X] Python 3.10+ is required. Install from https://python.org
exit /b 1

:python_found
echo   [OK] Python !PYVER! (!PYTHON!)
echo.

:: ── Virtual environment ──────────────────────────────────────────────────────

echo Setting up Python virtual environment ...
if not exist ".venv" (
    !PYTHON! -m venv .venv
    echo   [OK] Created .venv\
) else (
    echo   [OK] .venv\ already exists
)

call .venv\Scripts\activate.bat
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo   [OK] Installed Python dependencies
echo.

:: ── Config files ─────────────────────────────────────────────────────────────

echo Setting up configuration files ...

if not exist "config\.env.secrets" (
    if exist "config\.env.secrets.example" (
        copy "config\.env.secrets.example" "config\.env.secrets" >nul
        echo   [!] Created config\.env.secrets -- please fill in your API keys
    ) else (
        type nul > "config\.env.secrets"
        echo   [!] Created empty config\.env.secrets
    )
) else (
    echo   [OK] config\.env.secrets exists
)

if not exist "config\config.yaml" (
    if exist "config\config.yaml.example" (
        copy "config\config.yaml.example" "config\config.yaml" >nul
        echo   [!] Created config\config.yaml -- please review and edit
    )
) else (
    echo   [OK] config\config.yaml exists
)
echo.

:: ── Docker image build ───────────────────────────────────────────────────────

echo Building Docker images ...
if exist "xr_runtime\streaming\Dockerfile" (
    docker build -f xr_runtime\streaming\Dockerfile -t labos_streaming:latest xr_runtime\
    echo   [OK] Built labos_streaming image
) else (
    echo   [!] Streaming Dockerfile not found -- skipping
)
echo.

:: ── Done ─────────────────────────────────────────────────────────────────────

echo Installation complete!
echo.
echo Next steps:
echo   1. Edit config\config.yaml    (set NAT server URL, STT/TTS endpoints)
echo   2. Edit config\.env.secrets    (add API keys if needed)
echo   3. Run: run.bat                (start the runtime)
echo   4. Optional: update_glasses.bat  (configure glasses USB)
echo.

endlocal
