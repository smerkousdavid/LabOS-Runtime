@echo off
REM LabOS Robot Runtime -- standalone launcher (Windows)
REM
REM Usage:
REM   run.bat                          -- reads config from ..\config\config.yaml
REM   run.bat --nat-url ws://host:8002/ws --xarm-ip 192.168.1.185
REM   run.bat --docker                 -- run via docker compose instead

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "CONFIG=%ROOT_DIR%\config\config.yaml"

REM --docker mode
if "%~1"=="--docker" (
    echo Starting robot-runtime via Docker Compose ...
    cd /d "%ROOT_DIR%"
    docker compose -f compose.yaml up -d robot-runtime
    docker compose -f compose.yaml logs -f robot-runtime
    exit /b 0
)

REM Try reading defaults from config.yaml
set "NAT_URL="
set "XARM_IP="
set "SESSION_ID=robot-1"
set "NO_VISION="

python -c "import yaml; cfg=yaml.safe_load(open(r'%CONFIG%')); nat=cfg.get('nat_server',{}).get('url',''); rob=cfg.get('robot',{}); print(f'set NAT_URL={nat}'); print(f'set XARM_IP={rob.get(\"xarm_ip\",\"\")}'); print(f'set SESSION_ID={rob.get(\"session_id\",\"robot-1\")}'); nv='--no-vision' if rob.get('no_vision') else ''; print(f'set NO_VISION={nv}')" > "%TEMP%\robot_cfg.bat" 2>nul
if exist "%TEMP%\robot_cfg.bat" call "%TEMP%\robot_cfg.bat"

echo Starting LabOS Robot Runtime ...
echo   NAT URL:    %NAT_URL%
echo   xArm IP:    %XARM_IP%
echo   Session:    %SESSION_ID%

cd /d "%SCRIPT_DIR%"

set "ARGS="
if defined NAT_URL     set "ARGS=!ARGS! --nat-url !NAT_URL!"
if defined XARM_IP     set "ARGS=!ARGS! --xarm-ip !XARM_IP!"
if defined SESSION_ID  set "ARGS=!ARGS! --session-id !SESSION_ID!"
if defined NO_VISION   set "ARGS=!ARGS! !NO_VISION!"

python robot_runtime.py !ARGS! %*
