#!/usr/bin/env bash
# LabOS Robot Runtime -- standalone launcher
#
# Usage:
#   ./run.sh                          # reads config from ../config/config.yaml
#   ./run.sh --nat-url ws://host:8002/ws --xarm-ip 192.168.1.185
#   ./run.sh --docker                 # run via docker compose instead
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$ROOT_DIR/config/config.yaml"

# --docker mode: delegate to docker compose
if [[ "${1:-}" == "--docker" ]]; then
    shift
    echo "Starting robot-runtime via Docker Compose ..."
    cd "$ROOT_DIR"
    docker compose -f compose.yaml up -d robot-runtime
    docker compose -f compose.yaml logs -f robot-runtime
    exit 0
fi

# Read defaults from config.yaml if python3 + pyyaml are available
CFG_NAT_URL=""
CFG_XARM_IP=""
CFG_SESSION_ID=""
CFG_NO_VISION=""

if command -v python3 &>/dev/null; then
    eval "$(python3 -c "
import yaml, sys
try:
    cfg = yaml.safe_load(open('$CONFIG'))
    nat = cfg.get('nat_server', {}).get('url', '')
    rob = cfg.get('robot', {})
    print(f'CFG_NAT_URL=\"{nat}\"')
    print(f'CFG_XARM_IP=\"{rob.get(\"xarm_ip\", \"\")}\"')
    print(f'CFG_SESSION_ID=\"{rob.get(\"session_id\", \"robot-1\")}\"')
    if rob.get('no_vision'):
        print('CFG_NO_VISION=\"--no-vision\"')
except Exception as e:
    print(f'# config parse failed: {e}', file=sys.stderr)
" 2>/dev/null || true)"
fi

# Detect which flags the user passed explicitly so we don't override them
USER_ARGS="$*"
has_flag() { [[ "$USER_ARGS" == *"$1"* ]]; }

ARGS=()

if has_flag "--nat-url"; then
    :
elif [[ -n "$CFG_NAT_URL" ]]; then
    ARGS+=(--nat-url "$CFG_NAT_URL")
fi

if has_flag "--xarm-ip"; then
    :
elif [[ -n "$CFG_XARM_IP" ]]; then
    ARGS+=(--xarm-ip "$CFG_XARM_IP")
fi

if has_flag "--session-id"; then
    :
elif [[ -n "$CFG_SESSION_ID" ]]; then
    ARGS+=(--session-id "$CFG_SESSION_ID")
fi

if has_flag "--no-vision"; then
    :
elif [[ -n "$CFG_NO_VISION" ]]; then
    ARGS+=($CFG_NO_VISION)
fi

# Append all user CLI arguments
ARGS+=("$@")

# Resolve final values for display
FINAL_NAT=""; FINAL_XARM=""; FINAL_SID=""
for i in "${!ARGS[@]}"; do
    case "${ARGS[$i]}" in
        --nat-url)    FINAL_NAT="${ARGS[$((i+1))]:-}" ;;
        --xarm-ip)    FINAL_XARM="${ARGS[$((i+1))]:-}" ;;
        --session-id) FINAL_SID="${ARGS[$((i+1))]:-}" ;;
    esac
done

echo "Starting LabOS Robot Runtime ..."
echo "  NAT URL:    ${FINAL_NAT:-<default>}"
echo "  xArm IP:    ${FINAL_XARM:-<default>}"
echo "  Session:    ${FINAL_SID:-robot-1}"

cd "$SCRIPT_DIR"
exec python3 robot_runtime.py "${ARGS[@]}"
