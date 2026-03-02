#!/bin/bash
# =============================================================================
# labos-runtime stop script
#
# Stops all labos-runtime containers, optionally removes volumes.
#
# Usage:
#   ./stop.sh              # Stop all containers
#   ./stop.sh --clean      # Stop and remove volumes (xr_socket, mysql data)
#   ./stop.sh --logs       # Stop and clear log files
#   ./stop.sh --all        # Stop, remove volumes, and clear logs
# =============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR"

CLEAN_VOLUMES=false
CLEAN_LOGS=false

for arg in "$@"; do
    case "$arg" in
        --clean)   CLEAN_VOLUMES=true ;;
        --logs)    CLEAN_LOGS=true ;;
        --all)     CLEAN_VOLUMES=true; CLEAN_LOGS=true ;;
        --help|-h)
            echo "Usage: ./stop.sh [--clean] [--logs] [--all]"
            echo ""
            echo "  --clean   Remove Docker volumes (xr sockets, mysql data)"
            echo "  --logs    Clear all log files in logs/"
            echo "  --all     Both --clean and --logs"
            exit 0
            ;;
    esac
done

echo "============================================"
echo "  labos-runtime -- stopping"
echo "============================================"
echo ""

# ===== SHOW CURRENT STATE =====
if [ -f compose.yaml ]; then
    running=$(docker compose -f compose.yaml ps --status running -q 2>/dev/null | wc -l)
    echo "  Running containers: $running"
else
    echo "  No compose.yaml found (nothing to stop)"
    exit 0
fi
echo ""

# ===== STOP CONTAINERS =====
echo "Stopping all containers..."
if [ "$CLEAN_VOLUMES" = true ]; then
    docker compose -f compose.yaml down --remove-orphans -v 2>&1
else
    docker compose -f compose.yaml down --remove-orphans 2>&1
fi
echo "  Containers stopped."

# ===== CLEAN LOGS =====
if [ "$CLEAN_LOGS" = true ]; then
    echo ""
    echo "Clearing log files..."
    if [ -d logs ]; then
        find logs/ -type f -name "*.log" -delete 2>/dev/null
        find logs/ -type f -name "*.json" -delete 2>/dev/null
        echo "  Log files cleared."
    else
        echo "  No logs/ directory found."
    fi
fi

echo ""
echo "============================================"
echo "  labos-runtime stopped"
echo "============================================"
echo ""
