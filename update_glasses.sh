#!/usr/bin/env bash
# LabOS XR Runtime -- Glasses Configuration
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Run install.sh first."
    exit 1
fi

source .venv/bin/activate
python scripts/update_glasses.py "$@"
