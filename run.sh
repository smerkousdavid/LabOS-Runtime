#!/usr/bin/env bash
# LabOS XR Runtime -- Start
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Run install.sh first."
    exit 1
fi

source .venv/bin/activate
python scripts/launcher.py "$@"
