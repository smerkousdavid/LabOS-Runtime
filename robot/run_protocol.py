#!/usr/bin/env python3
"""
Run a protocol by name from version2/protocols/ (supports subfolders).

Usage:
  python run_protocol.py vortexing
  python run_protocol.py protocols/vortexing
  python run_protocol.py vortexing --no-wait    # start and exit (for MCP)
  python run_protocol.py vortexing --vision     # start vision display first
  python run_protocol.py vortexing --status    # print status once and exit

Protocol status (state, current_step_*, progress_pct, error) is available
for MCP or other callers via aira.protocol_runner.get_protocol_status().
"""

import argparse
import sys
from pathlib import Path

# Ensure version2 is on path when run as script
if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from aira.protocol_runner import (
    run_protocol,
    join_protocol,
    get_protocol_status,
    PROTOCOLS_DIR,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a YAML protocol from protocols/ (e.g. vortexing, test).",
    )
    parser.add_argument(
        "protocol",
        nargs="?",
        default="test",
        help="Protocol name or path (e.g. vortexing, subfolder/name). Default: test",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start protocol in background and exit immediately (for MCP).",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Start vision display (camera + YOLO) before running protocol.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current protocol status once and exit (no run).",
    )
    args = parser.parse_args()

    if args.status:
        st = get_protocol_status()
        for k, v in st.items():
            print(f"  {k}: {v}")
        return 0

    if args.vision:
        from aira.robot import start_vision_display
        start_vision_display()

    protocol_name = args.protocol.strip()
    if not protocol_name:
        protocol_name = "test"

    try:
        started = run_protocol(protocol_name)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"Protocols dir: {PROTOCOLS_DIR}", file=sys.stderr)
        return 1

    if not started:
        print("A protocol is already running. Use get_protocol_status() for state.", file=sys.stderr)
        return 2

    print(f"Started protocol: {protocol_name}")

    if args.no_wait:
        print("Running in background. Use aira.protocol_runner.get_protocol_status() for updates.")
        return 0

    print("Waiting for protocol to finish (Ctrl+C to stop)...")
    join_protocol()
    st = get_protocol_status()
    print(f"State: {st.get('state')}")
    if st.get("error"):
        print(f"Error: {st['error']}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
