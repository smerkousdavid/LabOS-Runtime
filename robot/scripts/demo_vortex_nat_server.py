#!/usr/bin/env python3
"""Minimal NAT server demo for the robot runtime.

Hosts a WebSocket server on port 8002.  When the robot-runtime connects:
  1. Waits for ``robot_register`` (tool discovery)
  2. Calls ``list_objects`` and prints what the robot sees
  3. Calls ``start_protocol("vortexing")``
  4. Prints the result and exits

Usage:
    python demo_vortex_nat_server.py              # default :8002
    python demo_vortex_nat_server.py --port 9000  # custom port
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

import websockets

ROBOT_WS: websockets.WebSocketServerProtocol | None = None
TOOLS: dict[str, dict] = {}


async def _call_tool(ws, tool_name: str, arguments: dict | None = None) -> dict:
    """Send robot_execute and wait for the matching robot_result."""
    req_id = str(uuid.uuid4())
    await ws.send(json.dumps({
        "type": "robot_execute",
        "request_id": req_id,
        "tool_name": tool_name,
        "arguments": arguments or {},
    }))
    print(f"  -> Sent: {tool_name}({arguments or {}})")

    while True:
        raw = await ws.recv()
        msg = json.loads(raw)
        if msg.get("type") == "robot_result" and msg.get("request_id") == req_id:
            return msg


async def handler(ws):
    print(f"\n[server] Robot connected from {ws.remote_address}")

    # 1 ── Wait for robot_register
    raw = await ws.recv()
    msg = json.loads(raw)
    if msg.get("type") != "robot_register":
        print(f"[server] Expected robot_register, got: {msg.get('type')}")
        return

    tools = {t["name"]: t for t in msg.get("tools", [])}
    print(f"[server] Registered {len(tools)} tools: {', '.join(tools.keys())}")

    # 2 ── Call list_objects -- show what the robot sees
    print("\n=== What does the robot see? ===")
    result = await _call_tool(ws, "list_objects")
    if result.get("success"):
        print(f"  Result: {result['result']}")
    else:
        print(f"  (failed) {result.get('result', 'unknown error')}")

    # 3 ── Start vortexing protocol
    print("\n=== Starting vortexing protocol ===")
    result = await _call_tool(ws, "start_protocol", {"protocol_name": "vortexing"})
    if result.get("success"):
        print(f"  Result: {result['result']}")
    else:
        print(f"  (failed) {result.get('result', 'unknown error')}")

    # 4 ── Poll status until protocol finishes
    print("\n=== Polling status ===")
    while True:
        await asyncio.sleep(3)
        status = await _call_tool(ws, "get_status")
        text = status.get("result", "")
        print(f"  Status: {text}")
        if "waiting" in text.lower() or "no protocol" in text.lower():
            break

    print("\n[server] Demo complete.")


async def main(port: int):
    print(f"[server] Listening on ws://0.0.0.0:{port}/ws")
    print("[server] Waiting for robot-runtime to connect ...\n")

    async with websockets.serve(handler, "0.0.0.0", port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo NAT server for robot vortexing")
    parser.add_argument("--port", type=int, default=8002, help="WebSocket port (default: 8002)")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.port))
    except KeyboardInterrupt:
        print("\n[server] Stopped.")
