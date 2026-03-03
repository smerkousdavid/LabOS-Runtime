#!/usr/bin/env python3
"""Robot Runtime -- WebSocket client that connects to the NAT server and
exposes xArm robot tools as callable functions.

The runtime registers its available tools on connect (``robot_register``),
listens for ``robot_execute`` requests from the NAT agent, dispatches to the
matching function, and returns the result via ``robot_result``.

Can run standalone (bare-metal) or inside Docker.

Usage:
    python robot_runtime.py --nat-url ws://labos-nat:8002/ws
    python robot_runtime.py --nat-url ws://labos-nat:8002/ws --xarm-ip 192.168.1.185
    python robot_runtime.py --no-vision   # skip RealSense camera
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger

# Ensure the robot package is importable when running from the robot/ directory
_ROBOT_ROOT = Path(__file__).resolve().parent
if str(_ROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOT_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NAT_URL = os.environ.get("NAT_SERVER_URL", "ws://localhost:8002/ws")
XARM_IP = os.environ.get("XARM_IP", "192.168.1.185")
SESSION_ID = os.environ.get("ROBOT_SESSION_ID", "robot-1")
NO_VISION = os.environ.get("ROBOT_NO_VISION", "false").lower() in ("true", "1", "yes")

_LOG_DIR = Path(os.environ.get("LOG_DIR", str(_ROBOT_ROOT / "logs")))
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(_LOG_DIR / "robot_runtime.log", rotation="20 MB", retention="3 days", level="DEBUG")

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}
_TOOL_FUNCS: Dict[str, Callable[..., Any]] = {}


def _register_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    func: Callable[..., Any],
):
    TOOL_REGISTRY[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
    }
    _TOOL_FUNCS[name] = func


# ---------------------------------------------------------------------------
# Tool implementations (mirror mcp_server.py)
# ---------------------------------------------------------------------------

def _tool_get_status() -> str:
    from aira.protocol_runner import get_protocol_status_formatted
    return get_protocol_status_formatted()


def _tool_start_protocol(protocol_name: str) -> str:
    from aira.protocol_runner import run_protocol, PROTOCOLS_DIR
    name = (protocol_name or "").strip()
    if not name:
        return "Error: protocol_name is required (e.g. 'vortexing', 'test')."
    try:
        started = run_protocol(name)
    except FileNotFoundError as e:
        return f"Error: Protocol not found. {e}. Protocols dir: {PROTOCOLS_DIR}"
    if not started:
        return "A protocol is already running. Use get_status to see current state."
    return f"Started protocol '{name}'."


def _tool_get_protocols() -> str:
    from aira.protocol_runner import list_protocols
    protocols = list_protocols()
    if not protocols:
        return "No protocols found in protocols/."
    lines = []
    for p in protocols:
        name = p.get("name", "")
        brief = p.get("brief", "")
        steps = p.get("steps") or []
        lines.append(f"- {name}: {brief}")
        for s in steps:
            lines.append(f"  - {s}")
    return "\n".join(lines)


def _tool_describe_protocol(protocol_name: str) -> str:
    from aira.protocol_runner import describe_protocol as _describe
    name = (protocol_name or "").strip()
    if not name:
        return "Error: protocol_name is required."
    try:
        meta = _describe(name)
        brief = meta.get("brief", "")
        steps = meta.get("steps") or []
        out = [f"Protocol: {meta.get('name', name)}", f"Brief: {brief}", "Major steps:"]
        for s in steps:
            out.append(f"  - {s}")
        return "\n".join(out)
    except FileNotFoundError as e:
        return f"Error: {e}"


def _tool_stop_robot() -> str:
    from aira.protocol_runner import stop_protocol
    from aira.robot import arm
    stop_protocol()
    try:
        a = arm()
        a.set_position_mode()
    except Exception:
        pass
    return "Stopped. Protocol stopped and robot in position control mode."


def _tool_get_object_definitions() -> str:
    from aira.robot import get_object_definitions as _get_defs
    defs = _get_defs()
    if not defs:
        return "No object definitions in configs/objects.yaml."
    lines = []
    for d in defs:
        name = d.get("name", "?")
        shape = d.get("shape_size_mm", "?")
        yolo = d.get("yolo_class", "")
        pick = d.get("pick_type", "toolhead_close")
        lines.append(f'- "{name}": shape {shape}, yolo_class "{yolo}", pick_type {pick}')
    return "Configured objects (use the name in quotes exactly for move_to_object):\n" + "\n".join(lines)


def _tool_list_objects() -> str:
    from aira.robot import get_latest_detections_detailed
    detections = get_latest_detections_detailed()
    if not detections:
        return "No objects detected (vision may not be running or no objects in view)."
    parts = []
    for d in detections:
        name = d.get("object_name", "unknown")
        cx, cy = d.get("center_px", (0, 0))
        color = d.get("color", "unknown")
        depth_mm = d.get("depth_mm")
        if depth_mm is not None:
            parts.append(f"{name} at ({cx}, {cy}) - {color}, {depth_mm}mm depth")
        else:
            parts.append(f"{name} at ({cx}, {cy}) - {color}")
    return "; ".join(parts)


def _tool_move_to_object(
    object_name: str,
    target_px_x: Optional[int] = None,
    target_px_y: Optional[int] = None,
) -> str:
    from aira.robot import arm
    try:
        a = arm()
        a.set_position_mode()
        pick_type: Any = "toolhead_close"
        if target_px_x is not None and target_px_y is not None:
            pick_type = (float(target_px_x), float(target_px_y))
        result = a.move_to_object(object_name, pick_type=pick_type, display=False)
        if result.get("success"):
            moves = result.get("moves_done", 0)
            return f"Moved to '{object_name}'. {moves} correction move(s) made."
        else:
            return f"Failed to move to '{object_name}': {result.get('error', 'unknown error')}"
    except Exception as e:
        return f"Error: {e}"


def _tool_gripper(position: str) -> str:
    from aira.robot import arm
    positions = {"close": 0, "closed": 0, "midway": 300, "half": 300, "open": 600}
    pos_str = (position or "").strip().lower()
    if pos_str in positions:
        pos = positions[pos_str]
    else:
        try:
            pos = int(float(pos_str))
        except ValueError:
            return f"Invalid position '{position}'. Use 'close', 'midway', 'open', or a number 0-800."
    pos = max(0, min(800, pos))
    try:
        a = arm()
        code = a.set_gripper_position(pos, wait=True)
        if code == 0:
            return f"Gripper moved to position {pos}."
        return f"Gripper command returned code {code}."
    except Exception as e:
        return f"Error: {e}"


def _tool_z_level(level: str) -> str:
    from aira.robot import arm
    levels = {"low": 115, "medium": 200, "med": 200, "high": 300}
    level_str = (level or "").strip().lower()
    if level_str in levels:
        height = levels[level_str]
    else:
        try:
            height = float(level_str)
        except ValueError:
            return f"Invalid level '{level}'. Use 'low', 'medium', 'high', or a number in mm."
    try:
        a = arm()
        a.set_position_mode()
        code = a.z_level(height, wait=True)
        if code == 0:
            return f"Robot moved to z_level {height}mm."
        if a.check_error():
            a.clear_error()
        return f"z_level command returned code {code}."
    except Exception as e:
        return f"Error: {e}"


def _tool_is_holding_something() -> bool:
    from aira.robot import arm
    try:
        a = arm()
        code, pos = a.get_gripper_position()
        if code != 0:
            return False
        return float(pos) < 300
    except Exception:
        return False


def _tool_go_home() -> str:
    from aira.robot import arm
    try:
        a = arm()
        a.set_position_mode()
        code = a.go_to("home")
        if code == 0:
            return "Robot moved to home."
        if a.check_error():
            a.clear_error()
        return f"go_to returned code {code}."
    except Exception as e:
        return f"Error: {e}"


def _tool_see_object(object_name: str) -> bool:
    from aira.robot import see_object as _see_object
    return _see_object(object_name or "")


def _tool_manual_mode() -> str:
    from aira.robot import arm
    try:
        a = arm()
        a.set_manual_mode()
        return "Robot is now in manual mode. You can move it by hand. Use stop_robot to return to position control."
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Register all tools
# ---------------------------------------------------------------------------

def _register_all_tools():
    _register_tool("get_status",
        "Get the current robot protocol status (idle, running step, etc.).",
        {},
        _tool_get_status)

    _register_tool("start_protocol",
        "Start a protocol by name (e.g. 'vortexing', 'test'). Runs in background.",
        {"protocol_name": {"type": "string", "required": True,
                           "description": "Name of the protocol."}},
        _tool_start_protocol)

    _register_tool("get_protocols",
        "List available protocols with descriptions and major steps.",
        {},
        _tool_get_protocols)

    _register_tool("describe_protocol",
        "Describe a protocol by name: brief and major steps.",
        {"protocol_name": {"type": "string", "required": True,
                           "description": "Name of the protocol."}},
        _tool_describe_protocol)

    _register_tool("stop_robot",
        "Stop the current protocol and return to position control mode.",
        {},
        _tool_stop_robot)

    _register_tool("get_object_definitions",
        "List configured object names with shape, YOLO class, and pick type.",
        {},
        _tool_get_object_definitions)

    _register_tool("list_objects",
        "List objects visible in the camera frame with pixel locations, colors, and depth.",
        {},
        _tool_list_objects)

    _register_tool("move_to_object",
        "Move robot to the specified object visible in the camera frame.",
        {"object_name": {"type": "string", "required": True,
                         "description": "Exact object name from configs/objects.yaml."},
         "target_px_x": {"type": "integer", "required": False,
                         "description": "Optional X pixel coordinate."},
         "target_px_y": {"type": "integer", "required": False,
                         "description": "Optional Y pixel coordinate."}},
        _tool_move_to_object)

    _register_tool("gripper",
        "Control the gripper: 'close', 'midway', 'open', or a number 0-800.",
        {"position": {"type": "string", "required": True,
                       "description": "One of 'close', 'midway', 'open', or 0-800."}},
        _tool_gripper)

    _register_tool("z_level",
        "Move robot to a predefined height: 'low' (115mm), 'medium' (200mm), 'high' (300mm), or mm number.",
        {"level": {"type": "string", "required": True,
                   "description": "One of 'low', 'medium', 'high', or a number in mm."}},
        _tool_z_level)

    _register_tool("is_holding_something",
        "Return true if the gripper is closed (position < 300), likely holding something.",
        {},
        _tool_is_holding_something)

    _register_tool("go_home",
        "Send the robot to the home position.",
        {},
        _tool_go_home)

    _register_tool("see_object",
        "Return true if the given object is currently visible in the camera frame.",
        {"object_name": {"type": "string", "required": True,
                         "description": "Object preset name from config."}},
        _tool_see_object)

    _register_tool("manual_mode",
        "Put the robot in manual (teaching) mode so the user can move it by hand.",
        {},
        _tool_manual_mode)


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------

async def _execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Run a registered tool function in a thread executor (blocking-safe)."""
    func = _TOOL_FUNCS.get(tool_name)
    if func is None:
        return {"success": False, "result": f"Unknown tool: {tool_name}"}
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: func(**arguments))
        return {"success": True, "result": result}
    except Exception as exc:
        logger.error(f"[Robot] Tool '{tool_name}' failed: {exc}")
        return {"success": False, "result": f"Error: {exc}"}


async def run_client(nat_url: str, session_id: str):
    """Persistent WebSocket client with auto-reconnect."""
    import websockets

    backoff = 1.0
    url = f"{nat_url}?session_id={session_id}"

    while True:
        try:
            logger.info(f"[Robot] Connecting to {url}")
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                max_size=2 ** 22,
            ) as ws:
                backoff = 1.0
                logger.info(f"[Robot] Connected (session={session_id})")

                # Register available tools
                await ws.send(json.dumps({
                    "type": "robot_register",
                    "session_id": session_id,
                    "tools": list(TOOL_REGISTRY.values()),
                }))
                logger.info(f"[Robot] Registered {len(TOOL_REGISTRY)} tools")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type", "")

                    if msg_type == "robot_execute":
                        request_id = msg.get("request_id", str(uuid.uuid4()))
                        tool_name = msg.get("tool_name", "")
                        arguments = msg.get("arguments", {})
                        logger.info(f"[Robot] Execute: {tool_name}({arguments})")

                        result = await _execute_tool(tool_name, arguments)
                        await ws.send(json.dumps({
                            "type": "robot_result",
                            "request_id": request_id,
                            "tool_name": tool_name,
                            **result,
                        }))
                        logger.info(f"[Robot] Result: {tool_name} -> success={result['success']}")

                    elif msg_type == "pong":
                        pass

                    elif msg_type == "ping":
                        await ws.send(json.dumps({"type": "pong"}))

                    else:
                        logger.debug(f"[Robot] Unhandled message type: {msg_type}")

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning(f"[Robot] WebSocket error: {type(exc).__name__}: {exc}; "
                           f"reconnecting in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


# ---------------------------------------------------------------------------
# Vision bootstrap
# ---------------------------------------------------------------------------

def _ensure_vision():
    """Start the camera + YOLO display thread (non-blocking)."""
    try:
        from aira.robot import start_vision_display
        start_vision_display()
        logger.info("[Robot] Vision display started")
    except Exception:
        logger.warning(f"[Robot] Vision display failed: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LabOS Robot Runtime")
    parser.add_argument("--nat-url", default=NAT_URL,
                        help=f"NAT WebSocket URL (default: {NAT_URL})")
    parser.add_argument("--xarm-ip", default=XARM_IP,
                        help=f"xArm controller IP (default: {XARM_IP})")
    parser.add_argument("--session-id", default=SESSION_ID,
                        help=f"WebSocket session identifier (default: {SESSION_ID})")
    parser.add_argument("--no-vision", action="store_true", default=NO_VISION,
                        help="Skip RealSense camera / vision display")
    parser.add_argument("--log-level", default="INFO",
                        help="Log level (default: INFO)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    logger.add(_LOG_DIR / "robot_runtime.log", rotation="20 MB", retention="3 days", level="DEBUG")

    # Set xArm IP in environment for the aira package to pick up
    os.environ["XARM_IP"] = args.xarm_ip

    _register_all_tools()
    logger.info(f"[Robot] Registered {len(TOOL_REGISTRY)} tools: "
                f"{', '.join(TOOL_REGISTRY.keys())}")

    if not args.no_vision:
        _ensure_vision()

    try:
        asyncio.run(run_client(args.nat_url, args.session_id))
    except KeyboardInterrupt:
        logger.info("[Robot] Shutdown")


if __name__ == "__main__":
    main()
