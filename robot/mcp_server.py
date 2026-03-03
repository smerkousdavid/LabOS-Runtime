#!/usr/bin/env python3
"""
MCP server for the ARM robot protocol runner (remote or stdio).

Exposes:
  - Tool get_status: current protocol status (waiting / running with step and next step).
  - Tool start_protocol: start a protocol by name (e.g. vortexing).
  - Resource state://protocol: current status (clients can subscribe; server notifies on change).

When you start this server, the OpenCV vision window (camera + YOLO) is shown immediately.
Run remote (default):  python mcp_server.py
Run stdio (local):      python mcp_server.py --stdio

With --stdio, all print() and logging output are suppressed so the MCP JSON-RPC link over
stdin/stdout is not broken by xArm or other library output.

Requires: pip install mcp
"""

from __future__ import annotations

import argparse
import asyncio
import builtins
import logging
import sys
import traceback
from pathlib import Path

# Ensure version2 is on path when run as script
if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

# version2 folder (for log file)
MCP_ROOT = Path(__file__).resolve().parent
MCP_LOG_FILE = MCP_ROOT / "mcp_server.log"

# When True, print() is a no-op and logging is disabled (used for --stdio)
_stdio_quiet = False
_original_print = builtins.print


def _log(message: str, level: str = "INFO", exc_info: bool = False) -> None:
    """Write to the local log file (version2/mcp_server.log). Also to stderr when not stdio quiet and level is WARNING or ERROR."""
    try:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        line = f"{ts} [{level}] mcp_server: {message}\n"
        with open(MCP_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
            if exc_info:
                f.write(traceback.format_exc())
                f.write("\n")
    except Exception:
        pass
    if not _stdio_quiet and level in ("WARNING", "ERROR"):
        _original_print(message, file=sys.stderr)
        if exc_info:
            _original_print(traceback.format_exc(), file=sys.stderr)


def _suppress_output_for_stdio() -> None:
    """Disable all print() and logging output so stdio is only used for MCP JSON-RPC."""
    global _stdio_quiet
    _stdio_quiet = True

    def _noop_print(*args, **kwargs):
        pass

    builtins.print = _noop_print
    logging.disable(logging.CRITICAL)
    # Silence xarm SDK loggers (they may write to stdout)
    for _name in ("xarm", "xarm.core.utils.log", "xarm.wrapper"):
        _log = logging.getLogger(_name)
        _log.setLevel(logging.CRITICAL)
        _log.handlers.clear()
        _log.addHandler(logging.NullHandler())


def _restore_output() -> None:
    """Restore print and logging (e.g. for tests)."""
    global _stdio_quiet
    _stdio_quiet = False
    builtins.print = _original_print
    logging.disable(logging.NOTSET)


def _ensure_vision_display() -> None:
    """Start the camera + YOLO visualization window (non-blocking). Never exits the process on failure."""
    try:
        from aira.robot import start_vision_display
        start_vision_display()
    except BaseException:
        _log("Vision display failed to start (server will continue)", level="WARNING", exc_info=True)
        if not _stdio_quiet:
            _original_print("Vision display failed to start (server will continue):", file=sys.stderr)
            _original_print(traceback.format_exc(), file=sys.stderr)


# Poll interval for status changes (seconds)
_STATUS_POLL_INTERVAL = 1.0
_STREAM_UPDATES_POLL_INTERVAL = 0.35


def _get_status_raw() -> str:
    from aira.protocol_runner import get_protocol_status_formatted
    return get_protocol_status_formatted()


async def _notify_status_changed(mcp, interval: float = _STATUS_POLL_INTERVAL) -> None:
    """Wait for status-change event (or timeout); notify clients when status string changes. Runs indefinitely."""
    from aira import protocol_runner
    last_status = ""
    while True:
        try:
            event = protocol_runner.get_status_changed_event()
            await asyncio.to_thread(event.wait, timeout=interval)
            event.clear()
            current = _get_status_raw()
            if current != last_status:
                last_status = current
                await mcp.notify_resource_changed("state://protocol")
                logging.debug("Protocol status changed; notified clients.")
        except asyncio.CancelledError:
            break
        except Exception as e:
            _log(f"Status poll failed: {e}", level="WARNING")
            if not _stdio_quiet:
                _original_print(f"MCP status poll failed: {e}", file=sys.stderr)


def create_mcp(host: str = "127.0.0.1", port: int = 8000):
    from mcp.server.fastmcp import FastMCP

    # host/port are passed to FastMCP.__init__ (used by streamable-http); run() only takes transport
    mcp = FastMCP(
        "ARM Protocol Server",
        json_response=True,
        host=host,
        port=port,
    )

    @mcp.resource("state://protocol")
    def protocol_status_resource() -> str:
        """Return the current robot protocol status (waiting, or running step and next step)."""
        return _get_status_raw()

    @mcp.tool()
    def get_status() -> str:
        """
        Get the current robot protocol status for the user/LLM.

        Returns a short message: either "Waiting. No protocol is currently running."
        or "Running protocol '<name>'. Current step: <description>. Next: <next step>."
        (or "will finish up soon" if on the last step).
        Use this to tell the user what the robot is doing or that it is idle.
        """
        return _get_status_raw()

    @mcp.tool()
    def start_protocol(protocol_name: str) -> str:
        """
        Start a protocol by name. The protocol runs in the background.

        Args:
            protocol_name: Name of the protocol (e.g. 'vortexing', 'test').
                Can include subpath (e.g. 'subfolder/name'). Looked up under protocols/.

        Returns:
            A message: either "Started protocol '<name>'." or
            "A protocol is already running. Use get_status to see current state."
            If the protocol file is not found, returns an error message.
        """
        from aira.protocol_runner import run_protocol, PROTOCOLS_DIR

        name = (protocol_name or "").strip()
        if not name:
            return "Error: protocol_name is required (e.g. 'vortexing', 'test')."

        try:
            started = run_protocol(name)
        except FileNotFoundError as e:
            return f"Error: Protocol not found. {e}. Protocols dir: {PROTOCOLS_DIR}"

        if not started:
            return (
                "A protocol is already running. Use get_status to see current state."
            )
        return f"Started protocol '{name}'."

    @mcp.tool()
    def get_protocols() -> str:
        """
        List available protocols with brief description and major steps.
        Returns a formatted string suitable for the LLM (name, brief, steps for each protocol).
        """
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

    @mcp.tool()
    def describe_protocol(protocol_name: str) -> str:
        """
        Describe a protocol by name: return its brief and major steps only (no low-level steps).

        Args:
            protocol_name: Name of the protocol (e.g. 'vortexing', 'test').
        """
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

    @mcp.tool()
    def stop_robot() -> str:
        """
        Stop the current protocol and put the robot back in position control mode.
        Use this to stop a running protocol or to disable manual (teaching) mode.
        """
        from aira.protocol_runner import stop_protocol
        from aira.robot import arm
        stop_protocol()
        try:
            a = arm()
            a.set_position_mode()
        except Exception:
            pass
        return "Stopped. Protocol stopped and robot in position control mode."

    @mcp.tool()
    def get_object_definitions() -> str:
        """
        List the exact object names and their definitions/sizes from configs/objects.yaml.
        Use these exact names with move_to_object(object_name). Do not guess or paraphrase (e.g. use 'rack hole' not '4 way rack').
        Returns for each object: name (use this string exactly), shape and size in mm, yolo_class, pick_type.
        """
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
            lines.append(f"- \"{name}\": shape {shape}, yolo_class \"{yolo}\", pick_type {pick}")
        return "Configured objects (use the name in quotes exactly for move_to_object):\n" + "\n".join(lines)

    @mcp.tool()
    def list_objects() -> str:
        """
        List objects visible in the camera frame with their pixel locations, dominant colors, and estimated depth (mm) when available.
        Depth is from RealSense depth camera or estimated from object size and camera intrinsics; omitted if both fail.
        Returns detailed info like: "50ml eppendorf at (320, 240) - orange, 450mm depth"
        Use the pixel location with move_to_object(object_name, target_px_x=320, target_px_y=240) to move to the specific instance.
        Use get_object_definitions() to see the exact object_name strings allowed (e.g. 'rack hole', '50ml eppendorf').
        If vision is not running, returns a message.
        """
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

    @mcp.tool()
    def move_to_object(object_name: str, target_px_x: int = None, target_px_y: int = None) -> str:
        """
        Move robot to the specified object visible in the camera frame.

        Args:
            object_name: Exact object name from configs/objects.yaml (e.g. '50ml eppendorf', 'rack hole', 'vortex genie hole').
                        Call get_object_definitions() to see the exact list of allowed names—do not guess (e.g. use 'rack hole' not '4 way rack').
            target_px_x: Optional X pixel coordinate to target a specific instance. Get from list_objects().
            target_px_y: Optional Y pixel coordinate to target a specific instance. Get from list_objects().

        If target_px_x and target_px_y are provided, moves to the object closest to that pixel location.
        Otherwise, moves to the object closest to the toolhead (toolhead_close pick type).
        First move uses the specified target, subsequent correction moves use toolhead_close.
        """
        from aira.robot import arm
        try:
            a = arm()
            a.set_position_mode()
            pick_type = "toolhead_close"
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

    @mcp.tool()
    def gripper(position: str) -> str:
        """
        Control the gripper to open or close it.

        Args:
            position: One of 'close' (0), 'midway' (300), or 'open' (600).
                      Can also be a number 0-800 (0=fully closed, 800=fully open).
        """
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

    @mcp.tool()
    def z_level(level: str) -> str:
        """
        Move the robot to a predefined height above the ground.

        Args:
            level: One of 'low' (115mm), 'medium' (200mm), or 'high' (300mm).
                   Can also be a number in mm.
        """
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

    @mcp.tool()
    def is_holding_something() -> bool:
        """
        Return True if the gripper is closed (position < 300), i.e. the robot is likely holding something.
        Returns False if gripper is open or if the arm is not connected.
        """
        from aira.robot import arm
        try:
            a = arm()
            code, pos = a.get_gripper_position()
            if code != 0:
                return False
            return float(pos) < 300
        except Exception:
            return False

    @mcp.tool()
    def go_home() -> str:
        """Send the robot to the home position (go_to('home')). Returns success or error message."""
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

    @mcp.tool()
    def see_object(object_name: str) -> bool:
        """
        Return True if the given object is currently visible in the camera frame.

        Args:
            object_name: Object preset name from config (e.g. '50ml eppendorf', 'rack hole').
        """
        from aira.robot import see_object as _see_object
        return _see_object(object_name or "")

    @mcp.tool()
    def manual_mode() -> str:
        """
        Put the robot in manual (teaching) mode so the user can move it by hand.
        Use stop_robot to disable manual mode and return to position control.
        """
        from aira.robot import arm
        try:
            a = arm()
            a.set_manual_mode()
            return "Robot is now in manual mode. You can move it by hand. Use stop_robot to return to position control."
        except Exception as e:
            return f"Error: {e}"

    return mcp


def main() -> int:
    parser = argparse.ArgumentParser(description="ARM Protocol MCP server (remote or stdio).")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Use stdio transport (local). Default is streamable-http (remote).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for remote server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for remote server (default: 8000).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--stream-updates",
        action="store_true",
        help="Poll protocol status more frequently and push updates to subscribers (faster step-change notifications).",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Do not start the camera/vision window (use if running headless or vision causes exit).",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    _log("MCP server starting")

    if args.stdio:
        _suppress_output_for_stdio()

    if not args.no_vision:
        _ensure_vision_display()

    try:
        mcp = create_mcp(host=args.host, port=args.port)

        # Create and set the event loop before mcp.run() uses it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        status_interval = _STREAM_UPDATES_POLL_INTERVAL if args.stream_updates else _STATUS_POLL_INTERVAL
        loop.create_task(_notify_status_changed(mcp, interval=status_interval))

        if args.stdio:
            mcp.run(transport="stdio")
        else:
            _log(f"Starting remote MCP server on {args.host}:{args.port}")
            if not _stdio_quiet:
                _original_print(f"Starting remote MCP server on {args.host}:{args.port}", file=sys.stderr)
            mcp.run(transport="streamable-http")

        return 0
    except BaseException:
        _log("MCP server exited with error", level="ERROR", exc_info=True)
        if not _stdio_quiet:
            _original_print("MCP server exited with error:", file=sys.stderr)
            _original_print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
