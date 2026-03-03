"""
Interactive YAML protocol recorder.

Use the arm2 mamba env, then from version2 run:
  python scripts/record.py

1. Enables manual mode and opens camera.
2. Optionally go to a known location, create a new location, or skip.
3. Loop: (a) record relative tool move by moving arm manually,
   (b) run a command (e.g. z_level(270)) and append to protocol,
   (c) complete and save to protocols/<name>.yaml.
4. After each step, ask for a short description.
"""

import ast
import json
import sys
from pathlib import Path

# Ensure version2 is on path when run as script
if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import numpy as np
import yaml
from aira.utils.paths import get_project_root
from aira.robot import _rpy_deg_to_rotation_matrix

PROTOCOLS_DIR = get_project_root() / "protocols"
LOCATIONS_DIR = get_project_root() / "locations"

# Commands shown in option (b) and their YAML step mapping
COMMAND_HELP = """
Available commands (enter as Python-style call, e.g. z_level(270) or grip(200, speed=1000)):
  go_to(location_name, speed=100, acc=500)
  z_level(height, speed=100, acc=500)
  grip(state, speed=None)       -- gripper position 0=closed, 800=open
  move(dx, dy, dz=0, roll=0, pitch=0, yaw=0)
  sleep(seconds)
  home(speed=100, acc=500)     -- requires ref frame loaded
  load_ref_frame(file)         -- e.g. load_ref_frame("home.json")
"""

ALLOWED_FUNCTIONS = frozenset({
    "go_to", "z_level", "grip", "set_gripper_position",
    "move", "sleep", "home", "load_ref_frame",
})


def _literal_value(node):
    """Extract a literal from ast node (Constant, Num, Str for older Python)."""
    if isinstance(node, ast.Constant):
        return node.value
    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        return node.n
    if hasattr(ast, "Str") and isinstance(node, ast.Str):
        return node.s
    raise ValueError(f"Only literal arguments allowed, got {type(node).__name__}")


def parse_command(line: str) -> tuple:
    """
    Parse a single-line Python call. Returns (func_name, args_list, kwargs_dict).
    Raises ValueError on invalid or disallowed input.
    """
    line = (line or "").strip()
    if not line:
        raise ValueError("Empty input")
    try:
        tree = ast.parse(line, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}")
    if not isinstance(tree.body, ast.Call):
        raise ValueError("Input must be a single function call")
    call = tree.body
    if not isinstance(call.func, ast.Name):
        raise ValueError("Only simple function names allowed")
    name = call.func.id
    if name not in ALLOWED_FUNCTIONS:
        raise ValueError(f"Unknown command '{name}'. Allowed: {sorted(ALLOWED_FUNCTIONS)}")
    args = [_literal_value(a) for a in call.args]
    kwargs = {k.arg: _literal_value(k.value) for k in (call.keywords or [])}
    return name, args, kwargs


def run_command(name: str, args: list, kwargs: dict, a):
    """Execute command via arm() and return None on success, or raise / return non-zero."""
    if name == "go_to":
        loc = args[0] if args else kwargs.get("location", "")
        speed = kwargs.get("speed", 100)
        acc = kwargs.get("acc", 500)
        code = a.go_to(loc, speed=speed, acc=acc, wait=True)
    elif name == "z_level":
        h = args[0] if args else kwargs.get("height", 0)
        speed = kwargs.get("speed", 100)
        acc = kwargs.get("acc", 500)
        code = a.z_level(h, speed=speed, acc=acc, wait=True)
    elif name in ("grip", "set_gripper_position"):
        pos = args[0] if args else kwargs.get("state", kwargs.get("pos", 0))
        speed = kwargs.get("speed")
        code = a.set_gripper_position(float(pos), wait=True, speed=float(speed) if speed is not None else None)
    elif name == "move":
        dx = args[0] if len(args) > 0 else kwargs.get("dx", 0)
        dy = args[1] if len(args) > 1 else kwargs.get("dy", 0)
        dz = args[2] if len(args) > 2 else kwargs.get("dz", 0)
        roll = args[3] if len(args) > 3 else kwargs.get("roll", 0)
        pitch = args[4] if len(args) > 4 else kwargs.get("pitch", 0)
        yaw = args[5] if len(args) > 5 else kwargs.get("yaw", 0)
        code = a.tool_move(float(dx), float(dy), float(dz), float(roll), float(pitch), float(yaw))
    elif name == "sleep":
        secs = args[0] if args else kwargs.get("seconds", 0)
        import time
        time.sleep(float(secs))
        code = 0
    elif name == "home":
        speed = kwargs.get("speed", 100)
        acc = kwargs.get("acc", 500)
        code = a.home(speed=speed, acc=acc, wait=True)
    elif name == "load_ref_frame":
        f = args[0] if args else kwargs.get("file", "home.json")
        a.load_ref_frame(str(f))
        code = 0
    else:
        raise ValueError(f"Unhandled command: {name}")
    if code != 0:
        if a.check_error():
            a.clear_error()
        raise RuntimeError(f"Command returned code {code}")
    return None


def command_to_step(name: str, args: list, kwargs: dict) -> dict:
    """Convert parsed command to protocol step dict (no description yet)."""
    if name == "go_to":
        loc = args[0] if args else kwargs.get("location", "")
        step = {"step": "go_to", "location": str(loc).strip()}
        if kwargs.get("speed") is not None:
            step["speed"] = float(kwargs["speed"])
        if kwargs.get("acc") is not None:
            step["acc"] = float(kwargs["acc"])
        return step
    if name == "z_level":
        h = args[0] if args else kwargs.get("height", 0)
        step = {"step": "z_level", "height": float(h)}
        if kwargs.get("speed") is not None:
            step["speed"] = float(kwargs["speed"])
        if kwargs.get("acc") is not None:
            step["acc"] = float(kwargs["acc"])
        return step
    if name in ("grip", "set_gripper_position"):
        pos = args[0] if args else kwargs.get("state", kwargs.get("pos", 0))
        step = {"step": "grip", "state": float(pos)}
        if kwargs.get("speed") is not None:
            step["speed"] = float(kwargs["speed"])
        return step
    if name == "move":
        dx = args[0] if len(args) > 0 else kwargs.get("dx", 0)
        dy = args[1] if len(args) > 1 else kwargs.get("dy", 0)
        dz = args[2] if len(args) > 2 else kwargs.get("dz", 0)
        roll = args[3] if len(args) > 3 else kwargs.get("roll", 0)
        pitch = args[4] if len(args) > 4 else kwargs.get("pitch", 0)
        yaw = args[5] if len(args) > 5 else kwargs.get("yaw", 0)
        step = {"step": "move", "relative": [float(dx), float(dy), float(dz), float(roll), float(pitch), float(yaw)]}
        return step
    if name == "sleep":
        secs = args[0] if args else kwargs.get("seconds", 0)
        return {"step": "sleep", "seconds": float(secs)}
    if name == "home":
        step = {"step": "home"}
        if kwargs.get("speed") is not None:
            step["speed"] = float(kwargs["speed"])
        if kwargs.get("acc") is not None:
            step["acc"] = float(kwargs["acc"])
        return step
    if name == "load_ref_frame":
        f = args[0] if args else kwargs.get("file", "home.json")
        return {"step": "load_home", "file": str(f).strip()}
    raise ValueError(f"No step mapping for {name}")


def do_startup():
    """Enable manual mode and open camera."""
    from aira.robot import arm, start_vision_display
    start_vision_display()
    a = arm()
    a.set_manual_mode()
    print("Manual mode and camera are ready.")
    return a


def do_location_choice(a):
    """
    Prompt: go to known / create new / skip.
    Returns the chosen location name (for "Go to known") so it can be saved as the first go_to step; otherwise None.
    """
    locations = sorted(LOCATIONS_DIR.glob("*.json"))
    loc_names = [p.stem for p in locations]
    print("\n--- Start position ---")
    print("1) Go to known location")
    print("2) Create new location")
    print("3) Skip")
    choice = input("Choice [1/2/3]: ").strip() or "3"
    if choice == "3":
        return None
    if choice == "1":
        if not loc_names:
            print("No locations in locations/. Skipping.")
            return None
        print("Known locations:", ", ".join(loc_names))
        name = input("Location name: ").strip()
        if not name:
            return None
        if name.endswith(".json"):
            name = name[:-5]
        a.set_position_mode()
        try:
            a.go_to(name, wait=True)
        finally:
            a.set_manual_mode()
        print("Moved to", name, ". Back in manual mode.")
        return name
    if choice == "2":
        name = input("New location name (e.g. start, home): ").strip()
        if not name:
            return None
        if name.endswith(".json"):
            name = name[:-5]
        input("Move the arm to the desired pose, then press Enter...")
        code, pose = a.get_position()
        if code != 0 or len(pose) < 6:
            print("Failed to get position (code=%s). Location not saved." % code)
            return None
        code_j, joints = a.get_joint_angles()
        data = {
            "pose": [float(pose[i]) for i in range(6)],
            "position_mm": [float(pose[0]), float(pose[1]), float(pose[2])],
            "orientation_deg": [float(pose[3]), float(pose[4]), float(pose[5])],
        }
        if code_j == 0 and joints and len(joints) >= 7:
            data["joint_angles_deg"] = [float(j) for j in joints[:7]]
        path = LOCATIONS_DIR / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print("Saved location to", path)
        return name
    print("Unknown choice. Skipping.")
    return None


def do_relative_move(a, steps: list):
    """
    Option (a): use current position as start, then user moves and presses Enter.
    Compute delta in base frame, then convert translation to tool frame via R.T @ delta_base
    so that tool_move (which does R @ delta_tool) recreates the same base-frame motion.
    Rotation delta is the simple difference in RPY (same convention as tool_move adds to current RPY).
    """
    code_b, pose_before = a.get_position()
    if code_b != 0 or len(pose_before) < 6:
        print("Failed to get position (code=%s). Step not added." % code_b)
        return
    input("Move the arm, then press Enter...")
    code_a, pose_after = a.get_position()
    if code_a != 0 or len(pose_after) < 6:
        print("Failed to get position after move (code=%s). Step not added." % code_a)
        return

    # Translation delta in base frame
    delta_base = np.array([
        float(pose_after[0]) - float(pose_before[0]),
        float(pose_after[1]) - float(pose_before[1]),
        float(pose_after[2]) - float(pose_before[2]),
    ], dtype=np.float64)

    # Rotation matrix at start pose (columns = tool axes in base)
    roll_b, pitch_b, yaw_b = float(pose_before[3]), float(pose_before[4]), float(pose_before[5])
    R = _rpy_deg_to_rotation_matrix(roll_b, pitch_b, yaw_b)

    # Convert translation from base frame to tool frame: delta_tool = R.T @ delta_base
    delta_tool = R.T @ delta_base
    dx = round(float(delta_tool[0]), 2)
    dy = round(float(delta_tool[1]), 2)
    dz = round(float(delta_tool[2]), 2)

    # Rotation delta: tool_move simply adds these to current RPY, so delta = after - before
    rx = round(float(pose_after[3]) - roll_b, 2)
    ry = round(float(pose_after[4]) - pitch_b, 2)
    rz = round(float(pose_after[5]) - yaw_b, 2)

    step = {"step": "move", "relative": [dx, dy, dz, rx, ry, rz]}
    steps.append(step)
    desc = input("Description for this step: ").strip()
    if desc:
        step["description"] = desc
    print("Added move step: relative [dx, dy, dz, roll, pitch, yaw] =", step["relative"])


def do_command(a, steps: list):
    """Option (b): list commands, parse input, run, on success append step and ask description."""
    print(COMMAND_HELP)
    while True:
        line = input("Command (or empty to cancel): ").strip()
        if not line:
            return
        try:
            name, args, kwargs = parse_command(line)
        except ValueError as e:
            print("Parse error:", e)
            print("Please fix and try again.")
            continue
        try:
            run_command(name, args, kwargs, a)
        except Exception as e:
            print("Run error:", e)
            print("Please fix and try again.")
            continue
        step = command_to_step(name, args, kwargs)
        steps.append(step)
        desc = input("Description for this step: ").strip()
        if desc:
            step["description"] = desc
        print("Added step:", step.get("step"), step)
        return


def do_joint_move(a, steps: list):
    """
    Option (c): record relative joint move. Get current joint angles (global), user moves arm,
    then record (after - before) as relative joint deltas and save as move_joint step.
    """
    code_b, joints_before = a.get_joint_angles()
    if code_b != 0 or not joints_before or len(joints_before) < 7:
        print("Failed to get joint angles (code=%s). Step not added." % code_b)
        return
    input("Move the arm (joints) to the desired pose, then press Enter...")
    code_a, joints_after = a.get_joint_angles()
    if code_a != 0 or not joints_after or len(joints_after) < 7:
        print("Failed to get joint angles after move (code=%s). Step not added." % code_a)
        return
    relative = [round(float(joints_after[i]) - float(joints_before[i]), 2) for i in range(7)]
    step = {"step": "move_joint", "relative": relative}
    steps.append(step)
    desc = input("Description for this step: ").strip()
    if desc:
        step["description"] = desc
    print("Added move_joint step: relative [j1..j7] =", step["relative"])


def do_save(steps: list):
    """Option (d): prompt filename, write YAML to protocols/."""
    if not steps:
        print("No steps recorded. Nothing to save.")
        return
    name = input("Protocol filename (e.g. my_protocol or steps/rack/mine): ").strip()
    if not name:
        print("No filename given. Not saved.")
        return
    if name.endswith(".yaml") or name.endswith(".yml"):
        name = name[:-5] if name.endswith(".yaml") else name[:-4]
    path = PROTOCOLS_DIR / (name + ".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    brief = input("Brief protocol description (optional): ").strip()
    data = {"brief": brief or "", "protocol": steps}
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print("Saved to", path)


def main():
    a = do_startup()
    start_location = do_location_choice(a)
    steps = []
    if start_location is not None:
        steps.append({"step": "go_to", "location": start_location, "description": "Start at " + start_location})
    while True:
        print("\n--- Step ---")
        print("a) Relative tool move (move arm manually, then record)")
        print("b) Run a command (e.g. z_level(270)) and add to protocol")
        print("c) Relative joint move (move joints manually, then record)")
        print("d) Complete and save to YAML")
        choice = input("Choice [a/b/c/d]: ").strip().lower()
        if choice == "d":
            do_save(steps)
            break
        if choice == "a":
            do_relative_move(a, steps)
        elif choice == "b":
            do_command(a, steps)
        elif choice == "c":
            do_joint_move(a, steps)
        else:
            print("Unknown choice. Use a, b, c, or d.")


if __name__ == "__main__":
    main()
