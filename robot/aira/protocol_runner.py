"""
YAML protocol runner: run robot procedures from YAML with queryable status.

- run_protocol(path): start protocol in background thread (non-blocking).
- join_protocol(timeout): wait for protocol to finish.
- get_protocol_status(): return current step, progress %, state (for MCP).

Step types: load_home, home, go_to, move, move_joint, move_to_object, z_level,
  z_level_object, grip, sleep, run (subprotocol), stop, move_world, move_other.

Steps support an optional ``arm`` field to target a specific arm (e.g. arm: left).
``go_to`` auto-detects from the location file's ``arm`` metadata.
``move_to_object`` also accepts ``camera_arm`` for cross-arm vision.

Protocols can define top-level "args" with defaults. Steps can use "{{arg_name}}".
The "run" step runs another YAML file (relative to protocols/ or current file) with optional args.
"""

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from aira.utils.paths import get_project_root

BASE = get_project_root()
PROTOCOLS_DIR = BASE / "protocols"

# Global status (thread-safe writes)
_status: Dict[str, Any] = {
    "state": "idle",
    "protocol_name": "",
    "current_step_index": 0,
    "current_step_name": "",
    "current_step_description": None,
    "total_steps": 0,
    "progress_pct": 0.0,
    "error": None,
    "started_at": None,
    "finished_at": None,
}
_status_lock = threading.Lock()
_status_changed_event = threading.Event()
_protocol_thread: Optional[threading.Thread] = None
_protocol_stop = threading.Event()


def _load_objects() -> Dict[str, Any]:
    """Load object presets from configs/objects.yaml (includes default_confidence). Single source of truth for yolo_class, shape, confidence, pick_type."""
    path = BASE / "configs" / "objects.yaml"
    if path.exists():
        try:
            import yaml
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {
        "default_confidence": 0.25,
        "50ml eppendorf": {
            "shape": {"type": "circle", "diameter": 33, "location": "center"},
            "yolo_class": "50Ml eppendorf cap",
            "pick_type": "toolhead_close",
        },
    }


def _object_presets_only(objects: Dict[str, Any]) -> Dict[str, Any]:
    """Return only preset entries (exclude default_confidence)."""
    return {k: v for k, v in objects.items() if isinstance(v, dict) and k != "default_confidence"}


def _substitute_args(obj: Any, args: Dict[str, Any]) -> Any:
    """Replace {{key}} in strings with args[key]. Recursively process dicts and lists."""
    if args is None or not args:
        return obj
    if isinstance(obj, str):
        for k, v in args.items():
            obj = obj.replace("{{" + str(k) + "}}", str(v))
        return obj
    if isinstance(obj, dict):
        return {key: _substitute_args(val, args) for key, val in obj.items()}
    if isinstance(obj, list):
        return [_substitute_args(item, args) for item in obj]
    return obj


def _update_status(**kwargs: Any) -> None:
    with _status_lock:
        for k, v in kwargs.items():
            if k in _status:
                _status[k] = v
    _status_changed_event.set()


def _resolve_arm_name(step: Dict[str, Any]) -> Optional[str]:
    """Extract the ``arm`` field from a step dict. None means default arm."""
    val = step.get("arm")
    if val is not None:
        return str(val).strip() or None
    return None


def _run_failure_steps(failure_steps: List[Dict[str, Any]]) -> None:
    """Execute failure block steps (same handlers as protocol steps)."""
    from aira.robot import arm
    for s in failure_steps:
        step_type = (s.get("step") or "").strip().lower()
        if step_type == "stop":
            continue
        arm_name = _resolve_arm_name(s)
        try:
            a = arm(name=arm_name)
            if step_type == "grip":
                pos = s.get("state", 800)
                a.set_gripper_position(float(pos), wait=True)
            elif step_type == "load_home":
                f = s.get("file", "home.json")
                a.load_ref_frame(BASE / f if not Path(f).is_absolute() else f)
            elif step_type == "home":
                a.home()
            elif step_type == "move":
                rel = s.get("relative", [0, 0, 0])
                dx = float(rel[0]) if len(rel) > 0 else 0
                dy = float(rel[1]) if len(rel) > 1 else 0
                dz = float(rel[2]) if len(rel) > 2 else 0
                rx = float(rel[3]) if len(rel) > 3 else 0
                ry = float(rel[4]) if len(rel) > 4 else 0
                rz = float(rel[5]) if len(rel) > 5 else 0
                a.tool_move(dx, dy, dz, rx, ry, rz)
            elif step_type == "move_joint":
                rel = s.get("relative", [0] * 7)
                d_j = [float(rel[i]) if i < len(rel) else 0.0 for i in range(7)]
                a.joint_move(d_j1=d_j[0], d_j2=d_j[1], d_j3=d_j[2], d_j4=d_j[3], d_j5=d_j[4], d_j6=d_j[5], d_j7=d_j[6])
        except Exception:
            pass


def _resolve_protocol_path(file_ref: str, relative_to: Optional[Path] = None) -> Path:
    """Resolve a protocol file path. file_ref can be 'name', 'name.yaml', or 'subfolder/name.yaml'."""
    p = Path(file_ref)
    if not p.suffix or p.suffix.lower() != ".yaml":
        p = Path(str(p) + ".yaml")
    if p.is_absolute() and p.exists():
        return p
    base = relative_to.parent if (relative_to and relative_to.is_file()) else PROTOCOLS_DIR
    candidate = base / p
    if candidate.exists():
        return candidate
    if not candidate.exists() and (PROTOCOLS_DIR / p).exists():
        return PROTOCOLS_DIR / p
    return candidate


def _execute_step(
    step: Dict[str, Any],
    objects: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
    protocol_path: Optional[Path] = None,
) -> None:
    """Execute a single protocol step. Raises on error.

    Steps may include ``arm: left|right`` to target a specific arm and
    ``camera_arm: ...`` for cross-arm vision in move_to_object.

    For ``go_to``, if the location file has ``"arm": "both"`` the step
    moves both arms in parallel (or sequentially with ``parallel: false``).
    """
    import yaml
    from aira.robot import arm, move_to_object, load_location, execute_parallel

    step = _substitute_args(dict(step), args) if args else step
    arm_name = _resolve_arm_name(step)
    a = arm(name=arm_name)
    step_type = (step.get("step") or "").strip().lower()

    # ------------------------------------------------------------------
    if step_type == "load_home":
        f = step.get("file", "home.json")
        path = Path(f)
        if not path.is_absolute():
            path = BASE / path
        a.load_ref_frame(path)

    elif step_type == "home":
        a.home()

    # ------------------------------------------------------------------
    elif step_type == "go_to":
        location = (step.get("location") or step.get("go_to") or "").strip()
        if not location:
            raise ValueError("go_to step requires 'location' or 'go_to'")
        speed = float(step.get("speed", 100))
        acc = float(step.get("acc", 500))

        loc_data = load_location(location)
        loc_arm = loc_data.get("arm")

        if loc_arm == "both" and arm_name is None:
            parallel = step.get("parallel", True)
            arm_names = [k for k in loc_data if k != "arm"]
            if parallel and len(arm_names) > 1:
                tasks = []
                for name in arm_names:
                    tasks.append((
                        lambda n=name: arm(name=n).go_to(location, speed=speed, acc=acc, wait=True),
                        (), {},
                    ))
                results = execute_parallel(tasks)
                for code in results:
                    if code != 0:
                        raise RuntimeError(f"go_to (parallel) returned code {code}")
            else:
                for name in arm_names:
                    code = arm(name=name).go_to(location, speed=speed, acc=acc, wait=True)
                    if code != 0:
                        arm(name=name).clear_error()
                        raise RuntimeError(f"go_to returned code {code} for arm '{name}'")
        else:
            code = a.go_to(location, speed=speed, acc=acc, wait=True)
            if code != 0:
                if a.check_error():
                    a.clear_error()
                raise RuntimeError(f"go_to returned code {code}")

    # ------------------------------------------------------------------
    elif step_type == "z_level":
        height = float(step.get("height") or step.get("z_level", 0))
        code = a.z_level(
            height,
            speed=float(step.get("speed", 100)),
            acc=float(step.get("acc", 500)),
            wait=True,
        )
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"z_level returned code {code}")

    elif step_type == "move":
        rel = step.get("relative", [0, 0, 0])
        dx = float(rel[0]) if len(rel) > 0 else 0
        dy = float(rel[1]) if len(rel) > 1 else 0
        dz = float(rel[2]) if len(rel) > 2 else 0
        rx = float(rel[3]) if len(rel) > 3 else 0
        ry = float(rel[4]) if len(rel) > 4 else 0
        rz = float(rel[5]) if len(rel) > 5 else 0
        code = a.tool_move(dx, dy, dz, rx, ry, rz)
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"tool_move returned code {code}")

    elif step_type == "move_joint":
        rel = step.get("relative", [0] * 7)
        d_j = [float(rel[i]) if i < len(rel) else 0.0 for i in range(7)]
        speed = float(step.get("speed", 100))
        acc = float(step.get("acc", 500))
        code = a.joint_move(
            d_j1=d_j[0], d_j2=d_j[1], d_j3=d_j[2], d_j4=d_j[3],
            d_j5=d_j[4], d_j6=d_j[5], d_j7=d_j[6],
            speed=speed, acc=acc, wait=True,
        )
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"joint_move returned code {code}")

    # ------------------------------------------------------------------
    elif step_type == "move_to_object":
        presets = _object_presets_only(objects)
        obj_name = (step.get("object") or "").strip()
        if not obj_name or obj_name not in presets:
            raise ValueError(f"Unknown object '{obj_name}' (not in configs/objects.yaml)")
        preset = objects[obj_name]
        shape = preset.get("shape", {})
        yolo_class = preset.get("yolo_class")
        conf_threshold = float(preset.get("confidence", objects.get("default_confidence", 0.25)))
        camera_arm = step.get("camera_arm")
        if camera_arm is not None:
            camera_arm = str(camera_arm).strip()
        kwargs: Dict[str, Any] = {
            "shape": shape,
            "yolo_class": yolo_class,
            "pick_type": step.get("pick_type") or preset.get("pick_type", "toolhead_close"),
            "conf_threshold": conf_threshold,
            "average_frames": step.get("average_frames", 5),
            "repeat": step.get("repeat", 3),
            "repeat_skip_mm": float(step.get("repeat_skip_mm", 3.0)),
            "speed": float(step.get("speed", 100)),
            "acc": float(step.get("acc", 500)),
            "display": step.get("display", True),
            "use_robot": True,
            "arm_name": arm_name,
            "camera_arm": camera_arm,
        }
        off = step.get("offset")
        if off is not None and len(off) >= 2:
            kwargs["offset"] = tuple(float(x) for x in off[:3]) if len(off) >= 3 else (float(off[0]), float(off[1]))
        result = move_to_object(**kwargs)
        if not result.get("success"):
            raise RuntimeError(result.get("error", "move_to_object failed"))

    elif step_type == "z_level_object":
        presets = _object_presets_only(objects)
        obj_name = (step.get("object") or "").strip()
        if not obj_name or obj_name not in presets:
            raise ValueError(f"Unknown object '{obj_name}' (not in configs/objects.yaml)")
        preset = objects[obj_name]
        z_offset = float(step.get("z_offset", 10.0))
        average_frames = int(step.get("average_frames", 5))
        pick_type = step.get("pick_type") or preset.get("pick_type", "toolhead_close")
        code = a.z_level_object(
            obj_name,
            z_offset=z_offset,
            average_frames=average_frames,
            pick_type=pick_type,
        )
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"z_level_object returned code {code}")

    # ------------------------------------------------------------------
    elif step_type == "grip":
        pos = step.get("state", 0)
        speed = step.get("speed")
        code = a.set_gripper_position(float(pos), wait=True, speed=float(speed) if speed is not None else None)
        if code != 0:
            raise RuntimeError(f"set_gripper_position returned code {code}")

    elif step_type == "sleep":
        import time
        secs = float(step.get("seconds") or step.get("sleep", 0))
        if secs > 0:
            time.sleep(secs)

    # ------------------------------------------------------------------
    elif step_type == "move_world":
        from aira.coords import world_to_base
        pos = step.get("position")
        if pos is None or len(pos) < 3:
            raise ValueError("move_world requires 'position' [x, y, z]")
        p_world = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
        target_arm = arm_name
        p_base = world_to_base(p_world, target_arm or "default")
        ori = step.get("orientation")
        if ori and len(ori) >= 3:
            roll, pitch, yaw = float(ori[0]), float(ori[1]), float(ori[2])
        else:
            _, cur_pose = a.get_position()
            roll, pitch, yaw = float(cur_pose[3]), float(cur_pose[4]), float(cur_pose[5])
        code = a._ctrl.move_to_absolute(
            x=float(p_base[0]), y=float(p_base[1]), z=float(p_base[2]),
            roll=roll, pitch=pitch, yaw=yaw,
            speed=float(step.get("speed", 100)),
            mvacc=float(step.get("acc", 500)),
            wait=True,
        )
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"move_world returned code {code}")

    elif step_type == "move_other":
        from aira.coords import base_to_base
        pos = step.get("position")
        if pos is None or len(pos) < 3:
            raise ValueError("move_other requires 'position' [x, y, z]")
        ref_arm = step.get("reference_arm")
        if not ref_arm:
            raise ValueError("move_other requires 'reference_arm'")
        ref_arm = str(ref_arm).strip()
        target_arm = arm_name
        p_source = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
        p_target = base_to_base(p_source, from_arm=ref_arm, to_arm=target_arm or "default")
        ori = step.get("orientation")
        if ori and len(ori) >= 3:
            roll, pitch, yaw = float(ori[0]), float(ori[1]), float(ori[2])
        else:
            _, cur_pose = a.get_position()
            roll, pitch, yaw = float(cur_pose[3]), float(cur_pose[4]), float(cur_pose[5])
        code = a._ctrl.move_to_absolute(
            x=float(p_target[0]), y=float(p_target[1]), z=float(p_target[2]),
            roll=roll, pitch=pitch, yaw=yaw,
            speed=float(step.get("speed", 100)),
            mvacc=float(step.get("acc", 500)),
            wait=True,
        )
        if code != 0:
            if a.check_error():
                a.clear_error()
            raise RuntimeError(f"move_other returned code {code}")

    # ------------------------------------------------------------------
    elif step_type == "run":
        file_ref = (step.get("file") or "").strip()
        if not file_ref:
            raise ValueError("run step requires 'file' (e.g. subprotocol/pickuptube.yaml)")
        sub_path = _resolve_protocol_path(file_ref, protocol_path)
        if not sub_path.exists():
            raise FileNotFoundError(f"Subprotocol not found: {sub_path}")
        with open(sub_path, "r") as f:
            sub_data = yaml.safe_load(f) or {}
        sub_steps = sub_data.get("protocol") or []
        file_args = dict(sub_data.get("args") or {})
        run_args = dict(step.get("args") or {})
        merged_args = {**file_args, **run_args}
        run_description = (step.get("description") or sub_path.stem).strip()
        for i, sub_step in enumerate(sub_steps):
            if _protocol_stop.is_set():
                raise RuntimeError("Stopped by user")
            _update_status(current_step_description=f"{run_description} (step {i + 1}/{len(sub_steps)})")
            _execute_step(sub_step, objects, merged_args, sub_path)

    elif step_type == "stop":
        raise RuntimeError(step.get("description") or "Stop step")

    else:
        raise ValueError(f"Unknown step type: {step_type}")


def _protocol_loop(protocol_path: Path) -> None:
    global _status
    import yaml
    with open(protocol_path, "r") as f:
        data = yaml.safe_load(f) or {}
    protocol_steps = data.get("protocol") or []
    failure_steps = data.get("failure") or []
    protocol_name = protocol_path.stem
    total = len(protocol_steps)
    args = dict(data.get("args") or {})
    objects = _load_objects()
    _update_status(
        state="running",
        protocol_name=protocol_name,
        total_steps=total,
        progress_pct=0.0,
        error=None,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
    )
    _protocol_stop.clear()
    try:
        from aira.robot import arm
        arm().set_position_mode()
        for i, step in enumerate(protocol_steps):
            if _protocol_stop.is_set():
                _update_status(
                    state="failed",
                    error="Stopped by user",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                _run_failure_steps(failure_steps)
                stop_protocol()
                try:
                    arm().clear_error()
                    arm().set_position_mode()
                except Exception:
                    pass
                return
            step_type = (step.get("step") or "").strip().lower()
            _update_status(
                current_step_index=i,
                current_step_name=step_type,
                current_step_description=step.get("major_description") or step.get("description"),
                progress_pct=(i / total * 100.0) if total else 0.0,
            )
            if step_type == "stop":
                _update_status(
                    state="failed",
                    error=step.get("description") or "Stop step",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                _run_failure_steps(failure_steps)
                stop_protocol()
                try:
                    arm().clear_error()
                    arm().set_position_mode()
                except Exception:
                    pass
                return
            try:
                _execute_step(step, objects, args, protocol_path)
            except Exception as e:
                _update_status(
                    state="failed",
                    error=str(e),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                _run_failure_steps(failure_steps)
                stop_protocol()
                try:
                    arm().clear_error()
                    arm().set_position_mode()
                except Exception:
                    pass
                return
        _update_status(
            state="finished",
            progress_pct=100.0,
            current_step_index=total,
            current_step_name="",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        _update_status(
            state="failed",
            error=str(e),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        _run_failure_steps(failure_steps)
        stop_protocol()
        try:
            from aira.robot import arm
            arm().clear_error()
            arm().set_position_mode()
        except Exception:
            pass


def run_protocol(path: str | Path) -> bool:
    """
    Start protocol from YAML file in a background thread (non-blocking).
    Path can be 'test', 'vortexing', 'subfolder/pickuptube', or 'test.yaml'.
    Lookup: protocols/<path>.yaml (with subfolders). Returns False if already running, True if started.
    """
    global _protocol_thread
    with _status_lock:
        if _status.get("state") == "running":
            return False
    p = Path(path)
    if not p.suffix or p.suffix.lower() != ".yaml":
        p = Path(str(p) + ".yaml")
    if not p.is_absolute():
        candidate = PROTOCOLS_DIR / p
        if not candidate.exists():
            candidate = PROTOCOLS_DIR / p.name
        p = candidate
    if not p.exists():
        raise FileNotFoundError(f"Protocol not found: {p}")
    _protocol_thread = threading.Thread(target=_protocol_loop, args=(p,), daemon=True)
    _protocol_thread.start()
    return True


def join_protocol(timeout: Optional[float] = None) -> bool:
    """
    Block until the protocol thread finishes or timeout.
    Returns True if finished (state is finished or failed), False if timeout.
    """
    if _protocol_thread is None or not _protocol_thread.is_alive():
        return True
    _protocol_thread.join(timeout=timeout)
    return not _protocol_thread.is_alive()


def get_protocol_status() -> Dict[str, Any]:
    """Return a copy of the current protocol status for MCP or other callers."""
    with _status_lock:
        return dict(_status)


def get_status_changed_event() -> threading.Event:
    """Return the event set whenever status is updated (for MCP to wait on)."""
    return _status_changed_event


def get_protocol_status_formatted() -> str:
    """
    Return a short, LLM-friendly status string: which protocol is running (or "waiting"),
    current step description, and what comes next (or "will finish up soon").
    """
    import yaml
    with _status_lock:
        state = _status.get("state") or "idle"
        protocol_name = (_status.get("protocol_name") or "").strip()
        current_step_index = int(_status.get("current_step_index") or 0)
        total_steps = int(_status.get("total_steps") or 0)
        current_step_description = _status.get("current_step_description")
        error = _status.get("error")

    if state in ("idle", "waiting", ""):
        return "Waiting. No protocol is currently running. You can start a protocol with start_protocol."

    if state == "failed":
        return (
            f"Protocol '{protocol_name}' failed. Error: {error or 'Unknown'}. "
            "No protocol is running now (waiting)."
        )

    if state == "finished":
        return (
            f"Protocol '{protocol_name}' has finished. No protocol is running now (waiting)."
        )

    if state != "running":
        return f"Status: {state}. Protocol: {protocol_name or 'none'} (waiting)."

    # Running: build message with current step and next step
    current_desc = (current_step_description or "running").strip()
    if not current_desc and protocol_name:
        current_desc = f"step {current_step_index + 1} of {total_steps}"

    next_part = "will finish up soon."
    protocol_path = PROTOCOLS_DIR / f"{protocol_name}.yaml"
    if protocol_path.exists() and total_steps > 0:
        try:
            with open(protocol_path, "r") as f:
                data = yaml.safe_load(f) or {}
            steps = data.get("protocol") or []
            if current_step_index + 1 < len(steps):
                next_step = steps[current_step_index + 1]
                next_desc = (next_step.get("description") or next_step.get("step") or "").strip()
                if next_desc:
                    next_part = f"next: {next_desc}."
                else:
                    next_part = f"next: step {current_step_index + 2} of {total_steps}."
        except Exception:
            pass

    return (
        f"Running protocol '{protocol_name}'. "
        f"Current step: {current_desc}. "
        f"{next_part}"
    )


def stop_protocol() -> None:
    """Request the running protocol to stop after the current step. Runs failure block then sets state failed."""
    _protocol_stop.set()


def _protocol_metadata_from_data(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Extract brief and major steps from loaded YAML data. Prefer protocol[].major_description; else step_descriptions or steps."""
    brief = (data.get("brief") or "").strip()
    steps = [str(s.get("major_description")).strip() for s in (data.get("protocol") or []) if s.get("major_description")]
    if not steps:
        steps_raw = data.get("step_descriptions") or data.get("steps") or []
        steps = [str(s).strip() for s in steps_raw if s]
    return {"name": name, "brief": brief, "steps": steps}


def list_protocols() -> List[Dict[str, Any]]:
    """
    Scan protocols/ recursively for *.yaml and return name, brief, and major steps for each.
    Protocol name is the path relative to protocols/ without .yaml (e.g. vortexing, steps/rack/pick_up_from_rack).
    """
    import yaml
    out: List[Dict[str, Any]] = []
    for path in sorted(PROTOCOLS_DIR.rglob("*.yaml")):
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            rel = path.relative_to(PROTOCOLS_DIR)
            name = str(rel.with_suffix("")).replace("\\", "/")
            if "test" in name.lower() or name.startswith("steps/"):
                continue
            out.append(_protocol_metadata_from_data(data, name))
        except Exception:
            continue
    return out


def describe_protocol(protocol_name: str) -> Dict[str, Any]:
    """
    Load a protocol by name and return its brief and major steps only (no low-level step breakdown).
    Uses same path resolution as run_protocol. Raises FileNotFoundError if not found.
    """
    import yaml
    p = Path(protocol_name.strip())
    if not p.suffix or p.suffix.lower() != ".yaml":
        p = Path(str(p) + ".yaml")
    if not p.is_absolute():
        candidate = PROTOCOLS_DIR / p
        if not candidate.exists():
            candidate = PROTOCOLS_DIR / p.name
        p = candidate
    if not p.exists():
        raise FileNotFoundError(f"Protocol not found: {p}")
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    name = str(p.relative_to(PROTOCOLS_DIR).with_suffix("")).replace("\\", "/")
    return _protocol_metadata_from_data(data, name)
