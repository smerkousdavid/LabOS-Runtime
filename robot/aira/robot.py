"""
Robot multi-arm registry and move commands.

Supports one or more xArm robots via configs/robot_mapping.json.
When no mapping exists, falls back to a single "default" arm (backward compatible).

arm()            - default arm (backward compat)
arm("left")      - named arm from robot_mapping.json
arm().go_to(...) - move to saved location
arm().tool_move(dx, dy, ...) - move relative to current tool frame
move_to_object(arm_name=..., camera_arm=...) - vision-guided, optionally cross-arm
execute_parallel(tasks) - run blocking commands on multiple arms concurrently
start_vision_display() - camera + YOLO viewer in background thread
"""

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict, Union, Callable
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Vision display thread: shows camera + YOLO in a separate window from program start.
# Feeds _vision_frame_queue so move_to_object can get frames without pausing the viewer.
_vision_display_thread: Optional[threading.Thread] = None
_vision_display_stop = threading.Event()
_vision_display_last_frame: Optional[np.ndarray] = None
_vision_display_lock = threading.Lock()
_vision_frame_queue: "queue.Queue[Tuple[Any, Any, Any]]" = queue.Queue(maxsize=1)  # (color_image, depth_image, yolo_results)

# Distinct BGR colors per class for bbox drawing (cycle if more classes)
_VISION_CLASS_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 165, 255), (255, 165, 0), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (180, 255, 100), (100, 180, 255),
]

from aira.utils.paths import get_project_root

try:
    import xarm.wrapper as xw
    HAS_XARM = True
except ImportError:
    HAS_XARM = False
    xw = None


class XArmController:
    """xArm controller using tool-frame motions (set_tool_position) and get_pose_offset."""

    def __init__(self, ip: str):
        self.ip = ip
        self.arm = None

    def connect(self) -> bool:
        if not HAS_XARM or xw is None:
            return False
        try:
            logger.info("Connecting to xArm at %s...", self.ip)
            self.arm = xw.XArmAPI(self.ip)
            self.arm.connect()
            self.arm.clean_error()
            self.arm.clean_warn()
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.5)
            logger.info("Connected to xArm successfully")
            return True
        except Exception as e:
            logger.error("Error connecting to xArm: %s", e)
            return False

    def set_manual_mode(self):
        if self.arm:
            self.arm.set_mode(2)
            self.arm.set_state(0)
            time.sleep(0.5)
            logger.info("Robot in MANUAL mode - you can move it by hand")

    def set_position_mode(self):
        if self.arm:
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.5)
            logger.info("Robot in POSITION CONTROL mode")

    def get_position(self) -> Tuple[int, List[float]]:
        """Current TCP pose in base frame: [x, y, z, roll, pitch, yaw] mm/deg."""
        if self.arm:
            code, pos = self.arm.get_position(is_radian=False)
            return code, list(pos) if code == 0 else [0.0] * 6
        return -1, [0.0] * 6

    def move_to_absolute(self, x: float, y: float, z: float,
                         roll: float, pitch: float, yaw: float,
                         speed: float = 50, mvacc: float = 500, wait: bool = True) -> int:
        if self.arm:
            return self.arm.set_position(
                x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                speed=speed, mvacc=mvacc, is_radian=False, wait=wait
            )
        return -1

    def set_tool_position(self, x: float = 0, y: float = 0, z: float = 0,
                           roll: float = 0, pitch: float = 0, yaw: float = 0,
                           speed: float = 50, mvacc: float = 500, wait: bool = True) -> int:
        """Move relative to current tool frame (mm, degrees)."""
        if self.arm:
            return self.arm.set_tool_position(
                x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                speed=speed, mvacc=mvacc, is_radian=False, wait=wait
            )
        return -1

    def get_pose_offset(self, pose1: List[float], pose2: List[float]) -> Tuple[int, List[float]]:
        if self.arm:
            return self.arm.get_pose_offset(pose1, pose2, orient_type_in=0, orient_type_out=0, is_radian=False)
        return -1, [0.0] * 6

    def close_gripper(self) -> int:
        if self.arm:
            self.arm.set_gripper_mode(0)
            self.arm.set_gripper_enable(True)
            self.arm.set_gripper_speed(5000)
            return self.arm.set_gripper_position(0, wait=True)
        return -1

    def get_gripper_position(self) -> Tuple[int, float]:
        """Return (code, position). Position 0 = closed, higher = open (e.g. 800)."""
        if self.arm and hasattr(self.arm, "get_gripper_position"):
            ret = self.arm.get_gripper_position()
            if isinstance(ret, (list, tuple)) and len(ret) >= 2:
                pos = ret[1]
                return int(ret[0]), float(pos) if pos is not None else 0.0
            if isinstance(ret, (list, tuple)) and len(ret) == 1:
                return int(ret[0]), 0.0
        return -1, 0.0

    def check_error(self) -> bool:
        return bool(self.arm and self.arm.has_error)

    def clear_error(self):
        if self.arm:
            self.arm.clean_error()
            self.arm.clean_warn()

    def disconnect(self):
        if self.arm:
            try:
                self.arm.disconnect()
                logger.info("Disconnected from xArm")
            except Exception:
                pass


BASE = get_project_root()

VISION_WINDOW_NAME = "Vision"
VISION_MAX_WIDTH = 1920
VISION_MAX_HEIGHT = 1080


# ---------------------------------------------------------------------------
# Robot mapping config
# ---------------------------------------------------------------------------

def load_robot_mapping() -> Dict[str, Dict[str, Any]]:
    """Load configs/robot_mapping.json. Returns arm-name -> config dict.
    Falls back to a single 'default' entry built from XARM_IP / legacy defaults."""
    path = BASE / "configs" / "robot_mapping.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                mapping = json.load(f)
            if isinstance(mapping, dict) and mapping:
                return mapping
        except Exception:
            pass
    ip = os.environ.get("XARM_IP") or _default_robot_ip()
    return {
        "default": {
            "ip": ip,
            "has_camera": True,
            "camera_device": 0,
            "camera_calibration": "configs/handeye_calibration_result.json",
            "camera_intrinsics": "calibration_images/calibration_matrix.npy",
            "camera_distortion": "calibration_images/distortion_coefficients.npy",
            "handeye_data": "configs/handeye_calibration_data.json",
            "tare": "configs/tare.json",
            "on_linear_rail": False,
        }
    }


class ArmRegistry:
    """Manages multiple XArmController instances keyed by name (e.g. 'left', 'right').
    Arms connect lazily on first access."""

    def __init__(self):
        self._controllers: Dict[str, XArmController] = {}
        self._proxies: Dict[str, ArmProxy] = {}
        self._config: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load_config()

    def _load_config(self) -> None:
        self._config = load_robot_mapping()

    def arm_names(self) -> List[str]:
        return list(self._config.keys())

    def default_name(self) -> str:
        names = self.arm_names()
        if "default" in names:
            return "default"
        return names[0] if names else "default"

    def arm_config(self, name: str) -> Dict[str, Any]:
        if name not in self._config:
            raise KeyError(f"Unknown arm '{name}'. Available: {self.arm_names()}")
        return dict(self._config[name])

    def get(self, arm_name: Optional[str] = None, ip: Optional[str] = None) -> "ArmProxy":
        """Return ArmProxy for the named arm. Connects on first call."""
        if arm_name is None:
            arm_name = self.default_name()
        with self._lock:
            if arm_name in self._proxies:
                return self._proxies[arm_name]
            cfg = self._config.get(arm_name)
            if cfg is None:
                if ip is not None:
                    cfg = {"ip": ip, "has_camera": False}
                    self._config[arm_name] = cfg
                else:
                    raise KeyError(f"Unknown arm '{arm_name}'. Available: {self.arm_names()}")
            arm_ip = ip or cfg.get("ip") or _default_robot_ip()
            if not HAS_XARM:
                raise RuntimeError("xArm SDK not available")
            ctrl = XArmController(arm_ip)
            if not ctrl.connect():
                raise RuntimeError(f"Failed to connect to arm '{arm_name}' at {arm_ip}")
            self._controllers[arm_name] = ctrl
            proxy = ArmProxy(ctrl, name=arm_name)
            self._proxies[arm_name] = proxy
            return proxy

    def reset(self, arm_name: Optional[str] = None) -> None:
        """Disconnect and remove one or all arms."""
        with self._lock:
            names = [arm_name] if arm_name else list(self._controllers.keys())
            for name in names:
                ctrl = self._controllers.pop(name, None)
                self._proxies.pop(name, None)
                if ctrl is not None:
                    try:
                        ctrl.disconnect()
                    except Exception:
                        pass


_registry: Optional[ArmRegistry] = None
_registry_lock = threading.Lock()


def _mode_depth_mm(
    depth_image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color_shape: Tuple[int, ...],
    depth_shape: Tuple[int, ...],
) -> Optional[int]:
    """Mode depth in mm over the detection ROI in depth image. Maps box from color to depth coords. Returns None if no valid depths (filter 0 and negative)."""
    ch, cw = color_shape[:2]
    dh, dw = depth_shape[:2]
    sx = dw / max(cw, 1)
    sy = dh / max(ch, 1)
    d1 = (max(0, int(x1 * sx)), max(0, int(y1 * sy)))
    d2 = (min(dw, int(x2 * sx) + 1), min(dh, int(y2 * sy) + 1))
    if d1[0] >= d2[0] or d1[1] >= d2[1]:
        return None
    roi = depth_image[d1[1]:d2[1], d1[0]:d2[0]]
    valid = roi[(roi > 0)]
    if valid.size == 0:
        return None
    # Mode: most frequent value (RealSense z16 is in mm)
    valid_int = valid.astype(np.int32)
    uniq, counts = np.unique(valid_int, return_counts=True)
    return int(uniq[counts.argmax()])


def _estimate_detection_depth_mm(
    depth_image: Optional[np.ndarray],
    color_shape: Tuple[int, ...],
    depth_shape: Tuple[int, ...],
    box_xyxy: Tuple[float, float, float, float],
    preset_shape: Optional[Dict[str, Any]],
    K: Optional[np.ndarray],
    T_cam_to_tool: Optional[np.ndarray],
) -> Optional[int]:
    """
    Estimate depth in mm for a detection in **tool frame** (depth relative to tool head).
    (1) RealSense mode depth over ROI -> 3D camera -> transform to tool -> tool Z.
    (2) Else geometry from object shape + intrinsics -> 3D camera -> transform to tool -> tool Z.
    Returns None if both fail or transform unavailable. Do not use placeholder like '???' — omit depth when None.
    """
    from aira.vision.vision import object_point_3d_camera, camera_to_tool

    if T_cam_to_tool is None:
        return None

    def _camera_to_tool_z(p_cam_mm: np.ndarray) -> Optional[int]:
        pt = camera_to_tool(p_cam_mm, T_cam_to_tool)
        if np.isfinite(pt).all() and pt[2] > 0:
            return int(round(float(pt[2])))
        return None

    # 1) Direct from RealSense depth (mode over ROI) -> 3D in camera frame -> tool frame Z
    if depth_image is not None and depth_shape[0] > 0 and depth_shape[1] > 0 and K is not None:
        x1, y1, x2, y2 = box_xyxy
        mode_mm = _mode_depth_mm(
            depth_image, int(x1), int(y1), int(x2), int(y2), color_shape, depth_shape,
        )
        if mode_mm is not None and mode_mm > 0:
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            x_cam = (u - cx) * mode_mm / fx
            y_cam = (v - cy) * mode_mm / fy
            p_cam = np.array([x_cam, y_cam, float(mode_mm)], dtype=np.float64)
            out = _camera_to_tool_z(p_cam)
            if out is not None:
                return out
    # 2) Geometry: known object size + camera intrinsics -> already 3D camera -> tool frame Z
    if preset_shape and K is not None:
        try:
            p_cam = object_point_3d_camera(box_xyxy, preset_shape, K)
            if np.isfinite(p_cam).all() and p_cam[2] > 0:
                return _camera_to_tool_z(p_cam)
        except Exception:
            pass
    return None


def _vision_display_loop() -> None:
    """Background thread: show camera + YOLO detections and depth (if available). Feeds _vision_frame_queue so move_to_object can get frames without pausing."""
    global _vision_display_last_frame
    from aira.vision.singletons import camera, yolo

    # Placeholder so the window is never gray before first frame
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    placeholder[:] = (40, 40, 40)
    cv2.putText(placeholder, "Loading camera", (120, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    cv2.namedWindow(VISION_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(VISION_WINDOW_NAME, placeholder)
    cv2.waitKey(1)

    try:
        cam = camera()
        model = yolo()
    except Exception as e:
        logger.warning("Vision display: could not init camera/yolo: %s", e)
        cv2.putText(placeholder, f"Error: {e}", (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(VISION_WINDOW_NAME, placeholder)
        cv2.waitKey(1)
        return

    last_display: Optional[np.ndarray] = placeholder.copy()
    conf = 0.25
    get_frames = getattr(cam, "get_frames", None)

    while not _vision_display_stop.is_set():
        if get_frames is not None:
            color_image, depth_image = get_frames()
            ok = color_image is not None
        else:
            ok, color_image = cam.read()
            depth_image = None
        if not ok or color_image is None:
            cv2.imshow(VISION_WINDOW_NAME, last_display)
            cv2.waitKey(30)
            continue

        results = model.predict(color_image, conf=conf, imgsz=640, verbose=False)
        disp = color_image.copy()
        color_shape = color_image.shape
        depth_shape = (depth_image.shape[:2] if depth_image is not None else (0, 0))
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(boxes.cls[i])
                color = _VISION_CLASS_COLORS[cls_id % len(_VISION_CLASS_COLORS)]
                names = getattr(model, "names", {})
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                conf_val = float(boxes.conf[i])
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 3)
                cv2.putText(disp, f"{label} {conf_val:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        try:
            _vision_frame_queue.put_nowait((color_image, depth_image, results))
        except queue.Full:
            try:
                _vision_frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                _vision_frame_queue.put_nowait((color_image, depth_image, results))
            except queue.Full:
                pass
        cv2.putText(disp, "Vision (q in move_to_object to quit centering)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if depth_image is not None:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )
            # Draw detections on depth map; label = estimated depth (mm) from RealSense or geometry — omit if both fail
            if results and len(results) > 0 and results[0].boxes is not None:
                from aira.vision.vision import resolve_class_to_index
                objects = _load_objects_for_robot()
                presets = _object_presets_only(objects)
                try:
                    from aira.vision.singletons import calibration
                    cal = calibration()
                    K = cal.get("K")
                    T_cam_to_tool = cal.get("T_cam_to_tool")
                except Exception:
                    K = None
                    T_cam_to_tool = None
                names = getattr(model, "names", {})
                classes = [names[i] for i in sorted(names.keys())] if isinstance(names, dict) else []
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    cls_id = int(boxes.cls[i])
                    preset_shape = None
                    for _on, preset in presets.items():
                        if resolve_class_to_index(classes, preset.get("yolo_class")) == cls_id:
                            preset_shape = preset.get("shape")
                            break
                    depth_mm = _estimate_detection_depth_mm(
                        depth_image, color_shape, depth_shape, (x1, y1, x2, y2), preset_shape, K, T_cam_to_tool,
                    )
                    # Box in depth image coords
                    dh, dw = depth_shape[0], depth_shape[1]
                    ch, cw = color_shape[0], color_shape[1]
                    sx, sy = dw / max(cw, 1), dh / max(ch, 1)
                    dx1 = max(0, int(x1 * sx))
                    dy1 = max(0, int(y1 * sy))
                    dx2 = min(dw, int(x2 * sx) + 1)
                    dy2 = min(dh, int(y2 * sy) + 1)
                    color = _VISION_CLASS_COLORS[cls_id % len(_VISION_CLASS_COLORS)]
                    cv2.rectangle(depth_colormap, (dx1, dy1), (dx2, dy2), color, 3)
                    if depth_mm is not None:
                        cv2.putText(depth_colormap, f"{depth_mm}mm", (dx1, dy1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # Resize depth for display: same height as color, width scaled to preserve aspect
            d_h, d_w = depth_colormap.shape[:2]
            disp_h, disp_w = disp.shape[:2]
            if (d_h, d_w) != (disp_h, disp_w):
                scale_h = disp_h / max(d_h, 1)
                new_d_w = int(d_w * scale_h)
                depth_colormap = cv2.resize(
                    depth_colormap, (new_d_w, disp_h), interpolation=cv2.INTER_AREA,
                )
            combined = np.hstack((disp, depth_colormap))
        else:
            combined = disp

        # Scale combined to fit within max size, preserving aspect ratio (one dimension may be smaller)
        h, w = combined.shape[:2]
        scale = min(VISION_MAX_WIDTH / w, VISION_MAX_HEIGHT / h)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)

        with _vision_display_lock:
            _vision_display_last_frame = combined
        last_display = combined
        cv2.imshow(VISION_WINDOW_NAME, combined)
        cv2.waitKey(30)

    try:
        cv2.destroyWindow(VISION_WINDOW_NAME)
    except Exception:
        pass


def start_vision_display() -> None:
    """Start the camera + YOLO viewer in a background thread. Call at program start to always show the vision window."""
    global _vision_display_thread
    if _vision_display_thread is not None and _vision_display_thread.is_alive():
        return
    _vision_display_stop.clear()
    _vision_display_thread = threading.Thread(target=_vision_display_loop, daemon=True)
    _vision_display_thread.start()


def _is_vision_display_running() -> bool:
    return _vision_display_thread is not None and _vision_display_thread.is_alive()


def _default_robot_ip() -> str:
    """Load robot IP from handeye_calibration_data.json if present."""
    path = BASE / "handeye_calibration_data.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            ip = data.get("metadata", {}).get("robot_ip")
            if ip:
                return str(ip)
        except Exception:
            pass
    return "192.168.1.195"


def load_location(location_name: str) -> Dict[str, Any]:
    """Load a location JSON file by name (without .json suffix) from locations/.
    Returns the full dict including any 'arm' metadata."""
    name = str(location_name).strip()
    if not name:
        raise ValueError("location_name cannot be empty")
    if not name.endswith(".json"):
        name = name + ".json"
    path = BASE / "locations" / name
    if not path.exists():
        raise FileNotFoundError(f"Location file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _get_registry() -> ArmRegistry:
    """Return (or create) the global ArmRegistry singleton."""
    global _registry
    if _registry is not None:
        return _registry
    with _registry_lock:
        if _registry is not None:
            return _registry
        _registry = ArmRegistry()
        return _registry


def arm(name: Optional[str] = None, ip: Optional[str] = None) -> "ArmProxy":
    """Return ArmProxy for the named arm. Backward compatible: arm() returns the default arm.
    arm("left") returns the left arm, etc."""
    return _get_registry().get(arm_name=name, ip=ip)


def get_arm_config(name: Optional[str] = None) -> Dict[str, Any]:
    """Return the robot_mapping.json config for the given arm (or default)."""
    reg = _get_registry()
    return reg.arm_config(name or reg.default_name())


def get_arm_names() -> List[str]:
    """Return all configured arm names."""
    return _get_registry().arm_names()


def reset_arm(name: Optional[str] = None) -> None:
    """Disconnect and clear one arm (by name) or all arms (name=None)."""
    if _registry is not None:
        _registry.reset(name)


def execute_parallel(
    tasks: List[Tuple[Callable, tuple, dict]],
) -> List[Any]:
    """Run blocking callables in parallel threads, join, collect results.
    tasks: list of (callable, args, kwargs). Each callable typically wraps an
    arm command like ``lambda: arm('left').go_to('home')``.
    Raises the first error encountered (after all threads have joined)."""
    results: List[Any] = [None] * len(tasks)
    errors: List[Optional[BaseException]] = [None] * len(tasks)
    threads: List[threading.Thread] = []
    for i, (fn, args, kwargs) in enumerate(tasks):
        def _worker(idx: int = i, _fn: Callable = fn,
                    _args: tuple = args, _kw: dict = kwargs) -> None:
            try:
                results[idx] = _fn(*_args, **_kw)
            except Exception as exc:
                errors[idx] = exc
        t = threading.Thread(target=_worker)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    for err in errors:
        if err is not None:
            raise err
    return results


def _rpy_deg_to_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Rotation matrix from RPY in degrees (xArm order: roll=X, pitch=Y, yaw=Z). R = Rz(yaw)*Ry(pitch)*Rx(roll); columns = tool axes in base."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    cx, sx = np.cos(r), np.sin(r)
    cy, sy = np.cos(p), np.sin(p)
    cz, sz = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _rotation_matrix_to_rpy_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract roll, pitch, yaw in degrees from 3x3 R (xArm: R = Rz*Ry*Rx). Handles gimbal lock."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")
    eps = 1e-6
    if abs(R[2, 0]) >= 1.0 - eps:
        pitch_deg = 90.0 if R[2, 0] < 0 else -90.0
        roll_deg = np.degrees(np.arctan2(R[0, 1], R[0, 2])) if R[2, 0] < 0 else np.degrees(np.arctan2(-R[0, 1], -R[0, 2]))
        yaw_deg = 0.0
        return float(roll_deg), float(pitch_deg), float(yaw_deg)
    pitch_deg = float(np.degrees(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0))))
    cp = np.cos(np.radians(pitch_deg))
    roll_deg = float(np.degrees(np.arctan2(R[2, 1] / cp, R[2, 2] / cp)))
    yaw_deg = float(np.degrees(np.arctan2(R[1, 0] / cp, R[0, 0] / cp)))
    return roll_deg, pitch_deg, yaw_deg


class ArmProxy:
    """Thin wrapper so arm().tool_move(...), arm().home(), arm().load_ref_frame(...) work; also exposes .arm for raw API."""

    def __init__(self, controller: XArmController, name: str = "default"):
        self._ctrl = controller
        self.name = name
        self.arm = controller.arm  # raw xArm API if needed
        self._ref_frame: Optional[Dict[str, Any]] = None  # loaded home pose for relative moves

    def set_manual_mode(self) -> None:
        """Put robot in manual (teaching) mode so it can be moved by hand."""
        self._ctrl.set_manual_mode()

    def set_position_mode(self) -> None:
        """Put robot in position control mode for normal moves."""
        self._ctrl.set_position_mode()

    def load_ref_frame(self, path: Union[str, Path]) -> None:
        """
        Load reference frame (e.g. home.json). When loaded, tool_move(dx,dy,dz) is
        interpreted in this frame and home() moves to this pose. If joint_angles_deg
        is present, home() uses joint-space motion for a stable trajectory.
        """
        p = Path(path)
        if not p.is_absolute():
            p = BASE / p
        with open(p, "r") as f:
            data = json.load(f)
        pose = data.get("pose")
        if pose is None:
            pos = data.get("position_mm", [])
            ori = data.get("orientation_deg", [])
            if len(pos) >= 3 and len(ori) >= 3:
                pose = [float(pos[0]), float(pos[1]), float(pos[2]),
                        float(ori[0]), float(ori[1]), float(ori[2])]
        if not pose or len(pose) < 6:
            raise ValueError("JSON must contain 'pose' (6 floats) or 'position_mm' + 'orientation_deg'")
        self._ref_frame = {
            "pose": [float(pose[i]) for i in range(6)],
            "position": np.array([float(pose[0]), float(pose[1]), float(pose[2])]),
            "rpy_deg": (float(pose[3]), float(pose[4]), float(pose[5])),
        }
        joints = data.get("joint_angles_deg")
        if joints and len(joints) >= 7:
            self._ref_frame["joint_angles_deg"] = [float(j) for j in joints[:7]]

    def clear_ref_frame(self) -> None:
        """Stop using reference frame. tool_move is always relative to current tool frame."""
        self._ref_frame = None

    def go_to(
        self,
        location_name: str,
        speed: float = 250,
        acc: float = 600,
        wait: bool = True,
    ) -> int:
        """
        Move to a saved location by name. Looks for locations/<name>.json under project root.
        Uses joint angles when present for a stable path, otherwise cartesian pose.
        For bimanual locations ("arm": "both"), extracts this arm's sub-entry.
        """
        data = load_location(location_name)
        arm_field = data.get("arm")
        if arm_field == "both":
            sub = data.get(self.name)
            if sub is None:
                raise ValueError(
                    f"Bimanual location '{location_name}' has no entry for arm '{self.name}'"
                )
            data = sub
        return self._go_to_data(data, speed=speed, acc=acc, wait=wait)

    def _go_to_data(
        self,
        data: Dict[str, Any],
        speed: float = 250,
        acc: float = 600,
        wait: bool = True,
    ) -> int:
        """Move to a pose described by a location dict (pose + optional joint_angles_deg)."""
        pose = data.get("pose")
        if pose is None:
            pos = data.get("position_mm", [])
            ori = data.get("orientation_deg", [])
            if len(pos) >= 3 and len(ori) >= 3:
                pose = [float(pos[0]), float(pos[1]), float(pos[2]),
                        float(ori[0]), float(ori[1]), float(ori[2])]
        if not pose or len(pose) < 6:
            raise ValueError("Location data must contain 'pose' (6 floats) or 'position_mm' + 'orientation_deg'")
        joints = data.get("joint_angles_deg")
        if joints and len(joints) >= 7:
            return self.arm.set_servo_angle(
                angle=joints,
                is_radian=False,
                speed=speed,
                mvacc=acc,
                wait=wait,
            )
        x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
        r, p, yaw = float(pose[3]), float(pose[4]), float(pose[5])
        return self._ctrl.move_to_absolute(
            x=x, y=y, z=z, roll=r, pitch=p, yaw=yaw,
            speed=speed, mvacc=acc, wait=wait,
        )

    def home(
        self,
        speed: float = 100,
        acc: float = 500,
        wait: bool = True,
    ) -> int:
        """Move to the loaded reference (home) pose. Uses joint angles when present for a stable path."""
        if self._ref_frame is None:
            raise RuntimeError("No reference frame loaded; call load_ref_frame('home.json') first")
        joints = self._ref_frame.get("joint_angles_deg")
        if joints and len(joints) >= 7:
            return self.arm.set_servo_angle(
                angle=joints,
                is_radian=False,
                speed=speed,
                mvacc=acc,
                wait=wait,
            )
        x, y, z = self._ref_frame["position"]
        r, p, yaw = self._ref_frame["rpy_deg"]
        return self._ctrl.move_to_absolute(
            x=x, y=y, z=z, roll=r, pitch=p, yaw=yaw,
            speed=speed, mvacc=acc, wait=wait,
        )

    def tool_move(
        self,
        dx: float = 0,
        dy: float = 0,
        dz: float = 0,
        roll: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        speed: float = 250,
        acc: float = 600,
        wait: bool = True,
    ) -> int:
        """
        Move toolhead relative to the current tool frame: (dx, dy, dz) in current tool axes,
        (roll, pitch, yaw) relative to current orientation. Unchanged by load_ref_frame (ref is only for home() etc).
        """
        code, pose = self._ctrl.get_position()
        if code != 0:
            return code
        curr_pos = np.array(pose[:3], dtype=np.float64)
        curr_rpy = (float(pose[3]), float(pose[4]), float(pose[5]))
        R = _rpy_deg_to_rotation_matrix(*curr_rpy)
        delta = np.array([dx, dy, dz], dtype=np.float64)
        target_pos = curr_pos + R @ delta
        target_r = curr_rpy[0] + roll
        target_p = curr_rpy[1] + pitch
        target_y = curr_rpy[2] + yaw
        return self._ctrl.move_to_absolute(
            x=float(target_pos[0]), y=float(target_pos[1]), z=float(target_pos[2]),
            roll=target_r, pitch=target_p, yaw=target_y,
            speed=speed, mvacc=acc, wait=wait,
        )

    def tool_z_move(
        self,
        height_mm_above_table: float,
        speed: float = 250,
        acc: float = 600,
        wait: bool = True,
    ) -> int:
        """
        Move toolhead to a base-frame Z = z0 + height_mm_above_table (mm).
        z0 is from handeye_calibration_data.json metadata.z0_reference (table height).
        X, Y, roll, pitch, yaw remain at current position.
        """
        from aira.vision.singletons import calibration
        cal = calibration()
        z0_ref = cal.get("z0_reference")
        if not z0_ref or len(z0_ref) < 3:
            raise RuntimeError("z0_reference not found in handeye_calibration_data.json")
        z0_z = float(z0_ref[2])
        target_z = z0_z + height_mm_above_table
        code, pos = self._ctrl.get_position()
        if code != 0:
            return code
        x, y, _, roll, pitch, yaw = pos
        return self._ctrl.move_to_absolute(
            x=x, y=y, z=target_z,
            roll=roll, pitch=pitch, yaw=yaw,
            speed=speed, mvacc=acc, wait=wait,
        )

    def z_level(
        self,
        height: float = 100.0,
        speed: float = 200,
        acc: float = 600,
        wait: bool = True,
    ) -> int:
        """
        Set tool height above the reference ground plane (z0 from handeye_calibration_data.json).
        TCP Z = z0 + height (mm). X, Y, roll, pitch, yaw remain at current position.
        """
        from aira.vision.singletons import calibration
        cal = calibration()
        z0_ref = cal.get("z0_reference")
        if not z0_ref or len(z0_ref) < 3:
            raise RuntimeError("z0_reference not found in handeye_calibration_data.json")
        z0_z = float(z0_ref[2])
        target_z = z0_z + height
        code, pos = self._ctrl.get_position()
        if code != 0:
            return code
        x, y, _, roll, pitch, yaw = pos
        return self._ctrl.move_to_absolute(
            x=x, y=y, z=target_z,
            roll=roll, pitch=pitch, yaw=yaw,
            speed=speed, mvacc=acc, wait=wait,
        )

    def z_down(
        self,
        speed: float = 200,
        acc: float = 600,
        wait: bool = True,
    ) -> int:
        """
        Orient the toolhead so the tool Z-axis aligns with base Z (pointing down).
        Rotation is done in the tool frame using only Rx and Ry (no Rz), so global
        Z ends up with zero angle and the tool XY plane aligns with the base XY plane.
        Keeps current (x, y, z).
        """
        code, pos = self._ctrl.get_position()
        if code != 0:
            return code
        x, y, z, roll_deg, pitch_deg, yaw_deg = pos
        R = _rpy_deg_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg)
        # Desired tool Z in base = down. So R_new[:,2] = [0,0,-1].
        # Apply rotation in tool frame: R_new = R @ R_delta, with R_delta = Rx(tx) @ Ry(ty).
        # We need R_delta @ e3 = R^T @ [0,0,-1] = v.
        v = (R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)).ravel()
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        ty = np.arcsin(np.clip(vx, -1.0, 1.0))
        cos_ty = np.cos(ty)
        if abs(cos_ty) < 1e-6:
            tx = 0.0
        else:
            tx = np.arctan2(-vy, vz)
        # Rx(tx) @ Ry(ty) in radians
        cx, sx = np.cos(tx), np.sin(tx)
        cy, sy = np.cos(ty), np.sin(ty)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
        R_delta = Rx @ Ry
        R_new = R @ R_delta
        roll_new, pitch_new, yaw_new = _rotation_matrix_to_rpy_deg(R_new)
        return self._ctrl.move_to_absolute(
            x=x, y=y, z=z,
            roll=roll_new, pitch=pitch_new, yaw=yaw_new,
            speed=speed, mvacc=acc, wait=wait,
        )

    def z_level_object(
        self,
        object_name: str,
        z_offset: float = 10.0,
        average_frames: int = 5,
        pick_type: str = "toolhead_close",
        speed: float = 200,
        acc: float = 600,
        wait: bool = True,
    ) -> int:
        """
        Set tool height so the TCP is z_offset mm above the object's Z level.
        Detects the object by name (from objects.yaml), gets its 3D position in tool frame,
        then moves so base Z = object_base_z + z_offset. Keeps X, Y, orientation.
        """
        p_tool = get_object_position_tool_frame(
            object_name,
            average_frames=average_frames,
            pick_type=pick_type,
        )
        if p_tool is None:
            raise RuntimeError(f"Object '{object_name}' not detected")
        code, pos = self._ctrl.get_position()
        if code != 0:
            return code
        x, y, z, roll, pitch, yaw = pos
        R = _rpy_deg_to_rotation_matrix(roll, pitch, yaw)
        p_tool_arr = np.array([p_tool[0], p_tool[1], p_tool[2]], dtype=np.float64)
        object_base_z = z + (R @ p_tool_arr)[2]
        target_z = object_base_z + z_offset
        return self._ctrl.move_to_absolute(
            x=x, y=y, z=target_z,
            roll=roll, pitch=pitch, yaw=yaw,
            speed=speed, mvacc=acc, wait=wait,
        )

    def move_to_object(
        self,
        object_name: str,
        offset: Optional[Tuple[float, ...]] = None,
        pick_type: Optional[str] = None,
        average_frames: int = 5,
        repeat: int = 3,
        repeat_skip_mm: float = 3.0,
        speed: float = 200,
        acc: float = 600,
        display: bool = True,
        ignore_z: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Move toolhead to the named object (from objects.yaml). E.g. a.move_to_object('50ml eppendorf', offset=(50, 0)).
        Forwards to robot.move_to_object with shape and yolo_class from the object preset.
        ignore_z: if True (default), only move in x,y; no up/down. Set False to use offset z.
        """
        objects = _load_objects_for_robot()
        presets = _object_presets_only(objects)
        if object_name not in presets:
            raise ValueError(f"Unknown object '{object_name}' (not in configs/objects.yaml)")
        preset = objects[object_name]
        shape = preset.get("shape", {})
        yolo_class = preset.get("yolo_class")
        conf_threshold = _conf_for_preset(objects, preset)
        resolved_pick_type = pick_type if pick_type is not None else preset.get("pick_type", "toolhead_close")
        return move_to_object(
            shape=shape,
            yolo_class=yolo_class,
            pick_type=resolved_pick_type,
            conf_threshold=conf_threshold,
            average_frames=average_frames,
            repeat=repeat,
            repeat_skip_mm=repeat_skip_mm,
            speed=speed,
            acc=acc,
            display=display,
            use_robot=True,
            offset=tuple(offset) if offset is not None else None,
            ignore_z=ignore_z,
            **kwargs,
        )

    def get_position(self) -> Tuple[int, List[float]]:
        return self._ctrl.get_position()

    def get_joint_angles(self) -> Tuple[int, List[float]]:
        """Get current joint angles in degrees. Returns (code, [j1, j2, ..., j7])."""
        if not hasattr(self.arm, "get_servo_angle"):
            return -1, []
        code, angles = self.arm.get_servo_angle(is_radian=False)
        return code, list(angles) if code == 0 else []

    def joint_move(
        self,
        d_j1: float = 0,
        d_j2: float = 0,
        d_j3: float = 0,
        d_j4: float = 0,
        d_j5: float = 0,
        d_j6: float = 0,
        d_j7: float = 0,
        speed: float = 100,
        acc: float = 500,
        wait: bool = True,
    ) -> int:
        """
        Move joints by relative angles (degrees). Uses current joint angles + deltas.
        Pass deltas for each joint (j1..j7); omitted joints default to 0.
        """
        code, current = self.get_joint_angles()
        if code != 0 or not current or len(current) < 7:
            return code if code != 0 else -1
        deltas = [d_j1, d_j2, d_j3, d_j4, d_j5, d_j6, d_j7]
        target = [float(current[i]) + float(deltas[i]) for i in range(7)]
        return self.arm.set_servo_angle(
            angle=target,
            is_radian=False,
            speed=speed,
            mvacc=acc,
            wait=wait,
        )

    def get_pose_offset(
        self, pose1: List[float], pose2: List[float]
    ) -> Tuple[int, List[float]]:
        """
        Return pose offset from pose1 to pose2 (same convention as controller/firmware).
        Returns (code, [dx, dy, dz, droll, dpitch, dyaw]) with offset in pose1's tool frame.
        Use (dx, dy, dz) for tool_move / move step.
        """
        return self._ctrl.get_pose_offset(pose1, pose2)

    def set_tool_position(self, x: float = 0, y: float = 0, z: float = 0,
                          roll: float = 0, pitch: float = 0, yaw: float = 0,
                          speed: float = 50, mvacc: float = 500, wait: bool = True) -> int:
        return self._ctrl.set_tool_position(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
            speed=speed, mvacc=mvacc, wait=wait,
        )

    def check_error(self) -> bool:
        return self._ctrl.check_error()

    def clear_error(self):
        return self._ctrl.clear_error()

    def disconnect(self):
        return self._ctrl.disconnect()

    def set_gripper_position(self, pos: float, wait: bool = True, speed: Optional[float] = None, **kwargs) -> int:
        """Set gripper position (0 closed, 800 open typical). Delegates to raw arm API."""
        if hasattr(self.arm, "set_gripper_position"):
            return self.arm.set_gripper_position(pos, wait=wait, speed=speed, **kwargs)
        return -1

    def get_gripper_position(self) -> Tuple[int, float]:
        """Return (code, position). Position 0 = closed, higher = open (e.g. 800)."""
        return self._ctrl.get_gripper_position()


def arm_proxy(name: Optional[str] = None, ip: Optional[str] = None) -> ArmProxy:
    """Return arm wrapped as ArmProxy (backward-compat alias for arm())."""
    return arm(name=name, ip=ip)


def _load_objects_for_robot() -> Dict[str, Any]:
    """Load object presets from configs/objects.yaml. Returns full dict (default_confidence + presets). Single source of truth for yolo_class, shape, confidence, pick_type."""
    path = BASE / "configs" / "objects.yaml"
    if path.exists():
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
                return data
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


def get_object_definitions() -> List[Dict[str, Any]]:
    """
    Return object definitions from configs/objects.yaml for move_to_object and protocols.
    Each entry has: name (exact string to use), shape (type + sizes in mm), yolo_class, confidence, pick_type.
    Use these exact names when calling move_to_object(object_name).
    """
    raw = _load_objects_for_robot()
    presets = _object_presets_only(raw)
    default_conf = raw.get("default_confidence", 0.2)
    out = []
    for name, cfg in presets.items():
        if not isinstance(cfg, dict):
            continue
        shape = cfg.get("shape") or {}
        shape_type = shape.get("type", "unknown")
        size_desc: str
        if shape_type == "circle":
            size_desc = f"diameter {shape.get('diameter', '?')} mm"
        elif shape_type == "square":
            size_desc = f"side {shape.get('side', '?')} mm"
        elif shape_type == "rect":
            size_desc = f"width {shape.get('width', '?')} x height {shape.get('height', '?')} mm"
        else:
            size_desc = str(shape)
        out.append({
            "name": name,
            "shape_type": shape_type,
            "shape_size_mm": size_desc,
            "location": shape.get("location", "center"),
            "yolo_class": cfg.get("yolo_class", ""),
            "confidence": cfg.get("confidence", default_conf),
            "pick_type": cfg.get("pick_type", "toolhead_close"),
        })
    return out


def _get_dominant_color(image: np.ndarray, bbox: Tuple[float, float, float, float]) -> str:
    """
    Extract dominant color from bbox ROI and return binned color name.
    Uses HSV color space with hue bins for color classification.
    Returns one of: red, orange, yellow, green, cyan, blue, purple, pink, brown, white, gray, black.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_vals = hsv[:, :, 0].flatten()
    s_vals = hsv[:, :, 1].flatten()
    v_vals = hsv[:, :, 2].flatten()
    avg_s = float(np.mean(s_vals))
    avg_v = float(np.mean(v_vals))
    if avg_s < 40:
        if avg_v < 50:
            return "black"
        elif avg_v > 200:
            return "white"
        else:
            return "gray"
    avg_h = float(np.mean(h_vals))
    if avg_h < 10 or avg_h >= 170:
        return "red"
    elif avg_h < 22:
        return "orange"
    elif avg_h < 35:
        return "yellow"
    elif avg_h < 78:
        return "green"
    elif avg_h < 100:
        return "cyan"
    elif avg_h < 130:
        return "blue"
    elif avg_h < 150:
        return "purple"
    else:
        return "pink"


def get_latest_detections_detailed() -> List[Dict[str, Any]]:
    """
    Get detailed info for each detected object in the latest vision frame:
    - object_name: preset name (e.g. '50ml eppendorf')
    - center_px: (cx, cy) bbox center in pixels
    - color: dominant color name in ROI
    - conf: confidence
    - depth_mm: estimated depth in mm (only present when available from RealSense or geometry)
    Returns [] if vision not running or queue empty.
    """
    from aira.vision.vision import resolve_class_to_index
    from aira.vision.dataset import get_class_names
    from aira.vision.singletons import yolo, calibration

    try:
        item = _vision_frame_queue.get_nowait()
    except Exception:
        return []
    # Queue item is (color_image, depth_image, results)
    if len(item) == 2:
        color_image, results = item
        depth_image = None
    else:
        color_image, depth_image, results = item
    try:
        objects = _load_objects_for_robot()
        presets = _object_presets_only(objects)
        model = yolo()
        classes = getattr(model, "names", None)
        if classes is not None and isinstance(classes, dict):
            classes = [classes[i] for i in sorted(classes.keys())]
        if classes is None:
            classes = get_class_names() or []
        try:
            cal = calibration()
            K = cal.get("K")
            T_cam_to_tool = cal.get("T_cam_to_tool")
        except Exception:
            K = None
            T_cam_to_tool = None
        color_shape = color_image.shape
        depth_shape = (depth_image.shape[:2] if depth_image is not None else (0, 0))
        detections: List[Dict[str, Any]] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_idx = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                for obj_name, preset in presets.items():
                    yolo_class = preset.get("yolo_class")
                    if resolve_class_to_index(classes, yolo_class) == cls_idx:
                        color = _get_dominant_color(color_image, (x1, y1, x2, y2))
                        preset_shape = preset.get("shape")
                        depth_mm = _estimate_detection_depth_mm(
                            depth_image, color_shape, depth_shape, (x1, y1, x2, y2), preset_shape, K, T_cam_to_tool,
                        )
                        d = {
                            "object_name": obj_name,
                            "center_px": (int(cx), int(cy)),
                            "color": color,
                            "conf": round(conf, 2),
                        }
                        if depth_mm is not None:
                            d["depth_mm"] = depth_mm
                        detections.append(d)
                        break
        return detections
    finally:
        try:
            if depth_image is not None:
                _vision_frame_queue.put_nowait((color_image, depth_image, results))
            else:
                _vision_frame_queue.put_nowait((color_image, None, results))
        except Exception:
            pass


def get_latest_detection_counts() -> Dict[str, int]:
    """
    Get counts of each configured object visible in the latest vision frame.
    Uses _vision_frame_queue (get_nowait, then put back). Returns {} if vision not running or queue empty.
    """
    from aira.vision.vision import resolve_class_to_index
    from aira.vision.dataset import get_class_names
    from aira.vision.singletons import yolo

    try:
        item = _vision_frame_queue.get_nowait()
    except Exception:
        return {}
    if len(item) == 2:
        color_image, results = item
        depth_image = None
    else:
        color_image, depth_image, results = item
    try:
        objects = _load_objects_for_robot()
        presets = _object_presets_only(objects)
        model = yolo()
        classes = getattr(model, "names", None)
        if classes is not None and isinstance(classes, dict):
            classes = [classes[i] for i in sorted(classes.keys())]
        if classes is None:
            classes = get_class_names() or []
        counts: Dict[str, int] = {name: 0 for name in presets}
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_idx = int(boxes.cls[i])
                for obj_name, preset in presets.items():
                    yolo_class = preset.get("yolo_class")
                    if resolve_class_to_index(classes, yolo_class) == cls_idx:
                        counts[obj_name] = counts.get(obj_name, 0) + 1
                        break
        return counts
    finally:
        try:
            _vision_frame_queue.put_nowait((color_image, depth_image, results))
        except Exception:
            pass


def see_object(object_name: str) -> bool:
    """Return True if the given object (e.g. '50ml eppendorf') is visible in the latest frame."""
    counts = get_latest_detection_counts()
    return counts.get(object_name.strip(), 0) >= 1


def _conf_for_preset(objects: Dict[str, Any], preset: Dict[str, Any]) -> float:
    """Confidence threshold for a preset (per-object or default)."""
    return float(preset.get("confidence", objects.get("default_confidence", 0.25)))


def get_object_position_tool_frame(
    object_name: str,
    average_frames: int = 5,
    pick_type: str = "toolhead_close",
    tare_mm: Optional[Tuple[float, float, float]] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Detect object by name and return its 3D position (x, y, z) in tool frame (mm),
    averaged over average_frames. Returns None if not detected.
    """
    from aira.vision.vision import (
        parse_shape,
        object_point_3d_camera,
        pick_detection,
        resolve_class_to_index,
        camera_to_tool,
    )
    from aira.vision.dataset import get_class_names
    from aira.vision.singletons import camera, yolo, calibration

    objects = _load_objects_for_robot()
    presets = _object_presets_only(objects)
    if object_name not in presets:
        return None
    preset = objects[object_name]
    shape = preset.get("shape", {})
    yolo_class = preset.get("yolo_class")
    conf_threshold = _conf_for_preset(objects, preset)

    cal = calibration()
    T_cam_to_tool = cal["T_cam_to_tool"]
    K = cal["K"]
    tare_arr = np.array(tare_mm if tare_mm is not None else cal["tare_mm"], dtype=np.float64)
    shape_norm = parse_shape(shape)

    model = yolo()
    classes = getattr(model, "names", None)
    if classes is not None and isinstance(classes, dict):
        classes = [classes[i] for i in sorted(classes.keys())]
    if classes is None:
        classes = get_class_names() or [
            "50ml eppendorf tube", "50Ml eppendorf cap", "50Ml 4 way rack",
        ]
    cls_idx = resolve_class_to_index(classes, yolo_class)

    cam = camera()
    if cam is None:
        return None

    buffer: List[Tuple[float, float, float]] = []
    for _ in range(average_frames):
        ok, color_image = cam.read()
        if not ok or color_image is None:
            continue
        results = model.predict(color_image, conf=conf_threshold, imgsz=640, verbose=False)
        detections: List[Dict[str, Any]] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                if int(boxes.cls[i]) != cls_idx:
                    continue
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(float, box)
                p_cam = object_point_3d_camera((x1, y1, x2, y2), shape_norm, K)
                if not np.isfinite(p_cam).all():
                    continue
                pt = camera_to_tool(p_cam, T_cam_to_tool) + tare_arr
                detections.append({
                    "bbox_xyxy": (x1, y1, x2, y2),
                    "class_id": cls_idx,
                    "conf": float(boxes.conf[i]),
                    "p_tool_mm": pt,
                })
        chosen = pick_detection(detections, pick_type, color_image.shape, T_cam_to_tool, tare_arr)
        if chosen is not None and chosen.get("p_tool_mm") is not None:
            p = chosen["p_tool_mm"]
            buffer.append((float(p[0]), float(p[1]), float(p[2])))
    if not buffer:
        return None
    return (
        float(np.mean([b[0] for b in buffer])),
        float(np.mean([b[1] for b in buffer])),
        float(np.mean([b[2] for b in buffer])),
    )


def move_to_object(
    shape: Dict[str, Any],
    pick_type: Union[str, Tuple[float, float]] = "toolhead_close",
    yolo_class: Optional[Union[str, int]] = None,
    conf_threshold: float = 0.25,
    average_frames: int = 5,
    repeat: int = 3,
    repeat_skip_mm: float = 3.0,
    speed: float = 250,
    acc: float = 600,
    display: bool = True,
    use_camera_singleton: bool = True,
    use_robot: bool = True,
    robot_ip: Optional[str] = None,
    tare_mm: Optional[Tuple[float, float, float]] = None,
    offset: Optional[Tuple[float, ...]] = None,
    ignore_z: bool = True,
    arm_name: Optional[str] = None,
    camera_arm: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Move toolhead to the selected object using camera, yolo, calibration and arm().

    arm_name: which arm to move (default: default arm).
    camera_arm: which arm's camera to use for detection (default: same as arm_name).
        When camera_arm != arm_name, the cross-arm pipeline is used:
        camera -> EE -> camera_arm base -> world -> arm_name base -> absolute move.

    offset: (dx, dy) or (dx, dy, dz) in mm added to object position.
    ignore_z: if True (default), only move in x,y; dz is forced to 0.
    """
    from aira.vision.vision import (
        parse_shape,
        object_point_3d_camera,
        pick_detection,
        resolve_class_to_index,
        camera_to_tool,
    )
    from aira.vision.dataset import get_class_names
    from aira.vision.singletons import camera, yolo, calibration

    effective_camera_arm = camera_arm or arm_name
    effective_arm_name = arm_name

    # Validate camera availability
    if effective_camera_arm is not None:
        try:
            cfg = get_arm_config(effective_camera_arm)
            if not cfg.get("has_camera", True):
                return {
                    "success": False, "final_xy_tool_mm": None, "moves_done": 0,
                    "error": f"Camera not available for arm '{effective_camera_arm}'",
                }
        except KeyError:
            pass

    cross_arm = (effective_camera_arm is not None
                 and effective_arm_name is not None
                 and effective_camera_arm != effective_arm_name)

    cal = calibration(arm_name=effective_camera_arm)
    T_cam_to_tool = cal["T_cam_to_tool"]
    K = cal["K"]
    tare_arr = np.array(tare_mm if tare_mm is not None else cal["tare_mm"], dtype=np.float64)
    shape_norm = parse_shape(shape)

    model = yolo()
    classes = getattr(model, "names", None)
    if classes is not None and isinstance(classes, dict):
        classes = [classes[i] for i in sorted(classes.keys())]
    if classes is None:
        classes = get_class_names() or [
            "Vortex Genie 2", "Vortex Genie Hole", "Vortex Genie Top Plate",
            "50ml eppendorf tube", "50Ml eppendorf cap", "50Ml 4 way rack",
            "4 way rack 50ml hole", "4 way rack 5ml hole",
        ]
    cls_idx = resolve_class_to_index(classes, yolo_class)

    cam = camera(arm_name=effective_camera_arm) if use_camera_singleton else None
    if cam is None:
        return {"success": False, "final_xy_tool_mm": None, "moves_done": 0, "error": "camera not available"}

    robot = arm(name=effective_arm_name, ip=robot_ip) if use_robot else None

    use_global_viewer = False
    if display:
        start_vision_display()
        use_global_viewer = _is_vision_display_running()
        if not use_global_viewer:
            cv2.namedWindow("Center on Object", cv2.WINDOW_AUTOSIZE)

    moves_done = 0
    final_xy_tool_mm = None
    user_quit = False

    try:
        for move_idx in range(repeat):
            if user_quit:
                break
            current_pick_type = pick_type if move_idx == 0 else "toolhead_close"
            buffer_xy: List[Tuple[float, float]] = []
            buffer_base: List[np.ndarray] = []
            if use_global_viewer:
                while True:
                    try:
                        _vision_frame_queue.get_nowait()
                    except queue.Empty:
                        break
            for _ in range(average_frames):
                if user_quit:
                    break
                if use_global_viewer:
                    try:
                        item = _vision_frame_queue.get(timeout=1.0)
                        color_image = item[0]
                        results = item[2] if len(item) >= 3 else item[1]
                    except queue.Empty:
                        continue
                else:
                    ok, color_image = cam.read()
                    if not ok or color_image is None:
                        continue
                    results = model.predict(color_image, conf=conf_threshold, imgsz=640, verbose=False)
                detections: List[Dict[str, Any]] = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        if int(boxes.cls[i]) != cls_idx:
                            continue
                        if use_global_viewer and float(boxes.conf[i]) < conf_threshold:
                            continue
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(float, box)
                        p_cam = object_point_3d_camera((x1, y1, x2, y2), shape_norm, K)
                        if not np.isfinite(p_cam).all():
                            continue
                        pt = camera_to_tool(p_cam, T_cam_to_tool) + tare_arr
                        detections.append({
                            "bbox_xyxy": (x1, y1, x2, y2),
                            "class_id": cls_idx,
                            "conf": float(boxes.conf[i]),
                            "p_tool_mm": pt,
                        })
                chosen = pick_detection(detections, current_pick_type, color_image.shape, T_cam_to_tool, tare_arr)
                if chosen is not None and chosen.get("p_tool_mm") is not None:
                    p = chosen["p_tool_mm"]
                    if cross_arm:
                        buffer_base.append(np.array([p[0], p[1], p[2]], dtype=np.float64))
                    else:
                        buffer_xy.append((float(p[0]), float(p[1])))
                if display and not use_global_viewer and color_image is not None:
                    disp = color_image.copy()
                    for d in detections:
                        b = d["bbox_xyxy"]
                        is_chosen = chosen is d if chosen else False
                        color = (0, 255, 0) if is_chosen else (128, 128, 128)
                        cv2.rectangle(disp, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
                        if is_chosen and chosen.get("p_tool_mm") is not None:
                            pt = chosen["p_tool_mm"]
                            cv2.putText(disp, f"Tool: [{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}] mm",
                                        (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(disp, f"Move {move_idx + 1}/{repeat} | buffer {len(buffer_xy) or len(buffer_base)}/{average_frames} | conf={conf_threshold:.2f} | q=quit", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Center on Object", disp)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        user_quit = True
                        break

            if cross_arm and buffer_base:
                from aira.coords import camera_to_other_base
                avg_tool = np.mean(np.array(buffer_base), axis=0)
                _, cam_pose = arm(name=effective_camera_arm).get_position()
                p_target = camera_to_other_base(
                    avg_tool, from_arm=effective_camera_arm, to_arm=effective_arm_name,
                    ee_pose=cam_pose, T_cam_to_ee=T_cam_to_tool, tare=tuple(tare_arr),
                    from_rail_pos=None, to_rail_pos=None,
                )
                if robot is not None:
                    code, cur_pose = robot.get_position()
                    if code == 0:
                        code = robot._ctrl.move_to_absolute(
                            x=float(p_target[0]), y=float(p_target[1]),
                            z=float(cur_pose[2]) if ignore_z else float(p_target[2]),
                            roll=float(cur_pose[3]), pitch=float(cur_pose[4]), yaw=float(cur_pose[5]),
                            speed=speed, mvacc=acc, wait=True,
                        )
                        if code == 0:
                            moves_done += 1
                        elif robot.check_error():
                            robot.clear_error()
                final_xy_tool_mm = (float(p_target[0]), float(p_target[1]))
                continue

            if not buffer_xy:
                if display and not use_global_viewer:
                    ok, last_frame = cam.read()
                    if last_frame is not None:
                        cv2.putText(last_frame, "No detection", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow("Center on Object", last_frame)
                    if cv2.waitKey(100) & 0xFF == ord("q"):
                        user_quit = True
                continue
            avg_dx = float(np.mean([b[0] for b in buffer_xy]))
            avg_dy = float(np.mean([b[1] for b in buffer_xy]))
            off_x = float(offset[0]) if offset and len(offset) > 0 else 0.0
            off_y = float(offset[1]) if offset and len(offset) > 1 else 0.0
            off_z = float(offset[2]) if offset and len(offset) > 2 else 0.0
            move_dx = avg_dx + off_x
            move_dy = avg_dy + off_y
            move_dz = 0.0 if ignore_z else off_z
            final_xy_tool_mm = (move_dx, move_dy)
            dist = np.sqrt(move_dx ** 2 + move_dy ** 2)
            if dist < repeat_skip_mm:
                if display and not use_global_viewer:
                    ok, skip_frame = cam.read()
                    if skip_frame is not None:
                        cv2.putText(skip_frame, f"Skipped (within {repeat_skip_mm}mm)", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.imshow("Center on Object", skip_frame)
                    if cv2.waitKey(100) & 0xFF == ord("q"):
                        user_quit = True
                continue
            if robot is not None:
                code = robot.tool_move(move_dx, move_dy, move_dz, 0, 0, 0, speed=speed, acc=acc, wait=True)
                if code == 0:
                    moves_done += 1
                else:
                    if robot.check_error():
                        robot.clear_error()
    finally:
        if display and not use_global_viewer:
            try:
                cv2.destroyWindow("Center on Object")
            except Exception:
                pass

    return {"success": True, "final_xy_tool_mm": final_xy_tool_mm, "moves_done": moves_done}


