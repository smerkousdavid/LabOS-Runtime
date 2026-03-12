"""
Arm-aware singletons for camera, YOLO model, and hand-eye calibration.

camera(arm_name)      -- per-arm camera (RealSense or OpenCV)
calibration(arm_name) -- per-arm hand-eye calibration, intrinsics, tare
yolo()                -- shared YOLO model (not arm-specific)

When arm_name is None the "default" (or only) arm's resources are returned,
preserving backward compatibility with single-arm setups.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from aira.utils.paths import get_project_root

BASE = get_project_root()
CALIBRATION_IMAGE_WIDTH = 1280
CALIBRATION_IMAGE_HEIGHT = 720
CONFIGS_PATH = BASE / "configs"
BASE_MODEL = BASE / "weights" / "segmentv7.pt"

# ---------------------------------------------------------------------------
# Camera -- keyed by arm name
# ---------------------------------------------------------------------------

_camera_singletons: Dict[str, Any] = {}
_camera_default_use_cv: Optional[bool] = None
_camera_default_device: int = 0


class _CvCamera:
    def __init__(self, cap: Any):
        self._cap = cap

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        return ret, np.asarray(frame) if ret and frame is not None else None

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        if ret and frame is not None:
            return np.asarray(frame), None
        return None, None

    def stop(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


class _RealSenseCamera:
    def __init__(self, pipeline: Any, align: Any):
        self._pipeline = pipeline
        self._align = align

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self._align.process(frames)
            cf = aligned.get_color_frame()
            if not cf:
                return False, None
            return True, np.asanyarray(cf.get_data())
        except Exception:
            return False, None

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self._align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf:
                return None, None
            color_image = np.asanyarray(cf.get_data())
            depth_image = np.asanyarray(df.get_data()) if df else None
            return color_image, depth_image
        except Exception:
            return None, None

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


def _resolve_arm_name(arm_name: Optional[str]) -> str:
    """Map None -> 'default' (or the first arm in robot_mapping.json)."""
    if arm_name is not None:
        return arm_name
    try:
        from aira.robot import get_arm_names
        names = get_arm_names()
        return names[0] if names else "default"
    except Exception:
        return "default"


def _arm_config(arm_name: str) -> Dict[str, Any]:
    """Return robot_mapping config for arm_name, or sensible defaults."""
    try:
        from aira.robot import load_robot_mapping
        mapping = load_robot_mapping()
        if arm_name in mapping:
            return mapping[arm_name]
        if mapping:
            return next(iter(mapping.values()))
    except Exception:
        pass
    return {
        "has_camera": True,
        "camera_device": 0,
        "camera_calibration": "configs/handeye_calibration_result.json",
        "camera_intrinsics": "calibration_images/calibration_matrix.npy",
        "camera_distortion": "calibration_images/distortion_coefficients.npy",
        "handeye_data": "configs/handeye_calibration_data.json",
        "tare": "configs/tare.json",
    }


def camera(
    use_cv_cap: Optional[bool] = None,
    cv_device: int = 0,
    arm_name: Optional[str] = None,
) -> Any:
    """Return camera singleton for the given arm.

    On first call for an arm, creates the camera from robot_mapping.json config.
    Raises RuntimeError if the arm has no camera (``has_camera: false``).
    """
    global _camera_default_use_cv, _camera_default_device
    key = _resolve_arm_name(arm_name)

    if key in _camera_singletons:
        return _camera_singletons[key]

    cfg = _arm_config(key)
    if not cfg.get("has_camera", True):
        raise RuntimeError(f"Camera not available for arm '{key}'")

    device = cfg.get("camera_device", cv_device)
    force_cv = use_cv_cap if use_cv_cap is not None else _camera_default_use_cv
    if force_cv is None:
        force_cv = False
    if use_cv_cap is not None:
        _camera_default_use_cv = use_cv_cap
    _camera_default_device = device if device is not None else cv_device

    if force_cv:
        import cv2
        cap = cv2.VideoCapture(device if device is not None else _camera_default_device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CALIBRATION_IMAGE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIBRATION_IMAGE_HEIGHT)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open webcam (device={device})")
        cam = _CvCamera(cap)
        _camera_singletons[key] = cam
        return cam

    try:
        import pyrealsense2 as rs
    except ImportError:
        raise RuntimeError("pyrealsense2 not installed")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        CALIBRATION_IMAGE_WIDTH,
        CALIBRATION_IMAGE_HEIGHT,
        rs.format.bgr8,
        30,
    )
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    for _ in range(30):
        pipeline.wait_for_frames()
    cam = _RealSenseCamera(pipeline, align)
    _camera_singletons[key] = cam
    return cam


# ---------------------------------------------------------------------------
# YOLO -- global (shared across arms)
# ---------------------------------------------------------------------------

_yolo_singleton: Any = None
_yolo_model_path: Optional[str] = None


def yolo(model_path: Optional[str] = None) -> Any:
    """Return global YOLO model singleton."""
    global _yolo_singleton, _yolo_model_path
    if _yolo_singleton is not None:
        return _yolo_singleton
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed")
    path = model_path or _yolo_model_path
    if path is None:
        path = str(BASE_MODEL)
    logger.info("Using model: %s", path)
    _yolo_singleton = YOLO(path)
    return _yolo_singleton


# ---------------------------------------------------------------------------
# Calibration -- keyed by arm name
# ---------------------------------------------------------------------------

_calibration_singletons: Dict[str, Dict[str, Any]] = {}


def calibration(
    calibration_path: Optional[str] = None,
    intrinsics_path: Optional[str] = None,
    distortion_path: Optional[str] = None,
    tare_path: Optional[str] = None,
    handeye_data_path: Optional[str] = None,
    arm_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Return calibration data for the given arm: T_cam_to_tool, K, D, tare_mm, z0_reference.

    Paths are resolved from robot_mapping.json config for the arm, falling back
    to the explicit arguments or legacy defaults.
    """
    key = _resolve_arm_name(arm_name)

    if key in _calibration_singletons:
        return _calibration_singletons[key]

    cfg = _arm_config(key)

    calib_p = calibration_path or cfg.get("camera_calibration") or "configs/handeye_calibration_result.json"
    calib_path_obj = Path(calib_p)
    if not calib_path_obj.is_absolute():
        calib_path_obj = BASE / calib_path_obj

    intr_p = intrinsics_path or cfg.get("camera_intrinsics") or "calibration_images/calibration_matrix.npy"
    matrix_path = Path(intr_p)
    if not matrix_path.is_absolute():
        matrix_path = BASE / matrix_path

    dist_p = distortion_path or cfg.get("camera_distortion") or "calibration_images/distortion_coefficients.npy"
    dist_path = Path(dist_p)
    if not dist_path.is_absolute():
        dist_path = BASE / dist_path

    tare_p = tare_path or cfg.get("tare") or "configs/tare.json"
    tare_path_obj = Path(tare_p)
    if not tare_path_obj.is_absolute():
        tare_path_obj = BASE / tare_path_obj

    hdata_p = handeye_data_path or cfg.get("handeye_data") or "configs/handeye_calibration_data.json"
    data_path = Path(hdata_p)
    if not data_path.is_absolute():
        data_path = BASE / data_path

    if not calib_path_obj.exists() or not matrix_path.exists() or not dist_path.exists():
        raise FileNotFoundError(
            f"Calibration or intrinsics files not found for arm '{key}': "
            f"calib={calib_path_obj}, K={matrix_path}, D={dist_path}"
        )

    with open(calib_path_obj, "r") as f:
        data = json.load(f)
    T_cam_to_tool = np.array(data["calibration"]["T_cam_to_tool"], dtype=np.float64)
    K = np.load(str(matrix_path))
    D = np.load(str(dist_path))

    tare_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    if tare_path_obj.exists():
        try:
            with open(tare_path_obj, "r") as f:
                t = json.load(f)
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                tare_mm = (float(t[0]), float(t[1]), float(t[2]))
        except Exception:
            pass

    z0_reference = None
    if data_path.exists():
        try:
            with open(data_path, "r") as f:
                hdata = json.load(f)
            z0_reference = hdata.get("metadata", {}).get("z0_reference")
            if z0_reference is not None:
                z0_reference = list(z0_reference)
        except Exception:
            pass

    result = {
        "T_cam_to_tool": T_cam_to_tool,
        "K": K,
        "D": D,
        "tare_mm": tare_mm,
        "z0_reference": z0_reference,
    }
    _calibration_singletons[key] = result
    return result


# ---------------------------------------------------------------------------
# Reset helpers
# ---------------------------------------------------------------------------

def reset_camera(arm_name: Optional[str] = None) -> None:
    """Stop and remove camera singleton(s)."""
    if arm_name is not None:
        key = arm_name
        cam = _camera_singletons.pop(key, None)
        if cam is not None:
            try:
                cam.stop()
            except Exception:
                pass
    else:
        for cam in _camera_singletons.values():
            try:
                cam.stop()
            except Exception:
                pass
        _camera_singletons.clear()


def reset_yolo() -> None:
    global _yolo_singleton
    _yolo_singleton = None


def reset_calibration(arm_name: Optional[str] = None) -> None:
    if arm_name is not None:
        _calibration_singletons.pop(arm_name, None)
    else:
        _calibration_singletons.clear()
