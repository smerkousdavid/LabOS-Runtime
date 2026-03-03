"""
Global singletons for camera, YOLO model, and hand-eye calibration.

Use camera(), yolo(), and calibration() to get the shared instances.
First call creates and caches; optional args only apply on first use.
Paths are resolved via get_project_root() (version2 root).
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

_camera_singleton: Any = None
_camera_use_cv: Optional[bool] = None
_camera_cv_device: int = 0

_yolo_singleton: Any = None
_yolo_model_path: Optional[str] = None

_calibration_singleton: Optional[Dict[str, Any]] = None


def camera(
    use_cv_cap: Optional[bool] = None,
    cv_device: int = 0,
) -> Any:
    """Return global camera singleton (RealSense or OpenCV). First call creates it."""
    global _camera_singleton, _camera_use_cv, _camera_cv_device
    if _camera_singleton is not None:
        return _camera_singleton
    if use_cv_cap is not None:
        _camera_use_cv = use_cv_cap
    if _camera_use_cv is None:
        _camera_use_cv = False
    _camera_cv_device = cv_device

    if _camera_use_cv:
        import cv2
        cap = cv2.VideoCapture(_camera_cv_device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CALIBRATION_IMAGE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIBRATION_IMAGE_HEIGHT)
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")
        _camera_singleton = _CvCamera(cap)
        return _camera_singleton

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
    _camera_singleton = _RealSenseCamera(pipeline, align)
    return _camera_singleton


class _CvCamera:
    def __init__(self, cap):
        self._cap = cap

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        return ret, np.asarray(frame) if ret and frame is not None else None

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (color_image, depth_image). depth_image is always None for CV camera."""
        ret, frame = self._cap.read()
        if ret and frame is not None:
            return np.asarray(frame), None
        return None, None

    def stop(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


class _RealSenseCamera:
    def __init__(self, pipeline, align):
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
        """Return (color_image, depth_image). depth_image is None if depth not available."""
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

    def stop(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


def yolo(model_path: Optional[str] = None) -> Any:
    """Return global YOLO model singleton. First call creates it (optionally with model_path)."""
    global _yolo_singleton, _yolo_model_path
    if _yolo_singleton is not None:
        return _yolo_singleton
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed")
    path = model_path or _yolo_model_path
    # if path is None:
    #     best = _find_best_model()
    #     path = str(best) if best else "yolov8s-seg.pt"
    if path is None:
        path = str(BASE_MODEL)
    logger.info("Using model: %s", path)
    _yolo_singleton = YOLO(path)
    return _yolo_singleton


# unused for now, but could be used to find the best model in the runs directory
# def _find_best_model() -> Optional[Path]:
#     search_dirs = [BASE / "runs" / "segment", BASE / "runs" / "detect"]
#     for runs_dir in search_dirs:
#         if not runs_dir.exists():
#             continue
#         for exp_dir in runs_dir.iterdir():
#             if exp_dir.is_dir():
#                 best_pt = exp_dir / "weights" / "best.pt"
#                 if best_pt.exists():
#                     return best_pt
#         nested = runs_dir / "runs" / runs_dir.name
#         if nested.exists():
#             for exp_dir in nested.iterdir():
#                 if exp_dir.is_dir():
#                     best_pt = exp_dir / "weights" / "best.pt"
#                     if best_pt.exists():
#                         return best_pt
#     return None


def calibration(
    calibration_path: Optional[str] = None,
    intrinsics_path: Optional[str] = None,
    distortion_path: Optional[str] = None,
    tare_path: Optional[str] = None,
    handeye_data_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return global calibration singleton: T_cam_to_tool, K, D, tare_mm, z0_reference.
    z0_reference is from handeye_calibration_data.json metadata (table Z reference).
    """
    global _calibration_singleton
    if _calibration_singleton is not None:
        return _calibration_singleton

    calib_path = Path(calibration_path or str(CONFIGS_PATH / "handeye_calibration_result.json"))
    if not calib_path.is_absolute():
        calib_path = CONFIGS_PATH / calib_path
    matrix_path = Path(intrinsics_path or str(BASE / "calibration_images" / "calibration_matrix.npy"))
    if not matrix_path.is_absolute():
        matrix_path = BASE / matrix_path
    dist_path = Path(distortion_path or str(BASE /  "calibration_images" / "distortion_coefficients.npy"))
    if not dist_path.is_absolute():
        dist_path = BASE / dist_path
    # Default: version2/configs/tare.json (offset [dx, dy, dz] mm applied to tool-frame object position)
    tare_path_p = Path(tare_path or str(CONFIGS_PATH / "tare.json"))
    if not tare_path_p.is_absolute():
        tare_path_p = CONFIGS_PATH / tare_path_p
    data_path = Path(handeye_data_path or str(CONFIGS_PATH / "handeye_calibration_data.json"))
    if not data_path.is_absolute():
        data_path = CONFIGS_PATH / data_path

    if not calib_path.exists() or not matrix_path.exists() or not dist_path.exists():
        raise FileNotFoundError("Calibration or intrinsics files not found")

    with open(calib_path, "r") as f:
        data = json.load(f)
    T_cam_to_tool = np.array(data["calibration"]["T_cam_to_tool"], dtype=np.float64)
    K = np.load(str(matrix_path))
    D = np.load(str(dist_path))

    tare_mm = (0.0, 0.0, 0.0)
    if tare_path_p.exists():
        try:
            with open(tare_path_p, "r") as f:
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

    _calibration_singleton = {
        "T_cam_to_tool": T_cam_to_tool,
        "K": K,
        "D": D,
        "tare_mm": tare_mm,
        "z0_reference": z0_reference,
    }
    return _calibration_singleton


def reset_camera():
    global _camera_singleton
    if _camera_singleton is not None:
        try:
            _camera_singleton.stop()
        except Exception:
            pass
        _camera_singleton = None


def reset_yolo():
    global _yolo_singleton
    _yolo_singleton = None


def reset_calibration():
    global _calibration_singleton
    _calibration_singleton = None
