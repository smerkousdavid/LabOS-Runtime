"""
Vision: background YOLO detection, visible_objects, object_within, and helpers.

- visible_objects() -> list of detections (bbox_xyxy, class_id, class_name, conf)
- object_within(obj_a, obj_b, proportion) -> overlap check
- parse_shape, object_point_3d_camera, pick_detection, camera_to_tool, etc.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

DEFAULT_CIRCLE_DIAMETER_MM = 33.0

_vision_thread: Optional[threading.Thread] = None
_vision_stop = threading.Event()
_visible: List[Dict[str, Any]] = []
_visible_lock = threading.Lock()
_vision_started = False
_vision_conf = 0.25
_vision_imgsz = 640
_vision_show_window = True


def _bbox_from_obj(obj: Union[Dict[str, Any], Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    if isinstance(obj, (list, tuple)) and len(obj) >= 4:
        return float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])
    if isinstance(obj, dict) and "bbox_xyxy" in obj:
        b = obj["bbox_xyxy"]
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    raise ValueError("object must have 'bbox_xyxy' or be (x1,y1,x2,y2)")


def _bbox_area(x1: float, y1: float, x2: float, y2: float) -> float:
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def object_within(
    obj_a: Union[Dict[str, Any], Tuple[float, float, float, float]],
    obj_b: Union[Dict[str, Any], Tuple[float, float, float, float]],
    proportion: float = 0.5,
) -> bool:
    ba = _bbox_from_obj(obj_a)
    bb = _bbox_from_obj(obj_b)
    area_a = _bbox_area(*ba)
    if area_a <= 0:
        return False
    inter = _bbox_intersection_area(ba, bb)
    return (inter / area_a) >= proportion


def camera_to_tool(p_cam_mm: np.ndarray, T_cam_to_tool: np.ndarray) -> np.ndarray:
    if p_cam_mm.size == 3:
        p = np.append(np.asarray(p_cam_mm, dtype=np.float64), 1.0)
    else:
        p = np.asarray(p_cam_mm, dtype=np.float64).ravel()[:4]
    p_tool = T_cam_to_tool @ p
    return p_tool[:3]


def load_classes_from_yaml(yaml_path: Path) -> Optional[List[str]]:
    try:
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        if isinstance(names, dict):
            return [names[i] for i in sorted(names.keys())]
        if isinstance(names, list):
            return names
    except Exception:
        pass
    return None


def load_tare_json(path: Path) -> Optional[Tuple[float, float, float]]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, (list, tuple)) and len(data) >= 3:
            return (float(data[0]), float(data[1]), float(data[2]))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _parse_mm(value: Union[int, float, str]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    if s.endswith("mm"):
        return float(s[:-2].strip())
    return float(s)


def parse_shape(shape: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(shape)
    stype = (out.get("type") or "circle").lower()
    out["type"] = stype
    out["location"] = (out.get("location") or "center").lower()
    if stype == "circle":
        d = out.get("diameter") or out.get("diameter_mm")
        out["diameter_mm"] = _parse_mm(d) if d is not None else DEFAULT_CIRCLE_DIAMETER_MM
    elif stype == "square":
        s = out.get("side") or out.get("side_mm")
        out["side_mm"] = _parse_mm(s) if s is not None else 24.0
    elif stype == "rect":
        out["width_mm"] = _parse_mm(out.get("width") or out.get("width_mm") or 40)
        out["height_mm"] = _parse_mm(out.get("height") or out.get("height_mm") or 30)
    return out


def _bbox_point_uv(bbox_xyxy: Tuple[float, float, float, float], location: str) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if location == "center":
        return cx, cy
    if location == "tl":
        return x1, y1
    if location == "tr":
        return x2, y1
    if location == "bl":
        return x1, y2
    if location == "br":
        return x2, y2
    return cx, cy


def object_point_3d_camera(
    bbox_xyxy: Tuple[float, float, float, float],
    shape: Dict[str, Any],
    K: np.ndarray,
) -> np.ndarray:
    shape = parse_shape(shape)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x1, y1, x2, y2 = bbox_xyxy
    w_px = x2 - x1
    h_px = y2 - y1
    if w_px <= 0 or h_px <= 0:
        return np.full(3, np.nan)
    stype = shape["type"]
    location = shape["location"]
    u, v = _bbox_point_uv(bbox_xyxy, location)
    if stype == "circle":
        size_mm = shape["diameter_mm"]
        size_px = max(w_px, h_px)
    elif stype == "square":
        size_mm = shape["side_mm"]
        size_px = max(w_px, h_px)
    else:
        size_mm = (shape.get("width_mm", 40) * shape.get("height_mm", 30)) ** 0.5
        size_px = (w_px * h_px) ** 0.5
    if size_px <= 0:
        return np.full(3, np.nan)
    Z_cam_mm = fx * (size_mm / size_px)
    x_cam_mm = (u - cx) * Z_cam_mm / fx
    y_cam_mm = (v - cy) * Z_cam_mm / fy
    return np.array([x_cam_mm, y_cam_mm, Z_cam_mm], dtype=np.float64)


def pick_detection(
    detections: List[Dict[str, Any]],
    pick_type: Union[str, Tuple[float, float]],
    image_shape: Tuple[int, int],
    T_cam_to_tool: np.ndarray,
    tare_arr: np.ndarray,
) -> Optional[Dict[str, Any]]:
    if not detections:
        return None
    # Support tuple (px_x, px_y) to pick detection closest to that pixel location
    if isinstance(pick_type, tuple) and len(pick_type) == 2:
        target_px = pick_type
        def key_px(d):
            b = d["bbox_xyxy"]
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            return (cx - target_px[0]) ** 2 + (cy - target_px[1]) ** 2
        return min(detections, key=key_px)
    pick = (pick_type or "toolhead_close").lower()
    H, W = image_shape[:2]
    cam_center = (W / 2.0, H / 2.0)
    if pick == "toolhead_close":
        def key(d):
            p = d.get("p_tool_mm")
            if p is None or not np.isfinite(p).all():
                return float("inf")
            return np.sqrt(p[0] ** 2 + p[1] ** 2)
        detections = sorted(detections, key=key)
        return detections[0] if key(detections[0]) != float("inf") else None
    if pick == "camera_center":
        def key(d):
            b = d["bbox_xyxy"]
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            return (cx - cam_center[0]) ** 2 + (cy - cam_center[1]) ** 2
        return min(detections, key=key)
    if pick == "largest":
        def key(d):
            b = d["bbox_xyxy"]
            return -((b[2] - b[0]) * (b[3] - b[1]))
        return min(detections, key=key)
    if pick == "highest_confidence":
        return max(detections, key=lambda d: d.get("conf", 0.0))
    if pick == "ranked":
        # Rank by area (1 = largest) and by confidence (1 = highest); pick lowest average rank.
        def area(d):
            b = d["bbox_xyxy"]
            return (b[2] - b[0]) * (b[3] - b[1])
        def conf(d):
            return d.get("conf", 0.0)
        by_area = sorted(detections, key=area, reverse=True)
        by_conf = sorted(detections, key=conf, reverse=True)
        rank_area = {id(d): i + 1 for i, d in enumerate(by_area)}
        rank_conf = {id(d): i + 1 for i, d in enumerate(by_conf)}
        def avg_rank(d):
            return (rank_area[id(d)] + rank_conf[id(d)]) / 2.0
        return min(detections, key=avg_rank)
    if pick == "tl":
        detections = sorted(detections, key=lambda d: (d["bbox_xyxy"][1], d["bbox_xyxy"][0]))
        return detections[0]
    if pick == "tr":
        detections = sorted(detections, key=lambda d: (d["bbox_xyxy"][1], -d["bbox_xyxy"][2]))
        return detections[0]
    if pick == "bl":
        detections = sorted(detections, key=lambda d: (-d["bbox_xyxy"][3], d["bbox_xyxy"][0]))
        return detections[0]
    if pick == "br":
        detections = sorted(detections, key=lambda d: (-d["bbox_xyxy"][3], -d["bbox_xyxy"][2]))
        return detections[0]
    return detections[0]


def resolve_class_to_index(classes: List[str], yolo_class: Optional[Union[str, int]]) -> int:
    if yolo_class is None:
        return 0
    if isinstance(yolo_class, int):
        return max(0, min(yolo_class, len(classes) - 1))
    name = str(yolo_class).strip().lower()
    for i, c in enumerate(classes):
        if name in c.lower():
            return i
    return 0


def _detection_loop(
    conf: float = 0.25,
    imgsz: int = 640,
    poll_interval: float = 0.05,
    show_window: bool = True,
) -> None:
    global _visible
    from aira.vision.singletons import camera, yolo
    cam = camera()
    model = yolo()
    class_names = getattr(model, "names", None)
    if class_names is not None and isinstance(class_names, dict):
        names_list = [class_names.get(i, str(i)) for i in sorted(class_names.keys())]
    else:
        names_list = []
    window_name = "Vision"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    while not _vision_stop.is_set():
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(poll_interval)
            continue
        try:
            results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        except Exception:
            time.sleep(poll_interval)
            continue
        out: List[Dict[str, Any]] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(float, box)
                cls_id = int(boxes.cls[i])
                conf_val = float(boxes.conf[i])
                name = names_list[cls_id] if cls_id < len(names_list) else str(cls_id)
                out.append({
                    "bbox_xyxy": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "class_name": name,
                    "conf": conf_val,
                })
        with _visible_lock:
            _visible = out
        if show_window and frame is not None:
            disp = frame.copy()
            for d in out:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                label = f"{d['class_name']} {d['conf']:.2f}"
                cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(disp, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(disp, f"Detections: {len(out)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window_name, disp)
            cv2.waitKey(1)
        time.sleep(poll_interval)
    if show_window:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass


def visible_objects(
    *,
    conf: float = 0.25,
    imgsz: int = 640,
    show_window: bool = True,
) -> List[Dict[str, Any]]:
    global _vision_thread, _vision_started, _vision_conf, _vision_imgsz, _vision_show_window
    if not _vision_started:
        with _visible_lock:
            if not _vision_started:
                _vision_conf = conf
                _vision_imgsz = imgsz
                _vision_show_window = show_window
                _vision_stop.clear()
                _vision_thread = threading.Thread(
                    target=_detection_loop,
                    kwargs={
                        "conf": _vision_conf,
                        "imgsz": _vision_imgsz,
                        "show_window": _vision_show_window,
                    },
                    daemon=True,
                )
                _vision_thread.start()
                _vision_started = True
    with _visible_lock:
        return list(_visible)


def stop_vision() -> None:
    global _vision_started
    _vision_stop.set()
    if _vision_thread is not None and _vision_thread.is_alive():
        _vision_thread.join(timeout=2.0)
    try:
        cv2.destroyWindow("Vision")
    except Exception:
        pass
    _vision_started = False
