#!/usr/bin/env python3
"""
YOLO Segment → COCO annotations.json

Runs a YOLO segment model on a folder of images and writes COCO-format
annotations.json with bounding boxes and segmentation masks (as polygons).

Usage:
    python predict.py --source dataset/images --output annotations.json
    python predict.py --model runs/segment/.../weights/best.pt --source ./images --output preds.json
    python predict.py --source ./frames --output coco_pred.json --conf 0.3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# Common image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def collect_images(folder: Path) -> List[Path]:
    """Return list of image paths in folder (recursive=False)."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_classes_from_yaml(yaml_path: Path) -> Optional[List[str]]:
    """Load class names from dataset YAML (names as dict or list)."""
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        if isinstance(names, dict):
            return [names[i] for i in sorted(names.keys())]
        if isinstance(names, list):
            return names
    except Exception:
        pass
    return None


def get_class_names(model, data_yaml: Optional[Path]) -> List[str]:
    """Get class names from model or from data YAML."""
    if data_yaml and data_yaml.exists():
        names = load_classes_from_yaml(data_yaml)
        if names:
            return names
    if hasattr(model, "model") and hasattr(model.model, "names") and model.model.names:
        n = model.model.names
        return [n[i] for i in sorted(n.keys())]
    return [f"class_{i}" for i in range(256)]


def mask_to_polygon(mask: np.ndarray, orig_shape: Tuple[int, int]) -> List[List[float]]:
    """
    Convert binary mask to COCO polygon(s) using contours.
    Returns list of polygons, each polygon is [x1,y1,x2,y2,...,xn,yn].
    """
    h_orig, w_orig = orig_shape
    if mask.shape[:2] != (h_orig, w_orig):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST
        )
    binary = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for cnt in contours:
        if cnt.size < 6:  # need at least 3 points
            continue
        flat = cnt.flatten().tolist()
        polygons.append(flat)
    return polygons


def xy_array_to_polygon(xy: np.ndarray) -> List[float]:
    """Convert (n, 2) array to COCO polygon [x1,y1,x2,y2,...,xn,yn]."""
    return np.asarray(xy, dtype=np.float64).flatten().tolist()


def build_coco_annotations(
    results_list: List[Tuple[Path, Any]],
    class_names: List[str],
    conf_threshold: float,
    image_id_map: Dict[Path, int],
) -> Tuple[List[dict], int]:
    """
    Build COCO annotations from YOLO segment results.
    Returns (annotations list, next_annotation_id).
    """
    annotations = []
    ann_id = 1

    for image_path, result in results_list:
        image_id = image_id_map.get(image_path)
        if image_id is None:
            continue
        if result is None or len(result) == 0:
            continue

        r = result[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        orig_shape = getattr(r, "orig_shape", None)
        if orig_shape is None and hasattr(r, "orig_img"):
            orig_shape = r.orig_img.shape[:2]
        if orig_shape is None:
            orig_shape = (640, 640)

        masks_xy = None
        masks_data = None
        if hasattr(r, "masks") and r.masks is not None:
            if hasattr(r.masks, "xy"):
                try:
                    masks_xy = list(r.masks.xy)
                except Exception:
                    masks_xy = None
            if masks_xy is None and hasattr(r.masks, "data"):
                try:
                    masks_data = r.masks.data.cpu().numpy()
                except Exception:
                    masks_data = None

        for i in range(len(boxes)):
            if float(boxes.conf[i]) < conf_threshold:
                continue

            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            w = x2 - x1
            h = y2 - y1
            # COCO bbox: [x, y, width, height]
            bbox = [x1, y1, w, h]

            cls_id = int(boxes.cls[i])
            category_id = cls_id + 1  # COCO categories often 1-indexed

            # Segmentation: polygons
            segmentation: List[List[float]] = []
            if masks_xy is not None and i < len(masks_xy):
                xy = masks_xy[i]
                if xy is not None and hasattr(xy, "shape") and xy.size >= 6:
                    segmentation.append(xy_array_to_polygon(np.asarray(xy)))
            if not segmentation and masks_data is not None and i < len(masks_data):
                polygons = mask_to_polygon(masks_data[i], orig_shape)
                segmentation.extend(polygons)
            if not segmentation:
                # Fallback: bbox as 4-point polygon
                segmentation.append([x1, y1, x2, y1, x2, y2, x1, y2])

            # Area: from first polygon (shoelace) or bbox
            if segmentation:
                pts = np.array(segmentation[0]).reshape(-1, 2)
                if len(pts) >= 3:
                    area = float(cv2_contour_area(pts))
                else:
                    area = w * h
            else:
                area = w * h

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "segmentation": segmentation,
                "area": round(area, 2),
                "iscrowd": 0,
            }
            annotations.append(ann)
            ann_id += 1

    return annotations, ann_id


def cv2_contour_area(pts: np.ndarray) -> float:
    """Polygon area (shoelace) so we don't require cv2 for area only."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def find_best_segment_model() -> Optional[Path]:
    """Find best.pt under runs/segment (and nested)."""
    for base in [Path("runs/segment"), Path("version2/runs/segment")]:
        if not base.exists():
            continue
        for exp in base.iterdir():
            if exp.is_dir():
                w = exp / "weights" / "best.pt"
                if w.exists():
                    return w
        nested = base / "runs" / "segment"
        if nested.exists():
            for exp in nested.iterdir():
                if exp.is_dir():
                    w = exp / "weights" / "best.pt"
                    if w.exists():
                        return w
    return None


def run(
    source: Path,
    output: Path,
    model_path: str,
    data_yaml: Optional[Path] = None,
    conf: float = 0.25,
    imgsz: int = 640,
) -> int:
    """Run YOLO segment on folder and write COCO annotations.json."""
    source = Path(source).resolve()
    output = Path(output).resolve()

    images = collect_images(source)
    if not images:
        print(f"No images found in {source}")
        return 1

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    class_names = get_class_names(model, data_yaml)
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    print(f"Classes ({len(class_names)}): {class_names}")

    # COCO image entries and id mapping
    coco_images = []
    image_id_map = {}
    for idx, im_path in enumerate(images, start=1):
        # Get size without loading full image (optional: use PIL/cv2)
        try:
            img = cv2.imread(str(im_path))
            if img is not None:
                h, w = img.shape[:2]
            else:
                w, h = 640, 640
        except Exception:
            w, h = 640, 640
        coco_images.append({
            "id": idx,
            "file_name": im_path.name,
            "width": w,
            "height": h,
        })
        image_id_map[im_path] = idx

    # Predict
    results_list: List[Tuple[Path, Any]] = []
    for im_path in images:
        pred = model.predict(
            str(im_path),
            conf=conf,
            imgsz=imgsz,
            verbose=False,
        )
        results_list.append((im_path, pred))

    # Build annotations
    annotations, _ = build_coco_annotations(
        results_list, class_names, conf, image_id_map
    )

    coco = {
        "info": {"description": "YOLO segment predictions", "version": "1.0"},
        "licenses": [],
        "images": coco_images,
        "annotations": annotations,
        "categories": categories,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"Wrote {len(annotations)} annotations for {len(coco_images)} images -> {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run YOLO segment on a folder and output COCO annotations.json (bboxes + polygon masks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --source dataset/images --output annotations.json
  python predict.py --model runs/segment/.../weights/best.pt --source ./images --output preds.json
  python predict.py --source ./frames --output coco_pred.json --conf 0.3 --imgsz 640
        """,
    )
    parser.add_argument("--source", "-s", type=str, required=True,
                        help="Folder containing images")
    parser.add_argument("--output", "-o", type=str, default="annotations.json",
                        help="Output COCO annotations JSON path (default: annotations.json)")
    parser.add_argument("--model", "-m", type=str, default='weights/segmentv3.pt',
                        help="Path to YOLO segment weights (default: auto-detect best.pt)")
    parser.add_argument("--data-yaml", type=str, default=None,
                        help="Dataset YAML for class names (default: dataset.yaml in cwd)")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=768,
                        help="Inference image size (default: 768)")

    args = parser.parse_args()

    if not ULTRALYTICS_AVAILABLE:
        print("Error: ultralytics not installed. pip install ultralytics")
        return 1

    model_path = args.model
    if model_path is None:
        best = find_best_segment_model()
        if best is not None:
            model_path = str(best)
            print(f"Using found model: {model_path}")
        else:
            model_path = "yolov8s-seg.pt"
            print(f"Using default model: {model_path}")

    data_yaml_path = None
    if args.data_yaml:
        data_yaml_path = Path(args.data_yaml)
    else:
        for d in [Path("."), Path(__file__).resolve().parent]:
            y = d / "dataset.yaml"
            if y.exists():
                data_yaml_path = y
                break

    return run(
        source=Path(args.source),
        output=Path(args.output),
        model_path=model_path,
        data_yaml=data_yaml_path,
        conf=args.conf,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    sys.exit(main())
