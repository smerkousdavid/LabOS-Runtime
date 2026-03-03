#!/usr/bin/env python3
"""
YOLO RealSense / Webcam Test Script (detect + segment)

Opens RealSense or OpenCV webcam and displays real-time object detection. Supports:
- detect: YOLO-World (bbox only).
- segment: YOLO segment model (bbox + masks), e.g. from train_yolo.py --task segment.

Usage:
    python -m aira.vision.test_yolo
    python -m aira.vision.test_yolo --cv-cap
    python -m aira.vision.test_yolo --task segment --model runs/segment/yolo_seg_train/weights/best.pt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

from aira.utils.paths import get_project_root
from aira.vision.dataset import get_class_names, get_dataset_yaml_path


def _default_confidence_from_config() -> float:
    """Default confidence from configs/objects.yaml (default_confidence), else 0.2."""
    root = get_project_root()
    path = root / "configs" / "objects.yaml"
    if path.exists():
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            return float(data.get("default_confidence", 0.2))
        except Exception:
            pass
    return 0.2


def _load_per_class_confidence(class_names: list, default: float) -> dict:
    """
    Load per-class confidence from configs/objects.yaml.
    Returns dict: class_name -> confidence. Classes not in config use default.
    """
    root = get_project_root()
    path = root / "configs" / "objects.yaml"
    out = {}
    default_conf = default
    yolo_class_to_conf = {}  # yolo_class from config -> confidence
    if path.exists():
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            default_conf = float(data.get("default_confidence", default))
            for key, val in data.items():
                if key == "default_confidence" or not isinstance(val, dict):
                    continue
                yolo_class = val.get("yolo_class")
                if yolo_class is None:
                    continue
                conf = val.get("confidence")
                if conf is not None:
                    yolo_class_to_conf[str(yolo_class).strip()] = float(conf)
        except Exception:
            pass
    for name in class_names:
        name_str = str(name).strip()
        out[name_str] = yolo_class_to_conf.get(name_str, default_conf)
    return out

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

try:
    from ultralytics import YOLOWorld, YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False
    sys.exit(1)


# Default class names when dataset YAML is missing (used by viewer when not from model/YAML)
DEFAULT_CLASSES = [
    "Vortex Genie 2",
    "Vortex Genie Hole",
    "Vortex Genie Top Plate",
    "50ml eppendorf tube",
    "50Ml eppendorf cap",
    "50Ml 4 way rack",
    "4 way rack 50ml hole",
    "4 way rack 5ml hole"
]


class YOLORealSenseViewer:
    """Real-time YOLO detection/segment with RealSense camera."""

    def __init__(
        self,
        model_path: str = "yolov8s-worldv2.pt",
        task: str = "detect",
        classes: list = None,
        conf_threshold: float = 0.25,
        show_masks: bool = True,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        use_cv_cap: bool = False,
        cv_device: int = 0
    ):
        self.model_path = model_path
        self.task = task  # "detect" (YOLO-World) or "segment" (YOLO seg, predicts masks)
        self.classes = classes or DEFAULT_CLASSES
        self.conf_threshold = conf_threshold
        # Per-class confidence from configs/objects.yaml (e.g. Vortex Genie Hole -> 0.01)
        self.conf_threshold_per_class = _load_per_class_confidence(self.classes, self.conf_threshold)
        self._conf_for_predict = min(self.conf_threshold_per_class.values())
        self.show_masks = show_masks
        self.width = width
        self.height = height
        self.fps = fps
        self.use_cv_cap = use_cv_cap
        self.cv_device = cv_device

        # RealSense (when not using OpenCV)
        self.pipeline = None
        self.profile = None
        self.align = None
        # OpenCV capture (when --cv-cap)
        self.cv_cap = None

        # YOLO model (YOLOWorld for detect, YOLO for segment)
        self.model = None

        # Colors for each class
        self.colors = self._generate_colors(len(self.classes))

    def _generate_colors(self, num_classes: int):
        """Generate distinct colors for each class."""
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color_bgr)))
        return colors

    def start_camera(self) -> bool:
        """Initialize camera: OpenCV VideoCapture (webcam) or RealSense."""
        if self.use_cv_cap:
            return self._start_cv_camera()
        return self._start_realsense_camera()

    def _start_cv_camera(self) -> bool:
        """Initialize OpenCV VideoCapture (webcam)."""
        print(f"Initializing OpenCV webcam (device {self.cv_device})...")
        self.cv_cap = cv2.VideoCapture(self.cv_device)
        if not self.cv_cap.isOpened():
            print(f"Failed to open webcam device {self.cv_device}")
            return False
        # Set resolution if supported
        self.cv_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cv_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cv_cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.width = int(self.cv_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Webcam started: {self.width}x{self.height}")
        return True

    def _start_realsense_camera(self) -> bool:
        """Initialize RealSense camera."""
        if not REALSENSE_AVAILABLE:
            print("Error: pyrealsense2 not installed. Install with: pip install pyrealsense2 (or use --cv-cap for webcam)")
            return False

        print("Initializing RealSense camera...")
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Try requested resolution with fallbacks
        resolutions = [
            (self.width, self.height),
            (1280, 720),
            (848, 480),
            (640, 480),
        ]

        started = False
        for w, h in resolutions:
            try:
                config = rs.config()
                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, self.fps)
                config.enable_stream(rs.stream.depth, w, h, rs.format.z16, self.fps)

                print(f"  Trying {w}x{h}...")
                self.profile = self.pipeline.start(config)
                self.width, self.height = w, h
                started = True
                break
            except RuntimeError:
                continue

        if not started:
            print("Failed to start RealSense with any resolution!")
            return False

        # Create alignment object
        self.align = rs.align(rs.stream.color)

        print(f"  Camera started: {self.width}x{self.height}")

        # Warm up
        print("  Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()

        return True

    def stop_camera(self):
        """Stop camera (RealSense or OpenCV)."""
        if self.use_cv_cap and self.cv_cap is not None:
            try:
                self.cv_cap.release()
                print("Webcam stopped.")
            except Exception:
                pass
            self.cv_cap = None
        elif self.pipeline and self.profile:
            try:
                self.pipeline.stop()
                print("Camera stopped.")
            except Exception:
                pass

    def load_model(self) -> bool:
        """Load YOLO model: segment (masks) with YOLO(), detect (bbox only) with YOLOWorld()."""
        print(f"\nLoading YOLO model ({self.task}): {self.model_path}")

        try:
            if self.task == "segment":
                self.model = YOLO(self.model_path)
                # Use class names from model when not overridden by user (e.g. from --classes or data-yaml)
                if (self.classes == DEFAULT_CLASSES and
                        hasattr(self.model, "model") and hasattr(self.model.model, "names") and self.model.model.names):
                    names = self.model.model.names
                    self.classes = [names[i] for i in sorted(names.keys())]
                    self.colors = self._generate_colors(len(self.classes))
                    print(f"  Class names from model: {self.classes}")
                print("  Segment model loaded (mask prediction enabled).")
            else:
                self.model = YOLOWorld(self.model_path)
                print(f"  Setting custom vocabulary: {self.classes}")
                self.model.set_classes(self.classes)
                print("  Detect model loaded (YOLO-World).")
            return True
        except Exception as e:
            print(f"  Error loading model: {e}")
            return False

    def get_frames(self):
        """Get color frame (and depth if RealSense). Returns (color_image, depth_image)."""
        if self.use_cv_cap and self.cv_cap is not None:
            ret, frame = self.cv_cap.read()
            if not ret or frame is None:
                return None, None
            return frame, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame:
                return None, None

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = None
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())

            return color_image, depth_image
        except Exception as e:
            print(f"Frame error: {e}")
            return None, None

    def draw_detections(self, image: np.ndarray, results) -> np.ndarray:
        """Draw detections, bounding boxes, and masks on image."""
        if results is None or len(results) == 0:
            return image

        result = results[0]
        display_image = image.copy()

        # Get detection data
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return display_image

        # Draw masks if available (segment model predicts masks; detect/YOLO-World may not)
        if self.show_masks and hasattr(result, 'masks') and result.masks is not None:
            try:
                masks_np = result.masks.data.cpu().numpy()
            except Exception:
                masks_np = None
            if masks_np is not None:
                for i in range(min(len(masks_np), len(boxes))):
                    cls_id = int(boxes.cls[i])
                    class_name = self.classes[cls_id] if cls_id < len(self.classes) else f"Class {cls_id}"
                    thresh = self.conf_threshold_per_class.get(class_name, self.conf_threshold)
                    if boxes.conf[i] < thresh:
                        continue
                    mask = masks_np[i]
                    # Resize mask to display size (segment masks may be inference resolution)
                    if mask.shape[:2] != (display_image.shape[0], display_image.shape[1]):
                        mask = cv2.resize(
                            mask.astype(np.float32),
                            (display_image.shape[1], display_image.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    color = self.colors[cls_id % len(self.colors)]
                    mask_colored = np.zeros_like(display_image)
                    mask_colored[mask_binary > 0] = color
                    display_image = cv2.addWeighted(display_image, 0.7, mask_colored, 0.3, 0)

        # Draw bounding boxes and labels (per-class confidence from configs/objects.yaml)
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            class_name = self.classes[cls_id] if cls_id < len(self.classes) else f"Class {cls_id}"
            thresh = self.conf_threshold_per_class.get(class_name, self.conf_threshold)
            if boxes.conf[i] < thresh:
                continue

            # Get box coordinates
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            # Get class info
            conf = float(boxes.conf[i])
            color = self.colors[cls_id % len(self.colors)]

            # Draw bounding box
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{class_name} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_image, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
            cv2.putText(display_image, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return display_image

    def run(self):
        """Main viewer loop."""
        print("\n" + "="*60)
        src = "WEBCAM (OpenCV)" if self.use_cv_cap else "REALSENSE"
        print("YOLO " + src + " " + ("SEGMENT (masks)" if self.task == "segment" else "DETECT"))
        print("="*60)
        print("Controls:")
        print("  Q/ESC  - Quit")
        print("  M      - Toggle masks")
        print("  +/-    - Adjust confidence threshold")
        print("="*60 + "\n")

        # Load model
        if not self.load_model():
            return

        # Start camera
        if not self.start_camera():
            return

        cv2.namedWindow('YOLO Detection', cv2.WINDOW_AUTOSIZE)

        frame_count = 0
        import time
        fps_start = time.time()
        fps = 0.0

        try:
            while True:
                # Get frames
                color_image, depth_image = self.get_frames()
                if color_image is None:
                    continue

                # Run inference (use min of per-class thresholds so low-conf objects like genie hole appear)
                results = self.model.predict(
                    color_image,
                    conf=self._conf_for_predict,
                    imgsz=640,
                    verbose=False
                )

                # Draw detections
                display_image = self.draw_detections(color_image, results)

                # Draw FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start)
                    fps_start = time.time()

                cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_image, f"Conf: min={self._conf_for_predict:.2f} (per-class)", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_image, f"Masks: {'ON' if self.show_masks else 'OFF'}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show
                cv2.imshow('YOLO Detection', display_image)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('m') or key == ord('M'):
                    self.show_masks = not self.show_masks
                    print(f"Masks: {'ON' if self.show_masks else 'OFF'}")
                elif key == ord('+') or key == ord('='):
                    self.conf_threshold = min(1.0, self.conf_threshold + 0.05)
                    self.conf_threshold_per_class = _load_per_class_confidence(self.classes, self.conf_threshold)
                    self._conf_for_predict = min(self.conf_threshold_per_class.values())
                    print(f"Confidence threshold (default): {self.conf_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.conf_threshold = max(0.0, self.conf_threshold - 0.05)
                    self.conf_threshold_per_class = _load_per_class_confidence(self.classes, self.conf_threshold)
                    self._conf_for_predict = min(self.conf_threshold_per_class.values())
                    print(f"Confidence threshold (default): {self.conf_threshold:.2f}")

        finally:
            cv2.destroyAllWindows()
            self.stop_camera()

        print("\nViewer closed.")


def find_best_model(task: str = "segment") -> Path:
    """Find the best trained model in runs directory (segment or detect)."""
    root = get_project_root()
    if task == "segment":
        search_dirs = [root / "runs" / "segment", root / "runs" / "detect"]
    else:
        search_dirs = [root / "runs" / "detect", root / "runs" / "segment"]

    for runs_dir in search_dirs:
        if not runs_dir.exists():
            continue
        for exp_dir in runs_dir.iterdir():
            if exp_dir.is_dir():
                best_pt = exp_dir / "weights" / "best.pt"
                if best_pt.exists():
                    return best_pt
        # Also check nested runs/segment/runs/segment/... (train_yolo layout)
        nested = runs_dir / "runs" / runs_dir.name
        if nested.exists():
            for exp_dir in nested.iterdir():
                if exp_dir.is_dir():
                    best_pt = exp_dir / "weights" / "best.pt"
                    if best_pt.exists():
                        return best_pt

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Test YOLO-World detection with RealSense camera',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Q/ESC  - Quit
  M      - Toggle mask display
  +/-    - Adjust confidence threshold

Examples:
  PYTHONPATH=. python scripts/test_yolo.py
  python scripts/test_yolo.py --task segment --model runs/segment/yolo_seg_train/weights/best.pt
  python scripts/test_yolo.py --conf 0.3 --no-masks
        """
    )

    root = get_project_root()
    default_yaml = str(get_dataset_yaml_path())
    default_model = "weights/segmentv7.pt"

    parser.add_argument('--task', type=str, choices=('detect', 'segment'), default='segment',
                       help='Task: detect (YOLO-World bbox) or segment (YOLO seg, predicts masks) (default: segment)')
    parser.add_argument('--model', type=str, default=default_model,
                       help='Path to model weights (default: auto-detect best.pt from runs/segment or runs/detect)')
    parser.add_argument('--data-yaml', type=str, default=default_yaml,
                       help='Dataset YAML to load classes from (default: configs/dataset.yaml)')
    parser.add_argument('--classes', type=str, nargs='+', default=None,
                       help='Custom class names (overrides YAML/model)')
    parser.add_argument('--conf', type=float, default=None,
                       help='Confidence threshold (default: from configs/objects.yaml default_confidence, else 0.25)')
    parser.add_argument('--no-masks', action='store_true',
                       help='Disable mask display')
    parser.add_argument('--width', type=int, default=1280,
                       help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Camera height (default: 720)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frame rate (default: 30)')
    parser.add_argument('--cv-cap', action='store_true',
                       help='Use OpenCV VideoCapture (webcam) instead of RealSense')
    parser.add_argument('--cv-device', type=int, default=1,
                       help='Webcam device index for --cv-cap (default: 0)')

    args = parser.parse_args()

    # Use config default_confidence (same as robot) when --conf not set
    conf_threshold = args.conf if args.conf is not None else _default_confidence_from_config()
    if args.conf is None:
        print(f"Using confidence threshold from config: {conf_threshold:.2f}")

    if not args.cv_cap and not REALSENSE_AVAILABLE:
        print("Error: pyrealsense2 not installed. Install with: pip install pyrealsense2 (or use --cv-cap for webcam)")
        return 1

    if not ULTRALYTICS_AVAILABLE:
        return 1

    # Determine model path and task (auto-detect best.pt when using default)
    task = args.task
    model_path = args.model
    if model_path == default_model:
        best_model = find_best_model(task=task)
        if best_model:
            model_path = str(best_model)
            if "segment" in model_path:
                task = "segment"
            print(f"Found trained model: {model_path} (task={task})")
        else:
            if task == "segment":
                model_path = "yolov8s-seg.pt"
            else:
                model_path = "yolov8s-worldv2.pt"
            print(f"Using default model: {model_path}")
    else:
        if "segment" in model_path:
            task = "segment"
        else:
            task = args.task

    # Load classes from dataset YAML (project config) or fallback
    classes = args.classes
    if classes is None:
        yaml_path = Path(args.data_yaml)
        if yaml_path.exists():
            classes = get_class_names(yaml_path)
        if not classes:
            classes = get_class_names() or DEFAULT_CLASSES
        if classes:
            print(f"Loaded {len(classes)} classes from dataset config")
        else:
            classes = DEFAULT_CLASSES
            print(f"Using default classes: {classes}")

    # Create viewer (inference and draw both use conf_threshold)
    viewer = YOLORealSenseViewer(
        model_path=model_path,
        task=task,
        classes=classes,
        conf_threshold=conf_threshold,
        show_masks=not args.no_masks,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_cv_cap=args.cv_cap,
        cv_device=args.cv_device
    )

    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
