#!/usr/bin/env python3
"""
Center on Object: move toolhead to a registered object by name.

Loads object presets from configs/objects.yaml (yolo_class, shape, pick_type, confidence).
Uses the general move_to_object() API from robot and global singletons (camera, yolo, calibration).

Usage:
    python center_object.py 50ml eppendorf
    python center_object.py 50ml eppendorf --move --repeat 3
    python center_object.py 50ml eppendorf --move --average-frames 5 --repeat-skip 3
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_object_presets() -> Dict[str, Dict[str, Any]]:
    """Load object presets from configs/objects.yaml (single source of truth)."""
    from aira.robot import _load_objects_for_robot, _object_presets_only
    data = _load_objects_for_robot()
    return _object_presets_only(data)


def _load_objects_full() -> Dict[str, Any]:
    """Load full objects config (default_confidence + presets)."""
    from aira.robot import _load_objects_for_robot
    return _load_objects_for_robot()


def main():
    import argparse
    from aira.robot import move_to_object
    from aira.vision.vision import load_tare_json
    from aira.vision.singletons import calibration, camera, yolo

    presets = _load_object_presets()
    if not presets:
        print("Error: No objects in configs/objects.yaml")
        return 1

    parser = argparse.ArgumentParser(
        description='Center on a registered object (from configs/objects.yaml)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('object', type=str, choices=list(presets.keys()),
                        help='Object name (e.g. 50ml eppendorf)')
    parser.add_argument('--move', action='store_true',
                        help='Move robot to object (otherwise display only)')
    parser.add_argument('--pick', type=str, default='toolhead_close',
                        choices=['toolhead_close', 'camera_center', 'largest', 'tl', 'tr', 'bl', 'br', 'highest_confidence', 'ranked'],
                        help='Which detection to move to (default from objects.yaml or toolhead_close)')
    parser.add_argument('--average-frames', type=int, default=5,
                        help='Frames to average for each move (default: 5)')
    parser.add_argument('--repeat', type=int, default=3,
                        help='Max moves before finishing (default: 3)')
    parser.add_argument('--repeat-skip', type=float, default=2.0,
                        help='Skip move if within this many mm of target (default: 2)')
    parser.add_argument('--speed', type=float, default=100,
                        help='Robot move speed (default: 100)')
    parser.add_argument('--acc', type=float, default=500,
                        help='Robot acceleration (default: 500)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display window')
    parser.add_argument('--model', type=str, default='weights/segmentv6.pt',
                        help='YOLO model path (default: auto-detect)')
    parser.add_argument('--calibration', '-c', type=str, default=None,
                        help='Hand-eye calibration JSON')
    parser.add_argument('--intrinsics', type=str, default=None,
                        help='Camera matrix .npy')
    parser.add_argument('--distortion', type=str, default=None,
                        help='Distortion coefficients .npy')
    parser.add_argument('--tare', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='Tare adjustment mm (default: from tare.json)')
    parser.add_argument('--ip', type=str, default='192.168.1.195',
                        help='Robot IP when using --move')
    parser.add_argument('--cv-cap', action='store_true',
                        help='Use OpenCV webcam instead of RealSense')
    parser.add_argument('--cv-device', type=int, default=0,
                        help='Webcam device index')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='YOLO confidence threshold')
    parser.add_argument('--offset', type=float, nargs='+', default=None,
                        metavar=('DX', 'DY', 'DZ'),
                        help='Offset mm (dx, dy) or (dx, dy, dz) from object center (e.g. 50 0 = 50mm in front)')
    args = parser.parse_args()

    objects_full = _load_objects_full()
    preset = objects_full[args.object]
    shape = preset.get("shape", {})
    yolo_class = preset.get("yolo_class")
    pick_type = preset.get("pick_type", args.pick)

    # Init singletons with CLI paths (first call wins)
    calibration(
        calibration_path=args.calibration or str(ROOT / 'configs' / 'handeye_calibration_result.json'),
        intrinsics_path=args.intrinsics or str(ROOT / 'configs' / 'calibration_images' / 'calibration_matrix.npy'),
        distortion_path=args.distortion or str(ROOT / 'configs' / 'calibration_images' / 'distortion_coefficients.npy'),
        tare_path=str(ROOT / 'configs' / 'tare.json'),
    )
    if args.model:
        yolo(model_path=args.model)
    camera(use_cv_cap=args.cv_cap, cv_device=args.cv_device)

    tare_mm = None
    if args.tare is not None:
        tare_mm = (float(args.tare[0]), float(args.tare[1]), float(args.tare[2]))
    else:
        loaded = load_tare_json(ROOT / 'configs' / 'tare.json')
        if loaded is not None:
            tare_mm = loaded

    offset_tuple = None
    if args.offset is not None and len(args.offset) >= 2:
        offset_tuple = tuple(float(x) for x in args.offset[:3]) if len(args.offset) >= 3 else (float(args.offset[0]), float(args.offset[1]))
    result = move_to_object(
        shape=shape,
        pick_type=pick_type,
        yolo_class=yolo_class,
        average_frames=args.average_frames,
        repeat=args.repeat,
        repeat_skip_mm=args.repeat_skip,
        speed=args.speed,
        acc=args.acc,
        display=not args.no_display,
        use_robot=args.move,
        robot_ip=args.ip if args.move else None,
        tare_mm=tare_mm,
        offset=offset_tuple,
    )

    if not result.get('success'):
        print('Error:', result.get('error', 'unknown'))
        return 1
    print(f"Done. moves_done={result.get('moves_done', 0)}, final_xy_tool_mm={result.get('final_xy_tool_mm')}")
    if not args.move and result.get('final_xy_tool_mm') is not None:
        print("Tip: run with --move to move the robot to the object.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
