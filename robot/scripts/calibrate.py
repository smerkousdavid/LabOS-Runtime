#!/usr/bin/env python3
"""
Single entrypoint for all calibration modes.
Run from version2: PYTHONPATH=. python scripts/calibrate.py [args]

  python scripts/calibrate.py --mode capture --output calibration_images
  python scripts/calibrate.py --mode intrinsics --input calibration_images --output calibration_images
  python scripts/calibrate.py --mode handeye --ip 192.168.1.195 --output handeye_calibration_data.json
  python scripts/calibrate.py --mode handeye_solve --input handeye_calibration_data.json --output configs/handeye_calibration_result.json
  python scripts/calibrate.py --mode depth --on-chip
"""

import argparse
import sys
from pathlib import Path

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Calibration: capture, intrinsics, hand-eye, handeye_solve, depth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["capture", "intrinsics", "handeye", "handeye_solve", "depth"],
        required=True,
        help="Calibration mode",
    )
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--input", "-i", type=str, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--checkerboard", type=int, nargs=2, default=[7, 9], metavar=("COLS", "ROWS"))
    parser.add_argument("--square-size", type=float, default=20.0, help="Checkerboard square size (mm)")
    parser.add_argument("--visualize", "-v", action="store_true")
    parser.add_argument("--pattern", type=str, default="*.png")
    parser.add_argument("--ip", type=str, default="192.168.1.195", help="Robot IP (handeye)")
    parser.add_argument("--aruco-dict", type=str, default="DICT_5X5_100")
    parser.add_argument("--aruco-size", type=float, default=0.04, help="ArUco marker size (m)")
    parser.add_argument("--intrinsics", type=str, default=None)
    parser.add_argument("--distortion", type=str, default=None)
    parser.add_argument("--on-chip", action="store_true", help="Depth: run on-chip calibration")
    parser.add_argument("--tare", type=float, default=None, metavar="MM", help="Depth: tare at distance (mm)")
    parser.add_argument("--health", action="store_true", help="Depth: check health only")
    parser.add_argument("--reset", action="store_true", help="Depth: reset to factory")
    parser.add_argument("--preview", action="store_true", help="Depth: live preview")
    args = parser.parse_args()

    from aira.utils.paths import get_project_root
    project_root = get_project_root()

    if args.mode == "capture":
        output = args.output or str(project_root / "calibration_images")
        from aira.vision.calibrate.capture import run_capture
        count = run_capture(
            output_dir=output,
            width=args.width,
            height=args.height,
            checkerboard_size=tuple(args.checkerboard),
        )
        return 0 if count >= 0 else 1

    if args.mode == "intrinsics":
        input_dir = args.input or str(project_root / "calibration_images")
        output_dir = args.output or input_dir
        import glob
        pattern = str(Path(input_dir) / args.pattern)
        paths = sorted(glob.glob(pattern))
        if not paths:
            paths = sorted(glob.glob(str(Path(input_dir) / "*.jpg")))
        if not paths:
            print(f"No images in {input_dir}/")
            return 1
        from aira.vision.calibrate.intrinsics import calibrate_camera, save_calibration
        results = calibrate_camera(
            paths,
            board_size=tuple(args.checkerboard),
            square_size_mm=args.square_size,
            visualize=args.visualize,
        )
        if results is None:
            return 1
        save_calibration(results, output_dir)
        print("Intrinsics saved.")
        return 0

    if args.mode == "handeye":
        output = args.output or str(project_root / "handeye_calibration_data.json")
        intrinsics_dir = args.intrinsics or str(project_root / "calibration_images")
        from aira.vision.calibrate.handeye import run_handeye_data_collection
        ok = run_handeye_data_collection(
            robot_ip=args.ip,
            aruco_dict_name=args.aruco_dict,
            aruco_size_m=args.aruco_size,
            output_path=output,
            intrinsics_dir=intrinsics_dir,
        )
        return 0 if ok else 1

    if args.mode == "handeye_solve":
        input_path = args.input or str(project_root / "handeye_calibration_data.json")
        output_path = args.output or str(project_root / "configs" / "handeye_calibration_result.json")
        from aira.vision.calibrate.handeye import run_handeye_solve
        ok = run_handeye_solve(
            input_path=input_path,
            output_path=output_path,
            intrinsics_path=args.intrinsics,
            distortion_path=args.distortion,
        )
        return 0 if ok else 1

    if args.mode == "depth":
        from aira.vision.calibrate.depth import run_depth
        return run_depth(
            mode="interactive",
            on_chip=args.on_chip,
            health=args.health,
            reset=args.reset,
            preview=args.preview,
            tare_mm=args.tare,
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
