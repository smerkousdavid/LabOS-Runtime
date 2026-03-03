#!/usr/bin/env python3
"""
Test Hand-Eye Calibration: ArUco Center in Tool Frame

Loads the computed T_cam_to_tool and camera intrinsics, connects to the robot
and camera, detects the ArUco marker in live frames, and reports the ArUco
center relative to the tool head origin (mm).

Usage:
    python test_handeye.py --ip 192.168.1.195
    python test_handeye.py -c handeye_calibration_result.json --ip 192.168.1.195 --once
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import robot, camera, and ArUco detection from aira modules
try:
    from aira.vision.calibrate.camera import RealSenseCamera, HAS_REALSENSE
    from aira.vision.calibrate.aruco import ARUCO_DICT, detect_aruco_pose
    from aira.robot import XArmController
    HAS_XARM = XArmController is not None
except ImportError as e:
    print(f"Error: Could not import aira modules: {e}")
    print("Run this script from the version2 directory with PYTHONPATH including version2.")
    sys.exit(1)


def load_calibration(calib_path: str) -> Tuple[np.ndarray, Optional[str], Optional[float]]:
    """
    Load hand-eye calibration result.
    Returns (T_cam_to_tool 4x4, aruco_dict_name or None, aruco_size_m or None).
    """
    with open(calib_path, 'r') as f:
        data = json.load(f)
    T = np.array(data['calibration']['T_cam_to_tool'], dtype=np.float64)
    meta = data.get('metadata', {})
    aruco_dict = meta.get('aruco_dict')
    aruco_size_m = meta.get('aruco_size_m')
    return T, aruco_dict, aruco_size_m


def load_intrinsics(matrix_path: str, distortion_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load K and D from numpy files."""
    K = np.load(matrix_path)
    D = np.load(distortion_path)
    return K, D


def main():
    parser = argparse.ArgumentParser(
        description='Report ArUco center in tool-head frame using hand-eye calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_handeye.py --ip 192.168.1.195
  python test_handeye.py -c handeye_calibration_result.json --ip 192.168.1.195 --once
        """
    )
    base = Path(__file__).resolve().parent
    project_root = base.parent
    parser.add_argument('--calibration', '-c', type=str,
                        default=str(project_root / 'configs' / 'handeye_calibration_result.json'),
                        help='Path to handeye_calibration_result.json')
    parser.add_argument('--ip', type=str, default='192.168.1.195',
                        help='Robot IP address')
    parser.add_argument('--intrinsics', type=str,
                        default=str(project_root / 'configs' / 'calibration_matrix.npy'),
                        help='Path to calibration_matrix.npy')
    parser.add_argument('--distortion', type=str,
                        default=str(project_root / 'configs' / 'distortion_coefficients.npy'),
                        help='Path to distortion_coefficients.npy')
    parser.add_argument('--aruco-dict', type=str, default='DICT_5X5_100',
                        help='ArUco dictionary name (used if not in calibration JSON)')
    parser.add_argument('--aruco-size', type=float, default=0.04,
                        help='ArUco marker size in meters (used if not in calibration JSON)')
    parser.add_argument('--once', action='store_true',
                        help='Run one measurement and exit; otherwise loop until Q')
    args = parser.parse_args()

    # Resolve relative paths against project root
    calib_path = Path(args.calibration)
    if not calib_path.is_absolute():
        calib_path = project_root / calib_path
    matrix_path = Path(args.intrinsics)
    if not matrix_path.is_absolute():
        matrix_path = project_root / matrix_path
    distortion_path = Path(args.distortion)
    if not distortion_path.is_absolute():
        distortion_path = project_root / distortion_path

    print("\n" + "="*60)
    print("HAND-EYE TEST: ArUco center in tool frame")
    print("="*60)

    if not calib_path.exists():
        print(f"Error: Calibration file not found: {calib_path}")
        return 1
    if not matrix_path.exists() or not distortion_path.exists():
        print(f"Error: Intrinsics not found: {matrix_path} or {distortion_path}")
        return 1

    T_cam_to_tool = np.eye(4)
    try:
        T_cam_to_tool, aruco_dict_name, aruco_size_m = load_calibration(str(calib_path))
        if aruco_dict_name is None:
            aruco_dict_name = args.aruco_dict
        if aruco_size_m is None:
            aruco_size_m = args.aruco_size
        print(f"Loaded calibration: {calib_path}")
        print(f"  ArUco: {aruco_dict_name}, size {aruco_size_m*1000:.1f} mm")
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return 1

    try:
        K, D = load_intrinsics(str(matrix_path), str(distortion_path))
        print(f"Loaded intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    except Exception as e:
        print(f"Error loading intrinsics: {e}")
        return 1

    aruco_dict_type = ARUCO_DICT.get(aruco_dict_name, cv2.aruco.DICT_5X5_100)

    if not HAS_XARM:
        print("Error: xArm SDK not available")
        return 1
    if not HAS_REALSENSE:
        print("Error: pyrealsense2 not available")
        return 1

    robot = XArmController(args.ip)
    camera = RealSenseCamera()
    if not robot.connect():
        print("Error: Failed to connect to robot")
        return 1
    if not camera.start():
        robot.disconnect()
        print("Error: Failed to start camera")
        return 1

    try:
        cv2.namedWindow('Hand-Eye Test', cv2.WINDOW_AUTOSIZE)
        loop = not args.once
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to read frame")
                if args.once:
                    break
                continue

            found, center_m, R_marker, normal, rvec, tvec = detect_aruco_pose(
                frame, aruco_dict_type, K, D, aruco_size_m
            )

            display = frame.copy()
            if found and center_m is not None and tvec is not None:
                # ArUco center in camera frame (mm); solvePnP tvec is in meters (object points in m)
                t_mm = np.array(center_m, dtype=np.float64) * 1000.0
                p_cam = np.append(t_mm, 1.0)
                p_tool = T_cam_to_tool @ p_cam
                x_tool, y_tool, z_tool = p_tool[0], p_tool[1], p_tool[2]
                print(f"ArUco center relative to tool head origin: [{x_tool:.2f}, {y_tool:.2f}, {z_tool:.2f}] mm")

                try:
                    cv2.drawFrameAxes(display, K, D, rvec, tvec, aruco_size_m * 0.5)
                except Exception:
                    pass
                label = f"Tool frame: [{x_tool:.1f}, {y_tool:.1f}, {z_tool:.1f}] mm"
                cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "ArUco NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow('Hand-Eye Test', display)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            if args.once:
                break
    finally:
        cv2.destroyAllWindows()
        camera.stop()
        robot.disconnect()

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
