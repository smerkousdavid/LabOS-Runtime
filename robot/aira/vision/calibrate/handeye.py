"""
Hand-eye calibration: data collection (ArUco + robot) and solver (T_cam_to_tool).
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2

from aira.utils.math import (
    pose_to_matrix,
    matrix_to_pose,
    tool_delta_from_current_to_target,
)
from aira.vision.calibrate.camera import RealSenseCamera, HAS_REALSENSE
from aira.vision.calibrate.aruco import ARUCO_DICT, detect_aruco_pose
from aira.vision.calibrate.grids import generate_checkerboard_points

try:
    from aira.robot import XArmController
except ImportError:
    XArmController = None

try:
    from pynput import keyboard
    USE_PYNPUT = True
except ImportError:
    USE_PYNPUT = False


def wait_for_keypress(message: str = "Press SPACE to continue, Q to quit") -> str:
    print(f"\n{message}")
    if USE_PYNPUT:
        result = [None]
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    result[0] = "space"
                    return False
                if getattr(key, "char", None) in ("q", "Q"):
                    result[0] = "q"
                    return False
            except Exception:
                pass
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        return result[0] or "other"
    response = input("Press Enter to continue, or type 'q' to quit: ")
    return "q" if response.strip().lower() == "q" else "space"


def _compute_checkerboard_pose(
    corners_2d: np.ndarray,
    obj_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    corners_2d = np.array(corners_2d, dtype=np.float32).reshape(-1, 1, 2)
    try:
        success, rvec, tvec = cv2.solvePnP(
            obj_points, corners_2d, K, D, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return False, None, None
        R_marker, _ = cv2.Rodrigues(rvec)
        t_marker = tvec.flatten()
        return True, R_marker, t_marker
    except Exception:
        return False, None, None


def hand_eye_calibration(
    tool_poses: List[List[float]],
    marker_poses: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, str]:
    """Solve hand-eye calibration. Returns (T_cam_to_tool 4x4, method_name)."""
    n = len(tool_poses)
    if n < 3:
        raise ValueError(f"Need at least 3 poses, got {n}")
    R_gripper2base = []
    t_gripper2base = []
    for pose in tool_poses:
        H = pose_to_matrix(pose)
        R_gripper2base.append(H[:3, :3])
        t_gripper2base.append(H[:3, 3].reshape(3, 1))
    R_target2cam = [mp[0] for mp in marker_poses]
    t_target2cam = [mp[1].reshape(3, 1) for mp in marker_poses]
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
        (cv2.CALIB_HAND_EYE_PARK, "Park"),
        (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
    ]
    best_result = None
    best_error = float("inf")
    best_method = None
    for method_id, method_name in methods:
        try:
            R_cam2tool, t_cam2tool = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=method_id,
            )
            T = np.eye(4)
            T[:3, :3] = R_cam2tool
            T[:3, 3] = t_cam2tool.flatten()
            err = _compute_verification_error(tool_poses, marker_poses, T)
            if err < best_error:
                best_error = err
                best_result = T
                best_method = method_name
        except Exception:
            continue
    if best_result is None:
        raise RuntimeError("All calibration methods failed")
    return best_result, best_method


def _compute_verification_error(
    tool_poses: List[List[float]],
    marker_poses: List[Tuple[np.ndarray, np.ndarray]],
    T_cam_to_tool: np.ndarray,
) -> float:
    positions_base = []
    for pose, (R_marker, t_marker) in zip(tool_poses, marker_poses):
        H_tool = pose_to_matrix(pose)
        p_cam = np.append(t_marker, 1)
        p_tool = T_cam_to_tool @ p_cam
        p_base = H_tool @ p_tool
        positions_base.append(p_base[:3])
    positions_base = np.array(positions_base)
    mean_pos = np.mean(positions_base, axis=0)
    errors = np.linalg.norm(positions_base - mean_pos, axis=1)
    return float(np.mean(errors))


def verify_calibration(
    tool_poses: List[List[float]],
    marker_poses: List[Tuple[np.ndarray, np.ndarray]],
    T_cam_to_tool: np.ndarray,
) -> Dict[str, Any]:
    positions_base = []
    x_axes_base = []
    z_axes_base = []
    for pose, (R_marker, t_marker) in zip(tool_poses, marker_poses):
        H_tool = pose_to_matrix(pose)
        p_cam = np.append(t_marker, 1)
        p_tool = T_cam_to_tool @ p_cam
        p_base = H_tool @ p_tool
        positions_base.append(p_base[:3])
        R_cam_to_tool = T_cam_to_tool[:3, :3]
        R_tool_to_base = H_tool[:3, :3]
        R_marker_base = R_tool_to_base @ R_cam_to_tool @ R_marker
        x_axes_base.append(R_marker_base[:, 0])
        z_axes_base.append(R_marker_base[:, 2])
    positions_base = np.array(positions_base)
    mean_pos = np.mean(positions_base, axis=0)
    position_errors = np.linalg.norm(positions_base - mean_pos, axis=1)
    position_error = float(np.mean(position_errors))
    position_std = float(np.std(position_errors))
    x_axes_base = np.array(x_axes_base)
    z_axes_base = np.array(z_axes_base)
    mean_x = np.mean(x_axes_base, axis=0)
    mean_z = np.mean(z_axes_base, axis=0)
    mean_x = mean_x / np.linalg.norm(mean_x)
    mean_z = mean_z / np.linalg.norm(mean_z)
    x_angles = [np.arccos(np.clip(np.dot(ax, mean_x), -1, 1)) * 180 / np.pi for ax in x_axes_base]
    z_angles = [np.arccos(np.clip(np.dot(ax, mean_z), -1, 1)) * 180 / np.pi for ax in z_axes_base]
    rotation_error = float(np.mean(x_angles + z_angles))
    if position_error < 2.0 and rotation_error < 2.0:
        quality = "EXCELLENT"
    elif position_error < 5.0 and rotation_error < 5.0:
        quality = "GOOD"
    elif position_error < 10.0 and rotation_error < 10.0:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"
    return {
        "position_error_mm": position_error,
        "position_std_mm": position_std,
        "rotation_error_deg": rotation_error,
        "checkerboard_position_base_mm": mean_pos.tolist(),
        "quality": quality,
    }


def run_handeye_data_collection(
    robot_ip: str,
    aruco_dict_name: str = "DICT_5X5_100",
    aruco_size_m: float = 0.04,
    output_path: Union[str, Path] = "handeye_calibration_data.json",
    intrinsics_dir: Union[str, Path] = "calibration_images",
) -> bool:
    """
    Run full hand-eye data collection: setup, z0, find height, manual traversal, save.
    Returns True on success.
    """
    if not HAS_REALSENSE:
        print("Error: pyrealsense2 required.")
        return False
    if XArmController is None:
        print("Error: aira.robot.XArmController not available.")
        return False
    intrinsics_dir = Path(intrinsics_dir)
    output_path = Path(output_path)
    K_path = intrinsics_dir / "calibration_matrix.npy"
    D_path = intrinsics_dir / "distortion_coefficients.npy"
    if not K_path.exists() or not D_path.exists():
        print(f"Error: Intrinsics not found in {intrinsics_dir}/")
        return False
    K = np.load(str(K_path))
    D = np.load(str(D_path))
    aruco_dict_type = ARUCO_DICT.get(aruco_dict_name, cv2.aruco.DICT_5X5_100)
    cam = RealSenseCamera(color_only=True)
    if not cam.start():
        return False
    robot = XArmController(robot_ip)
    if not robot.connect():
        cam.stop()
        return False
    z0_position = None
    start_position = None
    samples: List[Dict[str, Any]] = []
    try:
        print("\n" + "=" * 60)
        print("PHASE 1: Z=0 CALIBRATION")
        print("=" * 60)
        key = wait_for_keypress("Place ArUco on table. SPACE when ready, Q to quit")
        if key == "q":
            return False
        robot.close_gripper()
        time.sleep(1)
        robot.set_manual_mode()
        key = wait_for_keypress("Move gripper to table (Z=0). SPACE when ready, Q to quit")
        if key == "q":
            return False
        code, z0_position = robot.get_position()
        if code != 0:
            print("Failed to get position")
            return False
        print("\nPHASE 2: FIND STARTING HEIGHT")
        robot.set_position_mode()
        time.sleep(0.5)
        step_mm = 15
        max_steps = 30
        cv2.namedWindow("Calibration Preview", cv2.WINDOW_AUTOSIZE)
        for step in range(1, max_steps + 1):
            code = robot.set_tool_position(0, 0, -step_mm, 0, 0, 0, speed=30, wait=True)
            if code != 0:
                robot.clear_error()
                continue
            time.sleep(0.3)
            ok, frame = cam.read()
            if not ok or frame is None:
                continue
            found, center_m, R_m, normal, rvec, tvec = detect_aruco_pose(
                frame, aruco_dict_type, K, D, aruco_size_m
            )
            disp = frame.copy()
            if found:
                try:
                    cv2.drawFrameAxes(disp, K, D, rvec, tvec, aruco_size_m * 0.5)
                except Exception:
                    pass
            cv2.putText(
                disp, "ArUco FOUND" if found else "ArUco NOT FOUND",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2,
            )
            cv2.imshow("Calibration Preview", disp)
            cv2.waitKey(100)
            if found:
                break
        cv2.destroyAllWindows()
        code, start_position = robot.get_position()
        if code != 0:
            return False
        print("\nPHASE 3: MANUAL DATA COLLECTION")
        robot.set_manual_mode()
        time.sleep(0.5)
        cv2.namedWindow("Calibration Capture", cv2.WINDOW_AUTOSIZE)
        prev_pose = list(start_position)
        successful = 0
        try:
            while True:
                ok, frame = cam.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                found, center_m, R_marker, normal, rvec, tvec = detect_aruco_pose(
                    frame, aruco_dict_type, K, D, aruco_size_m
                )
                disp = frame.copy()
                if found:
                    try:
                        cv2.drawFrameAxes(disp, K, D, rvec, tvec, aruco_size_m * 0.5)
                    except Exception:
                        pass
                cv2.putText(
                    disp, f"ArUco DETECTED - Sample {successful}" if found else "ArUco NOT DETECTED",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2,
                )
                cv2.putText(disp, "SPACE = record | Q = quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Calibration Capture", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key != ord(" ") and key != 32:
                    continue
                code, actual_pose = robot.get_position()
                if code != 0:
                    continue
                actual_pose = list(actual_pose)
                # Delta in tool-head frame (not base): [dx, dy, dz, droll, dpitch, dyaw]
                tool_delta = tool_delta_from_current_to_target(prev_pose, actual_pose)
                if not found:
                    continue
                t_mm = (np.array(center_m) * 1000).tolist()
                sample = {
                    "index": successful,
                    "toolhead_pose": [float(p) for p in actual_pose],
                    "tool_delta": [float(x) for x in tool_delta],
                    "aruco_found": True,
                    "aruco_pose_camera": {"R": R_marker.tolist(), "t_mm": t_mm},
                    "timestamp": datetime.now().isoformat(),
                }
                samples.append(sample)
                successful += 1
                prev_pose = list(actual_pose)
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
        if successful == 0:
            print("No samples recorded.")
            return False
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "robot_ip": robot_ip,
                "aruco_dict": aruco_dict_name,
                "aruco_size_m": aruco_size_m,
                "z0_reference": [float(p) for p in z0_position] if z0_position else None,
                "start_position": [float(p) for p in start_position] if start_position else None,
                "total_samples": len(samples),
                "camera_resolution": [cam.width, cam.height],
            },
            "samples": samples,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(samples)} samples to {output_path}")
        return True
    finally:
        cam.stop()
        if start_position and robot.arm:
            robot.set_position_mode()
            robot.move_to_absolute(
                start_position[0], start_position[1], start_position[2],
                start_position[3], start_position[4], start_position[5],
                speed=30, wait=True,
            )
        robot.disconnect()


def run_handeye_solve(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    intrinsics_path: Optional[Union[str, Path]] = None,
    distortion_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Load handeye_calibration_data.json, solve T_cam_to_tool, verify, save result.
    Returns True on success.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    with open(input_path, "r") as f:
        calib_data = json.load(f)
    meta = calib_data["metadata"]
    samples = calib_data["samples"]
    if intrinsics_path is None:
        intrinsics_dir = input_path.parent / "calibration_images"
        intrinsics_path = intrinsics_dir / "calibration_matrix.npy"
        distortion_path = intrinsics_dir / "distortion_coefficients.npy"
    else:
        intrinsics_path = Path(intrinsics_path)
        distortion_path = Path(distortion_path or str(intrinsics_path).replace("calibration_matrix.npy", "distortion_coefficients.npy"))
    K = np.load(str(intrinsics_path))
    D = np.load(str(distortion_path))
    tool_poses: List[List[float]] = []
    marker_poses: List[Tuple[np.ndarray, np.ndarray]] = []
    if samples and "aruco_pose_camera" in samples[0]:
        # Reconstruct gripper poses from tool-frame deltas (rotations around tool origin)
        use_tool_delta = len(samples) > 1 and "tool_delta" in samples[1]
        if use_tool_delta:
            H_recon = [pose_to_matrix(samples[0]["toolhead_pose"])]
            for i in range(1, len(samples)):
                H_recon.append(H_recon[i - 1] @ pose_to_matrix(samples[i]["tool_delta"]))
        else:
            H_recon = None
        for i, sample in enumerate(samples):
            if not sample.get("aruco_found") or "aruco_pose_camera" not in sample:
                continue
            ap = sample["aruco_pose_camera"]
            R_marker = np.array(ap["R"])
            t_marker = np.array(ap["t_mm"])
            if use_tool_delta and H_recon is not None:
                tool_poses.append(matrix_to_pose(H_recon[i]))
            else:
                tool_poses.append(sample["toolhead_pose"])
            marker_poses.append((R_marker, t_marker))
    else:
        board_size = tuple(meta["checkerboard_size"])
        square_size_mm = meta["square_size_mm"]
        obj_points = generate_checkerboard_points(board_size, square_size_mm)
        for i, sample in enumerate(samples):
            if "corners_2d" not in sample:
                continue
            corners_2d = np.array(sample["corners_2d"], dtype=np.float32)
            success, R_marker, t_marker = _compute_checkerboard_pose(corners_2d, obj_points, K, D)
            if success and R_marker is not None and t_marker is not None:
                tool_poses.append(sample["toolhead_pose"])
                marker_poses.append((R_marker, t_marker))
    if len(tool_poses) < 3:
        print("Need at least 3 valid samples.")
        return False
    T_cam_to_tool, method = hand_eye_calibration(tool_poses, marker_poses)
    R_cam_to_tool = T_cam_to_tool[:3, :3]
    t_cam_to_tool = T_cam_to_tool[:3, 3]
    from aira.utils.math import euler_from_rotation_matrix
    roll, pitch, yaw = euler_from_rotation_matrix(R_cam_to_tool, degrees=True)
    verification = verify_calibration(tool_poses, marker_poses, T_cam_to_tool)
    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_file": str(input_path),
            "num_samples": len(tool_poses),
            "method": method,
            "aruco_dict": meta.get("aruco_dict"),
            "aruco_size_m": meta.get("aruco_size_m"),
        },
        "calibration": {
            "T_cam_to_tool": T_cam_to_tool.tolist(),
            "translation_mm": t_cam_to_tool.tolist(),
            "rotation_euler_deg": [roll, pitch, yaw],
            "rotation_matrix": R_cam_to_tool.tolist(),
        },
        "verification": verification,
        "source_data": {
            "z0_reference": meta.get("z0_reference"),
            "start_position": meta.get("start_position"),
            "camera_resolution": meta.get("camera_resolution"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {output_path}")
    return True
