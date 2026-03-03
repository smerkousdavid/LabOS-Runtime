"""
Pose and rotation helpers (ZYX convention, mm/degrees).
Used by hand-eye calibration and robot control.
"""

import math
from typing import List, Tuple

import numpy as np


def rotation_matrix_euler(
    roll: float, pitch: float, yaw: float, degrees: bool = True
) -> np.ndarray:
    """Build 3x3 rotation matrix from roll, pitch, yaw (ZYX convention)."""
    if degrees:
        r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
    else:
        r, p, y = roll, pitch, yaw
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def euler_from_rotation_matrix(
    R: np.ndarray, degrees: bool = True
) -> Tuple[float, float, float]:
    """Extract roll, pitch, yaw from 3x3 rotation matrix (ZYX convention)."""
    if abs(R[2, 0]) >= 1.0 - 1e-6:
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = math.pi / 2
            roll = math.atan2(R[0, 1], R[0, 2])
        else:
            pitch = -math.pi / 2
            roll = math.atan2(-R[0, 1], -R[0, 2])
    else:
        pitch = math.asin(-R[2, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    if degrees:
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
    return roll, pitch, yaw


def pose_to_matrix(pose: List[float]) -> np.ndarray:
    """[x, y, z, roll, pitch, yaw] (mm, deg) -> 4x4 homogeneous (base frame)."""
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = rotation_matrix_euler(pose[3], pose[4], pose[5], degrees=True)
    H[:3, 3] = [pose[0], pose[1], pose[2]]
    return H


def matrix_to_pose(H: np.ndarray) -> List[float]:
    """4x4 homogeneous matrix -> [x, y, z, roll, pitch, yaw] (mm, degrees)."""
    R = H[:3, :3]
    t = H[:3, 3]
    roll, pitch, yaw = euler_from_rotation_matrix(R, degrees=True)
    return [float(t[0]), float(t[1]), float(t[2]), roll, pitch, yaw]


def target_pose_from_start(
    start_pose: List[float],
    dx: float, dy: float, dz: float,
    drx: float, dry: float, drz: float,
) -> List[float]:
    """
    Target pose in base frame = start_pose + (dx,dy,dz,drx,dry,drz) in start's tool frame.
    Returns [x, y, z, roll, pitch, yaw] in base frame (mm, deg).
    """
    H_start = pose_to_matrix(start_pose)
    R_start = H_start[:3, :3]
    t_start = H_start[:3, 3]
    offset_base = R_start @ np.array([dx, dy, dz], dtype=np.float64)
    t_target = t_start + offset_base
    R_delta = rotation_matrix_euler(drx, dry, drz, degrees=True)
    R_target = R_start @ R_delta
    roll, pitch, yaw = euler_from_rotation_matrix(R_target, degrees=True)
    return [float(t_target[0]), float(t_target[1]), float(t_target[2]), roll, pitch, yaw]


def tool_delta_from_current_to_target(
    current_pose: List[float], target_pose: List[float]
) -> List[float]:
    """
    Delta in current tool frame to go from current_pose to target_pose.
    Returns [dx, dy, dz, droll, dpitch, dyaw] for set_tool_position (mm, deg).
    """
    H_cur = pose_to_matrix(current_pose)
    H_tgt = pose_to_matrix(target_pose)
    H_rel = np.linalg.inv(H_cur) @ H_tgt
    dx, dy, dz = H_rel[0, 3], H_rel[1, 3], H_rel[2, 3]
    droll, dpitch, dyaw = euler_from_rotation_matrix(H_rel[:3, :3], degrees=True)
    return [float(dx), float(dy), float(dz), float(droll), float(dpitch), float(dyaw)]
