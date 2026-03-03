"""
Shared utilities: project root and pose/rotation math.
"""

from aira.utils.paths import get_project_root
from aira.utils.math import (
    rotation_matrix_euler,
    euler_from_rotation_matrix,
    pose_to_matrix,
    matrix_to_pose,
    target_pose_from_start,
    tool_delta_from_current_to_target,
)

__all__ = [
    "get_project_root",
    "rotation_matrix_euler",
    "euler_from_rotation_matrix",
    "pose_to_matrix",
    "matrix_to_pose",
    "target_pose_from_start",
    "tool_delta_from_current_to_target",
]
