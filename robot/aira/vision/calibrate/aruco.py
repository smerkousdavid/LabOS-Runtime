"""
ArUco marker detection and pose in camera frame.
"""

from typing import Any, Optional, Tuple

import numpy as np
import cv2

ARUCO_DICT = {
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


def _parse_opencv_version(version_string: str) -> Tuple[int, int, int]:
    try:
        parts = version_string.split(".")
        return tuple(int(p) for p in parts[:3])
    except Exception:
        return (0, 0, 0)


def detect_aruco_pose(
    frame: np.ndarray,
    aruco_dict_type: int,
    K: np.ndarray,
    D: np.ndarray,
    marker_size_m: float = 0.04,
) -> Tuple[
    bool,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[Any],
    Optional[Any],
]:
    """
    Detect ArUco marker and return pose in camera frame.

    Returns:
        (success, center_m, R_marker, normal, rvec, tvec).
        center_m and tvec in meters; R_marker is 3x3 marker->camera.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    v = _parse_opencv_version(cv2.__version__)

    if v >= (4, 4, 0):
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        except AttributeError:
            aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    else:
        aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)

    if v >= (4, 7, 0):
        try:
            detector_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=params, cameraMatrix=K, distCoeff=D
            )
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=params, cameraMatrix=K, distCoeff=D
        )

    if ids is None or len(corners) == 0:
        return False, None, None, None, None, None

    marker_corners_2d = np.array(corners[0][0], dtype=np.float32)
    obj_points = np.array(
        [
            [-marker_size_m / 2, marker_size_m / 2, 0],
            [marker_size_m / 2, marker_size_m / 2, 0],
            [marker_size_m / 2, -marker_size_m / 2, 0],
            [-marker_size_m / 2, -marker_size_m / 2, 0],
        ],
        dtype=np.float32,
    )

    try:
        result = cv2.solvePnP(
            obj_points, marker_corners_2d, K, D, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if len(result) == 3:
            retval, rvec, tvec = result
            if not retval:
                return False, None, None, None, None, None
        else:
            rvec, tvec = result
        R_mat, _ = cv2.Rodrigues(rvec)
        center_m = tvec.flatten()
        normal = R_mat[:, 2].flatten()
        return True, center_m, R_mat, normal, rvec, tvec
    except Exception as e:
        print(f"Error in ArUco pose: {e}")
        return False, None, None, None, None, None
