"""
Checkerboard detection and 3D point generation for intrinsics and hand-eye.
"""

import os
from typing import Tuple, Union, List, Optional

import numpy as np
import cv2


def find_checkerboard_corners(
    image_path: Union[str, os.PathLike],
    board_size: Tuple[int, int],
    visualize: bool = False,
) -> Tuple[bool, Optional[np.ndarray], Optional[Tuple[int, int]]]:
    """
    Find checkerboard corners in an image.

    Args:
        image_path: Path to the image file
        board_size: (columns, rows) inner corners of checkerboard
        visualize: Whether to show the detection result

    Returns:
        (success, corners, image_shape) where image_shape is (width, height)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Warning: Could not read {image_path}")
        return False, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if visualize:
            vis_img = img.copy()
            cv2.drawChessboardCorners(vis_img, board_size, corners, found)
            max_dim = 800
            h, w = vis_img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                vis_img = cv2.resize(vis_img, (int(w * scale), int(h * scale)))
            cv2.imshow("Checkerboard Detection", vis_img)
            key = cv2.waitKey(500)
            if key == 27:
                cv2.destroyAllWindows()
                return found, corners, gray.shape[::-1]
    return found, corners, gray.shape[::-1]


def generate_checkerboard_points(
    board_size: Tuple[int, int], square_size_mm: float
) -> np.ndarray:
    """
    Generate 3D object points for checkerboard corners (Z=0 plane).

    Args:
        board_size: (columns, rows) inner corners
        square_size_mm: Size of each square in mm

    Returns:
        Nx3 array of 3D points in checkerboard frame (mm)
    """
    cols, rows = board_size
    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_points *= square_size_mm
    return obj_points


def detect_checkerboard(
    gray_image: np.ndarray, board_size: Tuple[int, int]
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Detect checkerboard corners in a grayscale image (no file I/O).
    Used by capture preview.

    Returns:
        (found, corners)
    """
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray_image, board_size, flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
    return found, corners
