"""
Camera intrinsics calibration from checkerboard images.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2

from aira.vision.calibrate.grids import find_checkerboard_corners


def calibrate_camera(
    image_paths: List[Union[str, Path]],
    board_size: Tuple[int, int],
    square_size_mm: float = 20.0,
    visualize: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Perform camera calibration from a set of checkerboard images.

    Args:
        image_paths: List of image file paths
        board_size: (columns, rows) inner corners of checkerboard
        square_size_mm: Size of one square in mm
        visualize: Whether to show detection results

    Returns:
        Dict with camera_matrix, dist_coeffs, rms_error, mean_reprojection_error,
        per_image_errors, image_size, num_images, board_size, square_size (in m).
        None if too few successful images.
    """
    square_size_m = square_size_mm / 1000.0
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    object_points: List[np.ndarray] = []
    image_points: List[np.ndarray] = []
    image_size = None
    successful_images: List[str] = []

    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(str(img_path))
        found, corners, img_shape = find_checkerboard_corners(
            img_path, board_size, visualize
        )
        if found:
            object_points.append(objp)
            image_points.append(corners)
            successful_images.append(filename)
            image_size = img_shape

    if visualize:
        cv2.destroyAllWindows()

    if len(object_points) < 3:
        print("\nError: Need at least 3 images with detected checkerboard")
        return None

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    total_error = 0.0
    errors: List[Tuple[str, float]] = []
    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        err = cv2.norm(image_points[i], projected, cv2.NORM_L2) / len(projected)
        errors.append((successful_images[i], float(err)))
        total_error += err
    mean_error = total_error / len(object_points)

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "rms_error": ret,
        "mean_reprojection_error": mean_error,
        "per_image_errors": errors,
        "image_size": image_size,
        "num_images": len(object_points),
        "board_size": board_size,
        "square_size": square_size_m,
        "square_size_mm": square_size_mm,
    }


def save_calibration(
    results: Dict[str, Any], output_dir: Union[str, Path]
) -> Tuple[str, str, str]:
    """
    Save calibration results to output_dir.
    Returns (matrix_path, dist_path, summary_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = str(output_dir / "calibration_matrix.npy")
    dist_path = str(output_dir / "distortion_coefficients.npy")
    np.save(matrix_path, results["camera_matrix"])
    np.save(dist_path, results["dist_coeffs"])

    summary_path = str(output_dir / "intrinsics.txt")
    K = results["camera_matrix"]
    D = results["dist_coeffs"].flatten()
    with open(summary_path, "w") as f:
        f.write("Camera Calibration Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Image size: {results['image_size'][0]} x {results['image_size'][1]}\n")
        f.write(
            f"Checkerboard: {results['board_size'][0]} x {results['board_size'][1]} inner corners\n"
        )
        f.write(f"Square size: {results.get('square_size_mm', results['square_size']*1000):.2f} mm\n")
        f.write(f"Number of images: {results['num_images']}\n\n")
        f.write("Intrinsic Matrix (K):\n")
        f.write("-" * 30 + "\n")
        for row in K:
            f.write(f"  [{row[0]:12.4f}, {row[1]:12.4f}, {row[2]:12.4f}]\n")
        f.write("\nCamera Parameters:\n")
        f.write("-" * 30 + "\n")
        f.write(f"  fx: {K[0,0]:.4f}  fy: {K[1,1]:.4f}  cx: {K[0,2]:.4f}  cy: {K[1,2]:.4f}\n\n")
        f.write("Distortion Coefficients:\n")
        f.write("-" * 30 + "\n")
        f.write(f"  k1: {D[0]:.6f}  k2: {D[1]:.6f}  p1: {D[2]:.6f}  p2: {D[3]:.6f}")
        if len(D) > 4:
            f.write(f"  k3: {D[4]:.6f}")
        f.write("\n\nCalibration Quality:\n")
        f.write("-" * 30 + "\n")
        f.write(f"  RMS reprojection error: {results['rms_error']:.4f} pixels\n")
        f.write(f"  Mean reprojection error: {results['mean_reprojection_error']:.4f} pixels\n")
        f.write("\nPer-image errors:\n")
        for name, err in results["per_image_errors"]:
            f.write(f"  {name}: {err:.4f}\n")
    return matrix_path, dist_path, summary_path
