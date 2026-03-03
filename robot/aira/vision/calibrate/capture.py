"""
Interactive calibration image capture (checkerboard) using RealSense.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

from aira.vision.calibrate.camera import RealSenseCamera, HAS_REALSENSE
from aira.vision.calibrate.grids import detect_checkerboard

if HAS_REALSENSE:
    import pyrealsense2 as rs


def _compute_depth_stats(depth_frame, roi_size: int = 100) -> Optional[dict]:
    """Depth statistics for center ROI. Returns None if insufficient valid pixels."""
    if not HAS_REALSENSE or depth_frame is None:
        return None
    depth_image = np.asanyarray(depth_frame.get_data())
    h, w = depth_image.shape
    cy, cx = h // 2, w // 2
    half = roi_size // 2
    roi = depth_image[cy - half : cy + half, cx - half : cx + half]
    valid = roi[roi > 0].astype(np.float32)
    if len(valid) < 10:
        return None
    return {
        "mean_mm": float(np.mean(valid)),
        "std_mm": float(np.std(valid)),
        "rms_percent": float(np.std(valid) / np.mean(valid) * 100) if np.mean(valid) > 0 else 0,
    }


def run_capture(
    output_dir: str = "calibration_images",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    checkerboard_size: Tuple[int, int] = (7, 9),
) -> int:
    """
    Run interactive calibration capture loop.
    SPACE=capture, D=toggle depth, C=toggle checkerboard, Q/ESC=quit.
    Returns number of images captured.
    """
    if not HAS_REALSENSE:
        print("Error: pyrealsense2 required for capture.")
        return 0
    os.makedirs(output_dir, exist_ok=True)
    cam = RealSenseCamera(width=width, height=height, fps=fps, color_only=False)
    if not cam.start():
        print("Failed to start RealSense.")
        return 0
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)
    show_depth = True
    show_checkerboard = True
    capture_count = 0

    print("\n" + "=" * 60)
    print("CALIBRATION IMAGE CAPTURE")
    print("=" * 60)
    print("  SPACE - Capture | D - depth | C - checkerboard | Q - Quit")
    print(f"  Saving to: {output_dir}/")
    print("=" * 60)

    cv2.namedWindow("Calibration Capture", cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            color_image, depth_frame = cam.get_frames()
            if color_image is None:
                continue
            display = color_image.copy()
            checkerboard_found = False
            if show_checkerboard:
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                found, corners = detect_checkerboard(gray, checkerboard_size)
                if found and corners is not None:
                    checkerboard_found = True
                    cv2.drawChessboardCorners(
                        display, checkerboard_size, corners, True
                    )
            if show_depth and depth_frame is not None:
                depth_colorized = np.asanyarray(
                    colorizer.colorize(depth_frame).get_data()
                )
                stats = _compute_depth_stats(depth_frame)
                if depth_colorized.shape[:2] != display.shape[:2]:
                    depth_colorized = cv2.resize(
                        depth_colorized, (display.shape[1], display.shape[0])
                    )
                display = np.hstack([display, depth_colorized])
                if stats:
                    y0 = 30
                    cv2.putText(
                        display, f"Depth: {stats['mean_mm']:.0f}mm",
                        (display.shape[1] // 2 + 10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
                    cv2.putText(
                        display, f"Noise: {stats['rms_percent']:.2f}%",
                        (display.shape[1] // 2 + 10, y0 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
            status_color = (0, 255, 0) if checkerboard_found else (0, 0, 255)
            status_text = "Checkerboard: FOUND" if checkerboard_found else "Checkerboard: NOT FOUND"
            cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display, f"Captured: {capture_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(
                display,
                f"[D]epth: {'ON' if show_depth else 'OFF'} | [C]hecker: {'ON' if show_checkerboard else 'OFF'}",
                (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            cv2.imshow("Calibration Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") or key == 32:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                color_path = os.path.join(output_dir, f"calib_{ts}.png")
                cv2.imwrite(color_path, color_image)
                if depth_frame is not None:
                    depth_path = os.path.join(output_dir, f"depth_{ts}.png")
                    depth_image = np.asanyarray(depth_frame.get_data())
                    cv2.imwrite(depth_path, depth_image)
                capture_count += 1
                print(f"Captured #{capture_count}: calib_{ts}.png")
            elif key in (ord("d"), ord("D")):
                show_depth = not show_depth
            elif key in (ord("c"), ord("C")):
                show_checkerboard = not show_checkerboard
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        cam.stop()
    print(f"\nSaved {capture_count} images to {output_dir}/")
    return capture_count
