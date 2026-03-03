"""
RealSense on-device depth calibration (on-chip, tare, reset, health, preview).
"""

import json
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    rs = None

ON_CHIP_CALIB_CONFIG = {
    "calib type": 0,
    "speed": 3,
    "scan parameter": 0,
    "adjust both sides": 0,
    "white wall mode": 0,
    "host assistance": 0,
}

TARE_CALIB_CONFIG = {
    "average step count": 20,
    "step count": 10,
    "accuracy": 2,
    "scan parameter": 0,
    "data sampling": 0,
}


def compute_depth_noise(
    depth_frame, roi_size: int = 100
) -> Dict[str, Any]:
    """
    Compute depth noise statistics from a depth frame (center ROI).
    Returns dict with valid, mean_mm, std_mm, rms_noise_percent, etc.
    """
    if depth_frame is None:
        return {"valid": False, "message": "No depth frame"}
    depth_image = np.asanyarray(depth_frame.get_data())
    h, w = depth_image.shape
    cy, cx = h // 2, w // 2
    half = roi_size // 2
    roi = depth_image[cy - half : cy + half, cx - half : cx + half]
    valid_depths = roi[roi > 0].astype(np.float32)
    if len(valid_depths) < 10:
        return {"valid": False, "message": "Not enough valid depth pixels"}
    mean_depth = float(np.mean(valid_depths))
    std_depth = float(np.std(valid_depths))
    rms_noise_percent = (std_depth / mean_depth * 100) if mean_depth > 0 else 0
    return {
        "valid": True,
        "mean_mm": mean_depth,
        "std_mm": std_depth,
        "min_mm": float(np.min(valid_depths)),
        "max_mm": float(np.max(valid_depths)),
        "rms_noise_percent": rms_noise_percent,
        "valid_pixels": len(valid_depths),
        "total_pixels": roi_size * roi_size,
    }


def progress_callback(progress: float) -> None:
    bar_length = 40
    filled = int(bar_length * progress)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"\r  Progress: [{bar}] {progress*100:.1f}%", end="", flush=True)


def run_on_chip_calibration(
    device, pipeline, timeout_ms: int = 120000
) -> Tuple[bool, Optional[Any], Optional[float]]:
    """Run on-chip calibration. Returns (success, new_calibration_table, health)."""
    if not HAS_REALSENSE:
        return False, None, None
    try:
        auto_calib = device.as_auto_calibrated_device()
        print("\n" + "=" * 60)
        print("ON-CHIP CALIBRATION")
        print("=" * 60)
        print("Point at FLAT, TEXTURED surface. Keep camera STILL.")
        print("=" * 60)
        input("\nPress ENTER when ready...")
        streaming_active = True
        stream_error = None

        def keep_streaming():
            nonlocal streaming_active, stream_error
            try:
                while streaming_active:
                    pipeline.wait_for_frames(timeout_ms=100)
                    time.sleep(0.05)
            except Exception as e:
                stream_error = e
                streaming_active = False

        stream_thread = threading.Thread(target=keep_streaming, daemon=True)
        stream_thread.start()
        time.sleep(0.2)
        config_json = json.dumps(ON_CHIP_CALIB_CONFIG)
        print("\nRunning on-chip calibration (30-60 s)...")
        try:
            new_table, health = auto_calib.run_on_chip_calibration(
                config_json, progress_callback, timeout_ms
            )
            print()
            return True, new_table, health
        finally:
            streaming_active = False
            stream_thread.join(timeout=1.0)
    except Exception as e:
        print(f"\nCalibration failed: {e}")
        return False, None, None


def run_tare_calibration(
    device, pipeline, ground_truth_mm: float, timeout_ms: int = 120000
) -> Tuple[bool, Optional[Any], Optional[float]]:
    """Run tare calibration at given distance (mm). Returns (success, new_table, health)."""
    if not HAS_REALSENSE:
        return False, None, None
    try:
        auto_calib = device.as_auto_calibrated_device()
        print("\n" + "=" * 60)
        print("TARE CALIBRATION")
        print("=" * 60)
        print(f"Ground truth distance: {ground_truth_mm} mm")
        print("Point at FLAT surface at EXACTLY that distance.")
        print("=" * 60)
        input("\nPress ENTER when ready...")
        streaming_active = True
        stream_error = None

        def keep_streaming():
            nonlocal streaming_active, stream_error
            try:
                while streaming_active:
                    pipeline.wait_for_frames(timeout_ms=100)
                    time.sleep(0.05)
            except Exception as e:
                stream_error = e
                streaming_active = False

        stream_thread = threading.Thread(target=keep_streaming, daemon=True)
        stream_thread.start()
        time.sleep(0.2)
        config_json = json.dumps(TARE_CALIB_CONFIG)
        print("\nRunning tare calibration...")
        try:
            new_table, health = auto_calib.run_tare_calibration(
                ground_truth_mm, config_json, progress_callback, timeout_ms
            )
            print()
            return True, new_table, health
        finally:
            streaming_active = False
            stream_thread.join(timeout=1.0)
    except Exception as e:
        print(f"\nCalibration failed: {e}")
        return False, None, None


def write_calibration(device, calibration_table) -> bool:
    """Write calibration table to device EEPROM."""
    if not HAS_REALSENSE:
        return False
    try:
        auto_calib = device.as_auto_calibrated_device()
        print("\nWARNING: This will permanently update device calibration!")
        confirm = input("Type 'YES' to confirm: ")
        if confirm.strip().upper() != "YES":
            print("Cancelled.")
            return False
        auto_calib.set_calibration_table(calibration_table)
        auto_calib.write_calibration()
        print("Calibration written successfully.")
        return True
    except Exception as e:
        print(f"Failed to write: {e}")
        return False


def reset_to_factory(device) -> bool:
    """Reset device to factory calibration."""
    if not HAS_REALSENSE:
        return False
    try:
        auto_calib = device.as_auto_calibrated_device()
        print("\nWARNING: This will restore factory calibration!")
        confirm = input("Type 'RESET' to confirm: ")
        if confirm.strip().upper() != "RESET":
            print("Cancelled.")
            return False
        auto_calib.reset_to_factory_calibration()
        print("Device reset to factory calibration.")
        return True
    except Exception as e:
        print(f"Failed to reset: {e}")
        return False


def live_depth_preview(pipeline, duration_seconds: int = 0) -> None:
    """Show live depth preview with noise stats. duration_seconds=0 means until 'q'."""
    if not HAS_REALSENSE:
        return
    colorizer = rs.colorizer()
    cv2.namedWindow("Depth Preview", cv2.WINDOW_AUTOSIZE)
    start = time.time()
    try:
        while True:
            if duration_seconds > 0 and (time.time() - start) > duration_seconds:
                break
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame:
                continue
            noise_stats = compute_depth_noise(depth_frame)
            depth_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.resize(
                    color_image, (depth_colorized.shape[1], depth_colorized.shape[0])
                )
                display = np.hstack([color_image, depth_colorized])
            else:
                display = depth_colorized
            if noise_stats.get("valid"):
                y0 = 30
                cv2.putText(
                    display, f"Mean: {noise_stats['mean_mm']:.1f}mm",
                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.putText(
                    display, f"RMS Noise: {noise_stats['rms_noise_percent']:.2f}%",
                    (10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
            cv2.putText(display, "Press 'q' to quit", (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Depth Preview", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


def run_depth(
    mode: str = "interactive",
    tare_mm: Optional[float] = None,
    on_chip: bool = False,
    health: bool = False,
    reset: bool = False,
    preview: bool = False,
) -> int:
    """
    Run depth calibration. Modes: interactive, on_chip, tare, health, reset, preview.
    Returns exit code (0 = success).
    """
    if not HAS_REALSENSE:
        print("Error: pyrealsense2 required.")
        return 1
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Failed to start camera: {e}")
        return 1
    device = profile.get_device()
    pending_table = None
    try:
        if health:
            try:
                auto_calib = device.as_auto_calibrated_device()
                _ = auto_calib.get_calibration_table()
                print("Calibration table retrieved.")
            except Exception as e:
                print(f"Health check: {e}")
            return 0
        if preview:
            live_depth_preview(pipeline, duration_seconds=0)
            return 0
        if on_chip:
            success, table, health_val = run_on_chip_calibration(device, pipeline)
            if success and table is not None:
                pending_table = table
                print("Use option 4 in interactive to write to device.")
            return 0
        if tare_mm is not None and tare_mm > 0:
            success, table, _ = run_tare_calibration(device, pipeline, tare_mm)
            if success and table is not None:
                pending_table = table
            return 0
        if reset:
            reset_to_factory(device)
            return 0
        # Interactive
        while True:
            print("\n" + "=" * 60)
            print("DEPTH CALIBRATION")
            print("=" * 60)
            print("1. Preview  2. On-chip  3. Tare  4. Write pending  5. Reset  6. Exit")
            if pending_table is not None:
                print("  ** Pending calibration ready to write **")
            print("=" * 60)
            choice = input("Choice (1-6): ").strip()
            if choice == "1":
                live_depth_preview(pipeline, duration_seconds=0)
            elif choice == "2":
                success, table, _ = run_on_chip_calibration(device, pipeline)
                if success and table is not None:
                    pending_table = table
            elif choice == "3":
                try:
                    dist = float(input("Ground truth distance (mm): "))
                    if dist > 0:
                        success, table, _ = run_tare_calibration(device, pipeline, dist)
                        if success and table is not None:
                            pending_table = table
                except ValueError:
                    print("Invalid number.")
            elif choice == "4":
                if pending_table is not None:
                    if write_calibration(device, pending_table):
                        pending_table = None
                else:
                    print("No pending calibration.")
            elif choice == "5":
                reset_to_factory(device)
                pending_table = None
            elif choice == "6":
                break
        return 0
    finally:
        pipeline.stop()
