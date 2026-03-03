"""
Shared RealSense pipeline for calibration (capture, hand-eye, depth).
Color-only or color+depth with resolution fallbacks.
"""

from typing import Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    rs = None

# Default calibration resolution
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30


class RealSenseCamera:
    """
    RealSense camera at a given resolution (color-only or color+depth).
    Use start() then read()/get_frames(); stop() when done.
    """

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        color_only: bool = True,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.color_only = color_only
        self.pipeline: Optional[object] = None
        self.profile: Optional[object] = None
        self.align = None  # rs.align(rs.stream.color) when depth enabled

    def start(self) -> bool:
        """Start pipeline with fallback resolutions. Returns True on success."""
        if not HAS_REALSENSE:
            print("Warning: pyrealsense2 not installed.")
            return False
        if self.pipeline is not None and self.profile is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = rs.pipeline()
        resolutions = [
            (self.width, self.height),
            (1280, 720),
            (1920, 1080),
            (848, 480),
            (640, 480),
        ]
        seen = set()
        unique = []
        for r in resolutions:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        for w, h in unique:
            try:
                config = rs.config()
                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, self.fps)
                if not self.color_only:
                    config.enable_stream(rs.stream.depth, w, h, rs.format.z16, self.fps)
                self.profile = self.pipeline.start(config)
                color_profile = rs.video_stream_profile(
                    self.profile.get_stream(rs.stream.color)
                )
                intr = color_profile.get_intrinsics()
                self.width = intr.width
                self.height = intr.height
                if not self.color_only:
                    self.align = rs.align(rs.stream.color)
                for _ in range(30):
                    self.pipeline.wait_for_frames()
                return True
            except RuntimeError:
                if self.profile is not None:
                    try:
                        self.pipeline.stop()
                    except Exception:
                        pass
                    self.profile = None
                continue
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get color frame. Returns (ok, bgr_image)."""
        if self.pipeline is None or self.profile is None:
            return False, None
        try:
            if self.color_only:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
            else:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
            if not color_frame:
                return False, None
            return True, np.asanyarray(color_frame.get_data())
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get (color_image, depth_frame). depth_frame is None if color_only."""
        if self.pipeline is None or self.profile is None:
            return None, None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            if not self.color_only and self.align is not None:
                frames = self.align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if not self.color_only else None
            color_image = np.asanyarray(color_frame.get_data()) if color_frame else None
            return color_image, depth_frame
        except Exception:
            return None, None

    def stop(self) -> None:
        """Stop pipeline."""
        if self.pipeline is not None and self.profile is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.profile = None
        self.align = None
