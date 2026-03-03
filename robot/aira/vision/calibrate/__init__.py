"""
Vision calibration: intrinsics, capture, hand-eye, depth.
Import submodules as needed: aira.vision.calibrate.capture, handeye, etc.
"""

from aira.vision.calibrate import camera
from aira.vision.calibrate import grids
from aira.vision.calibrate import intrinsics
from aira.vision.calibrate import capture
from aira.vision.calibrate import aruco
from aira.vision.calibrate import handeye
from aira.vision.calibrate import depth

__all__ = [
    "camera",
    "grids",
    "intrinsics",
    "capture",
    "aruco",
    "handeye",
    "depth",
]
