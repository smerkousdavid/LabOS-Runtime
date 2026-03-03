#!/usr/bin/env python3
"""
Launcher for YOLO test (RealSense / webcam). Runs aira.vision.test_yolo.
Use from version2: python test_yolo.py [args]
Or: PYTHONPATH=. python scripts/test_yolo.py [args]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aira.vision.test_yolo import main

if __name__ == "__main__":
    sys.exit(main())
