#!/usr/bin/env python3
"""
Train YOLO (detect or segment). Run from version2 with: PYTHONPATH=. python scripts/train.py [args]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aira.vision.train_yolo import main

if __name__ == "__main__":
    sys.exit(main())
