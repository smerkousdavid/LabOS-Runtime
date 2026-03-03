"""
Project path utilities. Resolve version2 root for configs and data.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return version2 project root (parent of aira package)."""
    # aira/utils/paths.py -> aira/utils -> aira -> version2
    return Path(__file__).resolve().parent.parent.parent
