"""
Dataset definition and config loading for YOLO (train/val path, class names).

Default dataset YAML: configs/dataset.yaml under project root.
Path in YAML is resolved relative to project root when loading.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from aira.utils.paths import get_project_root


def get_dataset_yaml_path(yaml_path: Optional[Path] = None) -> Path:
    """Return path to dataset YAML. Default: project_root/configs/dataset.yaml."""
    root = get_project_root()
    if yaml_path is not None:
        p = Path(yaml_path)
        return root / p if not p.is_absolute() else p
    return root / "configs" / "dataset.yaml"


def load_dataset_config(yaml_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Load dataset YAML (path, train, val, nc, names).
    Resolves 'path' to absolute (relative to project root or YAML dir).
    Returns None if file missing or invalid.
    """
    path = get_dataset_yaml_path(yaml_path)
    if not path.exists():
        return None
    try:
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not data:
            return None
        # Resolve dataset path to absolute
        base = path.parent
        root = get_project_root()
        raw_path = data.get("path", "dataset")
        if isinstance(raw_path, str):
            p = Path(raw_path)
            if not p.is_absolute():
                # Prefer relative to project root so path is portable
                if (root / p).exists():
                    data["path"] = str((root / p).resolve())
                else:
                    data["path"] = str((base / p).resolve())
            else:
                data["path"] = raw_path
        return data
    except Exception:
        return None


def get_class_names(yaml_path: Optional[Path] = None) -> List[str]:
    """Return list of class names from dataset YAML. Empty list if not found."""
    cfg = load_dataset_config(yaml_path)
    if not cfg:
        return []
    names = cfg.get("names")
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    if isinstance(names, list):
        return names
    return []


def get_dataset_path(yaml_path: Optional[Path] = None) -> Optional[Path]:
    """Return resolved dataset directory path from config, or None."""
    cfg = load_dataset_config(yaml_path)
    if not cfg or "path" not in cfg:
        return None
    return Path(cfg["path"])
