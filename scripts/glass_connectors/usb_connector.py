"""USB/MTP file-based connector for VITURE XR glasses.

Detects glasses mounted as USB storage and writes sop_config.json
to the Android data directory.
"""

import json
import os
import platform
import time
from pathlib import Path
from typing import Optional

from .base import GlassConnector


class USBConnector(GlassConnector):
    """Detect USB-mounted VITURE glasses and write config via filesystem."""

    VITURE_MARKER = "Android/data/com.viture.xr.labcapture"

    def _get_mount_search_paths(self) -> list[Path]:
        system = platform.system()
        paths = []
        if system == "Linux":
            user = os.environ.get("USER", "")
            paths.append(Path(f"/media/{user}"))
            paths.append(Path(f"/run/media/{user}"))
            paths.append(Path("/mnt"))
        elif system == "Darwin":
            paths.append(Path("/Volumes"))
        elif system == "Windows":
            import string
            for letter in string.ascii_uppercase:
                drive = Path(f"{letter}:\\")
                if drive.exists() and drive != Path("C:\\"):
                    paths.append(drive)
        return paths

    def detect_device(self) -> Optional[str]:
        for search_root in self._get_mount_search_paths():
            if not search_root.exists():
                continue
            try:
                for entry in search_root.iterdir():
                    if entry.is_dir():
                        marker = entry / self.VITURE_MARKER
                        if marker.exists():
                            return str(entry)
            except PermissionError:
                continue
        return None

    def _config_path(self, mount: str) -> Path:
        return Path(mount) / self.CONFIG_PATH

    def write_config(self, ip: str, port: int, mount: Optional[str] = None) -> bool:
        if mount is None:
            mount = self.detect_device()
        if mount is None:
            return False

        config_dir = self._config_path(mount).parent
        config_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "XR_AI_Runtime_IP": ip,
            "XR_AI_Runtime_PORT": str(port),
        }

        config_file = self._config_path(mount)
        config_file.write_text(json.dumps(config, indent=4))
        return True

    def read_config(self, mount: Optional[str] = None) -> Optional[dict]:
        if mount is None:
            mount = self.detect_device()
        if mount is None:
            return None

        config_file = self._config_path(mount)
        if not config_file.exists():
            return None

        try:
            return json.loads(config_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def wait_for_device(self, timeout: float = 120, poll_interval: float = 2.0) -> Optional[str]:
        """Poll for a glasses device to appear. Returns mount path or None on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            mount = self.detect_device()
            if mount is not None:
                return mount
            time.sleep(poll_interval)
        return None
