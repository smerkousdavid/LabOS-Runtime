"""Abstract base class for glasses connectors.

Subclasses implement device detection and config writing for different
transport mechanisms (USB file-based, Bluetooth, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional


class GlassConnector(ABC):
    """Interface for connecting to and configuring XR glasses."""

    CONFIG_PATH = "Android/data/com.viture.xr.labcapture/files/Config/sop_config.json"

    @abstractmethod
    def detect_device(self) -> Optional[str]:
        """Detect a connected glasses device.

        Returns the mount path / device identifier, or None if not found.
        """

    @abstractmethod
    def write_config(self, ip: str, port: int) -> bool:
        """Write runtime connection config to the glasses.

        Returns True on success.
        """

    @abstractmethod
    def read_config(self) -> Optional[dict]:
        """Read current config from connected glasses.

        Returns the parsed config dict, or None on failure.
        """
