"""COMPONENTS_STATUS manager for the voice bridge.

Tracks the current state of Voice_Assistant, Server_Connection, and
Robot_Status.  Sends COMPONENTS_STATUS messages to glasses via a
caller-provided async send function.
"""

from __future__ import annotations

import json
from typing import Awaitable, Callable, Optional

from loguru import logger


class StatusManager:
    """Track and broadcast COMPONENTS_STATUS to the XR glasses."""

    def __init__(self, send_fn: Callable[[str, str], Awaitable[bool]]):
        self._send_fn = send_fn
        self.voice_assistant: str = "idle"
        self.server_connection: str = "inactive"
        self.robot_status: str = "N/A"
        self._last_sent: Optional[dict] = None

    def _payload(self) -> dict:
        return {
            "Voice_Assistant": self.voice_assistant,
            "Server_Connection": self.server_connection,
            "Robot_Status": self.robot_status,
        }

    async def update(self, **kwargs) -> dict:
        """Update one or more fields and push to glasses if changed.

        Returns the current payload dict.
        """
        for key in ("voice_assistant", "server_connection", "robot_status"):
            if key in kwargs:
                setattr(self, key, kwargs[key])

        payload = self._payload()
        if payload != self._last_sent:
            sent = await self._push(payload)
            if sent:
                self._last_sent = payload.copy()
        return payload

    async def force_push(self):
        """Push current status regardless of _last_sent (e.g. after reconnect)."""
        payload = self._payload()
        sent = await self._push(payload)
        if sent:
            self._last_sent = payload.copy()
        return payload

    async def _push(self, payload: dict) -> bool:
        """Send COMPONENTS_STATUS to the glasses via the bridge's send function.

        Returns True if the send function reported success.
        """
        try:
            result = await self._send_fn("COMPONENTS_STATUS", json.dumps(payload))
            if result:
                logger.info(f"[Status] Pushed: {payload}")
                return True
            logger.warning(f"[Status] Push deferred (connection not ready): {payload}")
            return False
        except Exception as exc:
            logger.warning(f"[Status] Failed to push status: {exc}")
            return False
