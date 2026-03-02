"""COMPONENTS_STATUS manager for the voice bridge.

Tracks the current state of Voice_Assistant, Server_Connection, and
Robot_Status.  Sends COMPONENTS_STATUS messages to glasses via the
dashboard API whenever state changes.
"""

from __future__ import annotations

import json
from typing import Optional

from loguru import logger


class StatusManager:
    """Track and broadcast COMPONENTS_STATUS to the XR glasses."""

    def __init__(self, dashboard_url: str):
        self._dashboard_url = dashboard_url
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
            await self._push(payload)
            self._last_sent = payload.copy()
        return payload

    async def _push(self, payload: dict):
        """POST COMPONENTS_STATUS to the dashboard."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                await client.post(
                    f"{self._dashboard_url}/api/send_message",
                    json={
                        "message_type": "COMPONENTS_STATUS",
                        "payload": json.dumps(payload),
                    },
                )
            logger.debug(f"[Status] Pushed: {payload}")
        except Exception as exc:
            logger.warning(f"[Status] Failed to push status: {exc}")
