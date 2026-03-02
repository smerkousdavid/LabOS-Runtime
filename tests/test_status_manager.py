"""Unit tests for the StatusManager (mocked HTTP, no live services)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from status_manager import StatusManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sm():
    return StatusManager("http://fake-dashboard:5000")


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_defaults(self, sm):
        assert sm.voice_assistant == "idle"
        assert sm.server_connection == "inactive"
        assert sm.robot_status == "N/A"

    def test_payload_format(self, sm):
        p = sm._payload()
        assert p == {
            "Voice_Assistant": "idle",
            "Server_Connection": "inactive",
            "Robot_Status": "N/A",
        }


# ---------------------------------------------------------------------------
# update() behaviour
# ---------------------------------------------------------------------------

class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_pushes_on_change(self, sm):
        with patch.object(sm, "_push", new_callable=AsyncMock) as mock_push:
            await sm.update(voice_assistant="listening")
            mock_push.assert_called_once()
            payload = mock_push.call_args[0][0]
            assert payload["Voice_Assistant"] == "listening"

    @pytest.mark.asyncio
    async def test_update_dedup_no_push(self, sm):
        """Calling update with same values should not push again."""
        with patch.object(sm, "_push", new_callable=AsyncMock) as mock_push:
            await sm.update(voice_assistant="idle")
            first_count = mock_push.call_count
            await sm.update(voice_assistant="idle")
            assert mock_push.call_count == first_count

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self, sm):
        with patch.object(sm, "_push", new_callable=AsyncMock) as mock_push:
            result = await sm.update(voice_assistant="listening", server_connection="active")
            assert result["Voice_Assistant"] == "listening"
            assert result["Server_Connection"] == "active"
            mock_push.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_returns_payload(self, sm):
        with patch.object(sm, "_push", new_callable=AsyncMock):
            result = await sm.update(robot_status="moving")
            assert isinstance(result, dict)
            assert result["Robot_Status"] == "moving"


# ---------------------------------------------------------------------------
# Push payload format
# ---------------------------------------------------------------------------

class TestPushPayload:
    @pytest.mark.asyncio
    async def test_push_sends_correct_json(self, sm):
        mock_response = MagicMock()
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)

        mock_client_ctx = MagicMock()
        mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

        import httpx as httpx_mod
        with patch.object(httpx_mod, "AsyncClient", return_value=mock_client_ctx):
            await sm._push({"Voice_Assistant": "listening", "Server_Connection": "active", "Robot_Status": "N/A"})

        mock_client_instance.post.assert_called_once()
        call_kwargs = mock_client_instance.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["message_type"] == "COMPONENTS_STATUS"
        inner = json.loads(body["payload"])
        assert inner["Voice_Assistant"] == "listening"


# ---------------------------------------------------------------------------
# Wakeword-driven status transitions
# ---------------------------------------------------------------------------

class TestWakewordStatusFlow:
    @pytest.mark.asyncio
    async def test_idle_to_listening_to_idle(self, sm):
        """Simulate status changes during a wakeword detection cycle."""
        with patch.object(sm, "_push", new_callable=AsyncMock) as mock_push:
            await sm.update(voice_assistant="listening")
            assert sm.voice_assistant == "listening"
            assert mock_push.call_count == 1

            await sm.update(voice_assistant="idle")
            assert sm.voice_assistant == "idle"
            assert mock_push.call_count == 2
