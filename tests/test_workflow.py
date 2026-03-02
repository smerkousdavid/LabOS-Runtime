"""End-to-end unit tests: wakeword + status manager combined flow.

No live services required -- StatusManager._push is mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from wakeword import WakeWordFilter, State
from status_manager import StatusManager


@pytest.fixture
def ww():
    return WakeWordFilter(
        wake_words=["stella", "hey stella", "hi stella"],
        timeout_seconds=10.0,
        sleep_commands=["thanks", "goodbye", "go to sleep"],
    )


@pytest.fixture
def sm():
    return StatusManager("http://fake:5000")


# ---------------------------------------------------------------------------
# Full workflow
# ---------------------------------------------------------------------------

class TestWakewordStatusWorkflow:
    @pytest.mark.asyncio
    async def test_activate_and_deactivate(self, ww, sm):
        with patch.object(sm, "_push", new_callable=AsyncMock) as mock_push:
            # Initial state
            assert ww.state == State.IDLE

            # User says wake word + command
            cleaned = ww.process("hey stella look up a picture of a cat")
            assert cleaned == "look up a picture of a cat"
            assert ww.state == State.ACTIVE

            await sm.update(voice_assistant="listening")
            assert sm.voice_assistant == "listening"
            assert mock_push.call_count == 1

            # User says sleep command
            result = ww.process("thanks")
            assert result is None
            assert ww.state == State.IDLE

            await sm.update(voice_assistant="idle")
            assert sm.voice_assistant == "idle"
            assert mock_push.call_count == 2

    @pytest.mark.asyncio
    async def test_passthrough_while_active(self, ww, sm):
        with patch.object(sm, "_push", new_callable=AsyncMock):
            ww.process("stella begin")
            assert ww.state == State.ACTIVE

            result = ww.process("tell me about quantum physics")
            assert result == "tell me about quantum physics"
            assert ww.state == State.ACTIVE

    @pytest.mark.asyncio
    async def test_status_unchanged_no_extra_push(self, ww, sm):
        """If wakeword state doesn't change, no extra push should happen."""
        with patch.object(sm, "_push", new_callable=AsyncMock) as mock_push:
            await sm.update(voice_assistant="idle")
            initial_count = mock_push.call_count

            # Text without wakeword, state stays IDLE
            ww.process("random noise")
            assert ww.state == State.IDLE

            await sm.update(voice_assistant="idle")
            assert mock_push.call_count == initial_count

    @pytest.mark.asyncio
    async def test_server_connection_status(self, sm):
        """Simulate NAT connect/disconnect status updates."""
        with patch.object(sm, "_push", new_callable=AsyncMock) as mock_push:
            await sm.update(server_connection="active")
            assert sm.server_connection == "active"

            await sm.update(server_connection="inactive")
            assert sm.server_connection == "inactive"
            assert mock_push.call_count == 2

    @pytest.mark.asyncio
    async def test_initial_status_on_connect(self, sm):
        """Verify the initial COMPONENTS_STATUS payload sent on glasses connect."""
        with patch.object(sm, "_push", new_callable=AsyncMock) as mock_push:
            result = await sm.update(
                voice_assistant="idle",
                server_connection="inactive",
                robot_status="N/A",
            )
            assert result == {
                "Voice_Assistant": "idle",
                "Server_Connection": "inactive",
                "Robot_Status": "N/A",
            }
            mock_push.assert_called_once()
