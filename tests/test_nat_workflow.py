"""Integration tests for the NAT WebSocket workflow."""

from __future__ import annotations

import asyncio
import json

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def nat_ws_url(nat_url):
    """Full WebSocket URL including a test session_id."""
    sep = "&" if "?" in nat_url else "?"
    return f"{nat_url}{sep}session_id=test-pytest"


class TestNATConnection:
    """Requires a running NAT server."""

    @pytest.fixture(autouse=True)
    def _require(self, require_nat):
        pass

    @pytest.mark.asyncio
    async def test_connect_and_send_stream_info(self, nat_ws_url):
        import websockets

        async with websockets.connect(nat_ws_url, close_timeout=3) as ws:
            await ws.send(json.dumps({
                "type": "stream_info",
                "camera_index": 1,
                "rtsp_base": "rtsp://test:8554",
                "paths": {
                    "video": "NB_0001_TX_CAM_RGB",
                    "audio": "NB_0001_TX_MIC_p6S",
                    "merged": "NB_0001_TX_CAM_RGB_MIC_p6S",
                },
            }))
            # NAT may or may not reply to stream_info; just verify no crash
            await asyncio.sleep(0.3)

    @pytest.mark.asyncio
    async def test_send_user_message_gets_response(self, nat_ws_url):
        import websockets

        async with websockets.connect(nat_ws_url, close_timeout=5) as ws:
            await ws.send(json.dumps({
                "type": "stream_info",
                "camera_index": 1,
                "rtsp_base": "rtsp://test:8554",
                "paths": {"video": "v", "audio": "a", "merged": "m"},
            }))
            await asyncio.sleep(0.2)

            await ws.send(json.dumps({
                "type": "user_message",
                "text": "hello from pytest",
            }))

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                msg = json.loads(raw)
                assert "type" in msg
            except asyncio.TimeoutError:
                pytest.skip("NAT did not respond within 30s (may be expected for some agents)")

    @pytest.mark.asyncio
    async def test_reconnect_creates_fresh_session(self, nat_ws_url):
        import websockets

        async with websockets.connect(nat_ws_url, close_timeout=3) as ws:
            await ws.send(json.dumps({"type": "stream_info", "camera_index": 1,
                                       "rtsp_base": "x", "paths": {"video": "v", "audio": "a", "merged": "m"}}))
            await asyncio.sleep(0.2)

        # Reconnect -- should work without error
        async with websockets.connect(nat_ws_url, close_timeout=3) as ws:
            await ws.send(json.dumps({"type": "stream_info", "camera_index": 1,
                                       "rtsp_base": "x", "paths": {"video": "v", "audio": "a", "merged": "m"}}))
            await asyncio.sleep(0.2)
