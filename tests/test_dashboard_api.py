"""Integration tests for the Dashboard REST API."""

from __future__ import annotations

import json

import httpx
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def client(dashboard_url, require_dashboard):
    """Synchronous httpx client pointed at the dashboard."""
    return httpx.Client(base_url=dashboard_url, timeout=5.0)


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------

class TestStatusEndpoint:
    def test_status_returns_200(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "xr_service_connected" in data

    def test_status_has_socket_path(self, client):
        resp = client.get("/api/status")
        data = resp.json()
        assert "socket_path" in data


# ---------------------------------------------------------------------------
# Cameras endpoint
# ---------------------------------------------------------------------------

class TestCamerasEndpoint:
    def test_cameras_returns_list(self, client):
        resp = client.get("/api/cameras")
        assert resp.status_code == 200
        data = resp.json()
        assert "cameras" in data
        assert isinstance(data["cameras"], list)

    def test_camera_has_streams(self, client):
        resp = client.get("/api/cameras")
        data = resp.json()
        if data.get("cameras"):
            cam = data["cameras"][0]
            assert "streams" in cam
            assert "video" in cam["streams"]
            assert "audio" in cam["streams"]


# ---------------------------------------------------------------------------
# Send message endpoint
# ---------------------------------------------------------------------------

class TestSendMessage:
    def test_send_message_success(self, client):
        payload = json.dumps({"message": {"type": "rich-text", "content": "pytest test"}})
        resp = client.post(
            "/api/send_message",
            json={"message_type": "GENERIC", "payload": payload},
        )
        # May succeed or fail if XR service is not connected -- just check format
        data = resp.json()
        assert "success" in data or "error" in data

    def test_send_message_missing_type(self, client):
        resp = client.post("/api/send_message", json={"payload": "x"})
        assert resp.status_code == 400

    def test_send_message_missing_payload(self, client):
        resp = client.post("/api/send_message", json={"message_type": "GENERIC"})
        assert resp.status_code == 400

    def test_send_message_no_body(self, client):
        resp = client.post("/api/send_message")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Agent status endpoint
# ---------------------------------------------------------------------------

class TestAgentStatus:
    def test_agent_status_shape(self, client):
        resp = client.get("/api/agent/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "connected" in data
        assert "nat_url" in data
