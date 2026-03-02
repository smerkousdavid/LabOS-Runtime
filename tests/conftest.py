"""Shared fixtures and markers for the LabOS XR Runtime test suite."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

# Allow imports from xr_runtime sub-packages without Docker context
sys.path.insert(0, str(ROOT / "xr_runtime" / "voice_bridge"))


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _load_secrets():
    """Load config/.env.secrets into os.environ so API keys are available."""
    secrets_path = ROOT / "config" / ".env.secrets"
    if secrets_path.exists():
        with open(secrets_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if value and key not in os.environ:
                    os.environ[key] = value


@pytest.fixture(scope="session")
def config() -> dict:
    cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        pytest.skip("config/config.yaml not found")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# URL fixtures (derived from config)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dashboard_url() -> str:
    return os.environ.get("DASHBOARD_URL", "http://localhost:5001")


@pytest.fixture(scope="session")
def nat_url(config) -> str:
    return config.get("nat_server", {}).get("url", "ws://localhost:8002/ws")


@pytest.fixture(scope="session")
def stt_config(config) -> dict:
    return config.get("speech", {}).get("stt", {})


# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def assets_dir() -> Path:
    return ROOT / "assets"


@pytest.fixture(scope="session")
def audio_cat(assets_dir) -> Path:
    p = assets_dir / "hello_stella_look_up_a_picture_of_a_cat.mp3"
    if not p.exists():
        pytest.skip(f"Asset not found: {p}")
    return p


@pytest.fixture(scope="session")
def audio_protocols(assets_dir) -> Path:
    p = assets_dir / "hi_stella_list_some_protocols.mp3"
    if not p.exists():
        pytest.skip(f"Asset not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Connectivity helpers -- skip integration tests when services are down
# ---------------------------------------------------------------------------

def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def require_dashboard(dashboard_url):
    parsed = urlparse(dashboard_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5001
    if not _tcp_reachable(host, port):
        pytest.skip(f"Dashboard unreachable at {host}:{port}")


@pytest.fixture(scope="session")
def require_nat(nat_url):
    parsed = urlparse(nat_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8002
    if not _tcp_reachable(host, port):
        pytest.skip(f"NAT server unreachable at {host}:{port}")


@pytest.fixture(scope="session")
def require_stt(stt_config):
    protocol = stt_config.get("protocol", "grpc")
    if protocol == "elevenlabs":
        if not os.environ.get("ELEVENLABS_API_KEY"):
            pytest.skip("ELEVENLABS_API_KEY not set")
        return
    host = stt_config.get("host", "localhost")
    port = int(stt_config.get("port", 50051))
    if not _tcp_reachable(host, port):
        pytest.skip(f"STT service unreachable at {host}:{port}")
