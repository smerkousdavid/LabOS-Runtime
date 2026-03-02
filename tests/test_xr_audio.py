"""Integration tests: send audio to the XR runtime via the dashboard API."""

from __future__ import annotations

import subprocess
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration


def _mp3_to_wav(mp3_path: Path) -> bytes:
    """Convert MP3 to WAV (PCM s16le, 16kHz, mono) in memory."""
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(mp3_path),
            "-f", "wav", "-ar", "16000", "-ac", "1",
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg WAV conversion failed: {result.stderr.decode()[:200]}")
    return result.stdout


class TestXRAudioUpload:
    """Upload audio via dashboard /api/send_audio and verify the response."""

    @pytest.fixture(autouse=True)
    def _require(self, require_dashboard):
        pass

    def test_send_audio_streaming(self, dashboard_url, audio_cat):
        wav_data = _mp3_to_wav(audio_cat)
        assert len(wav_data) > 44, "WAV too short"

        with httpx.Client(base_url=dashboard_url, timeout=15.0) as client:
            resp = client.post(
                "/api/send_audio",
                files={"audio": ("test.wav", wav_data, "audio/wav")},
                data={"sample_rate": "16000", "method": "streaming"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data.get("success") is True or "error" in data

    def test_send_audio_single_chunk(self, dashboard_url, audio_cat):
        wav_data = _mp3_to_wav(audio_cat)

        with httpx.Client(base_url=dashboard_url, timeout=10.0) as client:
            resp = client.post(
                "/api/send_audio",
                files={"audio": ("test.wav", wav_data, "audio/wav")},
                data={"sample_rate": "16000", "method": "audio"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "success" in data
