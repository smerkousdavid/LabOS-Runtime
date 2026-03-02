"""Integration tests: verify the configured STT client transcribes audio assets.

Works with every STT backend defined in stt_client.py (grpc, grpc-batch, http,
vllm, parakeet_ws, elevenlabs).  The test reads ``config/config.yaml`` to decide
which client to instantiate, sends two audio clips through it, and asserts key
words appear in the transcriptions.

MP3 decoding priority: ffmpeg -> soundfile -> (skip).
For ``vllm`` and ``parakeet_ws`` protocols an additional direct-endpoint test is
included that sends the raw audio file, bypassing PCM conversion entirely.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Optional

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# MP3 -> PCM decoder (best-effort, multiple fallbacks)
# ---------------------------------------------------------------------------

def _decode_mp3_to_pcm(mp3_path: Path) -> Optional[bytes]:
    """Return s16le 16 kHz mono PCM, or *None* if no decoder is available."""
    # 1) ffmpeg
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", str(mp3_path),
             "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
            capture_output=True, timeout=10,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 2) soundfile (needs libsndfile >= 1.1 for MP3)
    try:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(str(mp3_path), dtype="float32", always_2d=True)
        data = data.mean(axis=1)  # stereo -> mono
        if sr != 16000:
            try:
                import soxr
                data = soxr.resample(data, sr, 16000)
            except ImportError:
                ratio = 16000 / sr
                indices = (np.arange(int(len(data) * ratio)) / ratio).astype(int)
                indices = np.clip(indices, 0, len(data) - 1)
                data = data[indices]
        return (data * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
    except Exception:
        pass

    return None


def _fuzzy_match(transcription: str, *keywords: str) -> bool:
    t = transcription.lower()
    return all(kw.lower() in t for kw in keywords)


# ---------------------------------------------------------------------------
# Async helpers for the STT client interface
# ---------------------------------------------------------------------------

async def _transcribe_via_client(stt_config: dict, pcm: bytes) -> Optional[str]:
    """Feed PCM through the configured STT client and return transcription."""
    from stt_client import (
        ElevenLabsSTTClient,
        GrpcBatchSTTClient,
        GrpcSTTClient,
        ParakeetWsSTTClient,
        VllmSTTClient,
        create_stt_client,
    )

    client = create_stt_client(stt_config)
    await client.start_stream()

    chunk_size = 16000 * 2 // 10  # 100 ms = 3200 bytes
    for i in range(0, len(pcm), chunk_size):
        await client.send_audio(pcm[i : i + chunk_size])

    if isinstance(client, GrpcSTTClient):
        await client.end_audio()
        for _ in range(50):
            await asyncio.sleep(0.2)
            text = await client.get_transcription()
            if text is not None:
                await client.stop_stream()
                return text
    elif isinstance(client, (ParakeetWsSTTClient, ElevenLabsSTTClient)):
        await asyncio.sleep(client.COMMIT_INTERVAL + 5)
    elif isinstance(client, (GrpcBatchSTTClient, VllmSTTClient)):
        await asyncio.sleep(client.BUFFER_SECONDS + 5)
    else:
        await asyncio.sleep(2)

    text = await client.get_transcription()
    await client.stop_stream()
    return text


async def _transcribe_rest_direct(stt_config: dict, audio_path: Path) -> Optional[str]:
    """POST the raw audio file to /v1/audio/transcriptions (vLLM or Parakeet)."""
    import httpx

    host = stt_config.get("host", "localhost")
    port = int(stt_config.get("port", 8000))
    model = stt_config.get("model", "")

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    async with httpx.AsyncClient(timeout=60.0) as http:
        resp = await http.post(
            f"http://{host}:{port}/v1/audio/transcriptions",
            files={"file": (audio_path.name, audio_data, "audio/mpeg")},
            data={"model": model, "language": "en"},
        )
        resp.raise_for_status()
        return resp.json().get("text", "").strip() or None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSTTTranscription:
    """Transcription accuracy for the configured STT backend."""

    @pytest.fixture(autouse=True)
    def _require(self, require_stt):
        pass

    @pytest.mark.asyncio
    async def test_cat_audio(self, stt_config, audio_cat):
        """'hello stella look up a picture of a cat' -> stella, cat."""
        text = await self._run(stt_config, audio_cat)
        assert text is not None, "STT returned no transcription"
        assert _fuzzy_match(text, "stella", "cat"), (
            f"Expected 'stella' and 'cat' in: '{text}'"
        )

    @pytest.mark.asyncio
    async def test_protocols_audio(self, stt_config, audio_protocols):
        """'hi stella list some protocols' -> stella, protocol."""
        text = await self._run(stt_config, audio_protocols)
        assert text is not None, "STT returned no transcription"
        assert _fuzzy_match(text, "stella", "protocol"), (
            f"Expected 'stella' and 'protocol' in: '{text}'"
        )

    # ---- internal dispatch ----

    async def _run(self, stt_config: dict, audio_path: Path) -> Optional[str]:
        protocol = stt_config.get("protocol", "grpc")

        pcm = _decode_mp3_to_pcm(audio_path)

        if pcm is not None:
            return await _transcribe_via_client(stt_config, pcm)

        if protocol in ("vllm", "parakeet_ws"):
            return await _transcribe_rest_direct(stt_config, audio_path)

        pytest.skip(
            f"No audio decoder available and protocol '{protocol}' "
            "needs PCM (install ffmpeg or soundfile)"
        )


class TestDirectRESTEndpoint:
    """Extra coverage for the /v1/audio/transcriptions REST endpoint.

    Sends the raw MP3 file — no PCM decode needed.  Skipped when the
    configured protocol does not expose a REST transcription endpoint.
    """

    _REST_PROTOCOLS = {"vllm", "parakeet_ws"}

    @pytest.fixture(autouse=True)
    def _require_rest(self, stt_config, require_stt):
        if stt_config.get("protocol") not in self._REST_PROTOCOLS:
            pytest.skip(f"REST-specific test; current protocol is not in {self._REST_PROTOCOLS}")

    @pytest.mark.asyncio
    async def test_cat_direct(self, stt_config, audio_cat):
        text = await _transcribe_rest_direct(stt_config, audio_cat)
        assert text is not None
        assert _fuzzy_match(text, "stella", "cat"), f"Got: '{text}'"

    @pytest.mark.asyncio
    async def test_protocols_direct(self, stt_config, audio_protocols):
        text = await _transcribe_rest_direct(stt_config, audio_protocols)
        assert text is not None
        assert _fuzzy_match(text, "stella", "protocol"), f"Got: '{text}'"
