"""Pluggable STT client implementations.

Supports gRPC (Riva/NIM Parakeet) and HTTP backends.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from loguru import logger


class STTClient(ABC):
    """Abstract STT client interface."""

    @abstractmethod
    async def start_stream(self) -> None:
        """Open a streaming session."""

    @abstractmethod
    async def send_audio(self, pcm_chunk: bytes) -> None:
        """Send a chunk of PCM audio (s16le, 16kHz, mono)."""

    @abstractmethod
    async def get_transcription(self) -> Optional[str]:
        """Get the latest final transcription, or None if no new result."""

    @abstractmethod
    async def stop_stream(self) -> None:
        """Close the streaming session."""


class GrpcSTTClient(STTClient):
    """Streaming gRPC client for Riva / NIM Parakeet ASR."""

    def __init__(self, host: str, port: int, language: str = "en-US"):
        self._host = host
        self._port = port
        self._language = language
        self._channel = None
        self._stream = None
        self._results: asyncio.Queue[str] = asyncio.Queue()

    async def start_stream(self) -> None:
        try:
            import grpc
            from grpc import aio as grpc_aio
            self._channel = grpc_aio.insecure_channel(f"{self._host}:{self._port}")
            logger.info(f"[STT] gRPC channel opened to {self._host}:{self._port}")
        except ImportError:
            raise RuntimeError("grpcio is required for gRPC STT. Install with: pip install grpcio")

    async def send_audio(self, pcm_chunk: bytes) -> None:
        pass  # In a full implementation, chunks are streamed to the ASR service

    async def get_transcription(self) -> Optional[str]:
        try:
            return self._results.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop_stream(self) -> None:
        if self._channel:
            await self._channel.close()
            self._channel = None


class HttpSTTClient(STTClient):
    """HTTP-based STT client. Accumulates audio and POSTs for transcription."""

    def __init__(self, url: str):
        self._url = url
        self._buffer = bytearray()
        self._min_buffer_bytes = 32000  # ~1 second at 16kHz 16-bit mono

    async def start_stream(self) -> None:
        self._buffer.clear()

    async def send_audio(self, pcm_chunk: bytes) -> None:
        self._buffer.extend(pcm_chunk)

    async def get_transcription(self) -> Optional[str]:
        if len(self._buffer) < self._min_buffer_bytes:
            return None

        import httpx
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self._url,
                    content=bytes(self._buffer),
                    headers={"Content-Type": "audio/pcm;rate=16000;encoding=signed-integer;bits=16"},
                )
                resp.raise_for_status()
                data = resp.json()
                self._buffer.clear()
                return data.get("text", "").strip() or None
        except Exception as exc:
            logger.warning(f"[STT] HTTP transcription failed: {exc}")
            self._buffer.clear()
            return None

    async def stop_stream(self) -> None:
        self._buffer.clear()


def create_stt_client(config: Dict[str, Any]) -> STTClient:
    """Factory: create STT client from config."""
    stt_cfg = config if "protocol" in config else config.get("speech", {}).get("stt", {})
    protocol = stt_cfg.get("protocol", "grpc")
    host = stt_cfg.get("host", "stt-service")
    port = int(stt_cfg.get("port", 50051))

    if protocol == "grpc":
        return GrpcSTTClient(host, port)
    elif protocol == "http":
        url = f"http://{host}:{port}/transcribe"
        return HttpSTTClient(url)
    else:
        raise ValueError(f"Unknown STT protocol: {protocol}")
