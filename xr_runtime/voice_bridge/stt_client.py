"""Pluggable STT client implementations.

Supports six backends:
  - grpc          : Riva StreamingRecognize (for transducer/RNNT models)
  - grpc-batch    : Riva Recognize batch API (for CTC models like Parakeet CTC)
  - http          : Generic HTTP POST (raw PCM body, expects JSON {text: ...})
  - vllm          : vLLM /v1/audio/transcriptions (e.g. Voxtral-Mini)
  - parakeet_ws   : Parakeet WebSocket /v1/realtime (streaming, low-latency)
  - elevenlabs    : ElevenLabs Scribe realtime WebSocket STT
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import struct
import wave
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from loguru import logger

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # s16le


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


# ---------------------------------------------------------------------------
# gRPC streaming (Riva StreamingRecognize -- for RNNT/TDT models)
# ---------------------------------------------------------------------------

class GrpcSTTClient(STTClient):
    """Bidirectional streaming gRPC client for Riva StreamingRecognize.

    Imports protobuf types from the ``riva.client.proto`` package (official SDK)
    when available, falling back to locally-compiled stubs (Docker build) if not.
    """

    def __init__(self, host: str, port: int, language: str = "en-US", model: str = ""):
        self._host = host
        self._port = port
        self._language = language
        self._model = model
        self._channel = None
        self._stub = None
        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._results: asyncio.Queue[str] = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None
        self._streaming = False

    async def start_stream(self) -> None:
        from grpc import aio as grpc_aio

        try:
            from riva.client.proto import riva_asr_pb2 as asr_pb2
            from riva.client.proto import riva_asr_pb2_grpc as asr_grpc
            from riva.client.proto import riva_audio_pb2 as audio_pb2
        except ImportError:
            import riva_asr_pb2 as asr_pb2        # type: ignore[no-redef]
            import riva_asr_pb2_grpc as asr_grpc   # type: ignore[no-redef]
            import riva_audio_pb2 as audio_pb2     # type: ignore[no-redef]

        self._asr_pb2 = asr_pb2
        self._audio_pb2 = audio_pb2

        self._channel = grpc_aio.insecure_channel(
            f"{self._host}:{self._port}",
            options=[
                ("grpc.max_receive_message_length", 8 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 10000),
            ],
        )
        self._stub = asr_grpc.RivaSpeechRecognitionStub(self._channel)
        self._streaming = True
        self._reader_task = asyncio.create_task(self._stream_loop())
        logger.info(f"[STT] gRPC streaming client started -> {self._host}:{self._port}")

    async def send_audio(self, pcm_chunk: bytes) -> None:
        if self._streaming:
            await self._audio_queue.put(pcm_chunk)

    async def end_audio(self) -> None:
        """Signal that no more audio will be sent (closes the gRPC stream)."""
        await self._audio_queue.put(None)

    async def get_transcription(self) -> Optional[str]:
        try:
            return self._results.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop_stream(self) -> None:
        self._streaming = False
        await self._audio_queue.put(None)
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None
        if self._channel:
            await self._channel.close()
            self._channel = None

    async def _request_generator(self):
        asr_pb2, audio_pb2 = self._asr_pb2, self._audio_pb2
        rec_config = asr_pb2.RecognitionConfig(
            encoding=audio_pb2.LINEAR_PCM,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self._language,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            audio_channel_count=1,
        )
        if self._model:
            rec_config.model = self._model
        config = asr_pb2.StreamingRecognitionConfig(
            config=rec_config,
            interim_results=True,
        )
        yield asr_pb2.StreamingRecognizeRequest(streaming_config=config)
        chunks_sent = 0
        while self._streaming:
            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if chunk is None:
                break
            chunks_sent += 1
            if chunks_sent == 1:
                logger.info("[STT] First audio chunk sent to gRPC stream")
            yield asr_pb2.StreamingRecognizeRequest(audio_content=chunk)

    async def _stream_loop(self):
        backoff = 0.05
        while self._streaming:
            try:
                logger.info("[STT] Opening gRPC StreamingRecognize stream")
                responses = self._stub.StreamingRecognize(self._request_generator())
                async for resp in responses:
                    for result in resp.results:
                        if result.alternatives:
                            text = result.alternatives[0].transcript.strip()
                            if not text:
                                continue
                            if result.is_final:
                                await self._results.put(text)
                            else:
                                logger.info(f"[STT] Interim: {text}")
                backoff = 0.05
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._streaming:
                    break
                logger.warning(f"[STT] gRPC stream error: {exc}; retrying in {backoff:.2f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 0.5)


# ---------------------------------------------------------------------------
# gRPC batch (Riva Recognize -- for CTC models like Parakeet-CTC)
# ---------------------------------------------------------------------------

class GrpcBatchSTTClient(STTClient):
    """Batch gRPC client using Riva Recognize (offline/CTC models).

    Accumulates audio in a buffer and sends a batch Recognize request
    once enough audio has been collected.
    """

    BUFFER_SECONDS = 3.0
    MIN_BUFFER_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 1.0)

    def __init__(self, host: str, port: int, language: str = "en-US", model: str = ""):
        self._host = host
        self._port = port
        self._language = language
        self._model = model
        self._channel = None
        self._stub = None
        self._buffer = bytearray()
        self._results: asyncio.Queue[str] = asyncio.Queue()
        self._max_buffer = int(SAMPLE_RATE * BYTES_PER_SAMPLE * self.BUFFER_SECONDS)
        self._lock = asyncio.Lock()
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None

    async def start_stream(self) -> None:
        from grpc import aio as grpc_aio
        import riva_asr_pb2 as asr_pb2
        import riva_asr_pb2_grpc as asr_grpc
        import riva_audio_pb2 as audio_pb2

        self._asr_pb2 = asr_pb2
        self._audio_pb2 = audio_pb2

        self._channel = grpc_aio.insecure_channel(
            f"{self._host}:{self._port}",
            options=[("grpc.max_receive_message_length", 8 * 1024 * 1024)],
        )
        self._stub = asr_grpc.RivaSpeechRecognitionStub(self._channel)
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info(f"[STT] gRPC batch client started -> {self._host}:{self._port}")

    async def send_audio(self, pcm_chunk: bytes) -> None:
        async with self._lock:
            self._buffer.extend(pcm_chunk)

    async def get_transcription(self) -> Optional[str]:
        try:
            return self._results.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop_stream(self) -> None:
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._channel:
            await self._channel.close()
            self._channel = None
        self._buffer.clear()

    async def _periodic_flush(self):
        """Periodically send accumulated audio for batch recognition."""
        backoff = 0.05
        _model_error_logged = False
        while self._running:
            await asyncio.sleep(self.BUFFER_SECONDS)
            async with self._lock:
                if len(self._buffer) < self.MIN_BUFFER_BYTES:
                    continue
                audio_data = bytes(self._buffer)
                self._buffer.clear()

            try:
                resp = await self._recognize(audio_data)
                if resp:
                    await self._results.put(resp)
                    backoff = 0.05
                    _model_error_logged = False
            except asyncio.CancelledError:
                break
            except Exception as exc:
                err_str = str(exc)
                if "Unavailable model" in err_str and not _model_error_logged:
                    logger.error(
                        "[STT] No ASR model available on the server. "
                        "If using NIM Parakeet, ensure SKIP_MODEL_BUILD is false "
                        "on first run so the TensorRT engine is built. "
                        "Will keep retrying silently."
                    )
                    _model_error_logged = True
                elif "Unavailable model" not in err_str:
                    logger.warning(f"[STT] gRPC batch error: {exc}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 0.5)

    async def _recognize(self, audio: bytes) -> Optional[str]:
        asr_pb2, audio_pb2 = self._asr_pb2, self._audio_pb2
        config = asr_pb2.RecognitionConfig(
            encoding=audio_pb2.LINEAR_PCM,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self._language,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            audio_channel_count=1,
        )
        if self._model:
            config.model = self._model
        req = asr_pb2.RecognizeRequest(config=config, audio=audio)
        resp = await self._stub.Recognize(req, timeout=15)
        for result in resp.results:
            if result.alternatives:
                text = result.alternatives[0].transcript.strip()
                if text:
                    return text
        return None


# ---------------------------------------------------------------------------
# HTTP POST (raw PCM body -- generic STT servers)
# ---------------------------------------------------------------------------

class HttpSTTClient(STTClient):
    """HTTP-based STT client. Accumulates audio and POSTs for transcription."""

    def __init__(self, url: str):
        self._url = url
        self._buffer = bytearray()
        self._min_buffer_bytes = 32000

    async def start_stream(self) -> None:
        self._buffer.clear()
        logger.info(f"[STT] HTTP client started -> {self._url}")

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


# ---------------------------------------------------------------------------
# vLLM OpenAI-compatible (e.g. Voxtral-Mini via /v1/audio/transcriptions)
# ---------------------------------------------------------------------------

class VllmSTTClient(STTClient):
    """Batch STT via vLLM's OpenAI-compatible /v1/audio/transcriptions.

    Buffers PCM audio and periodically POSTs a WAV file for transcription.
    Compatible with any vLLM-served ASR model (Voxtral, Whisper, etc.).
    """

    BUFFER_SECONDS = 3.0
    MIN_BUFFER_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 1.0)

    def __init__(self, host: str, port: int, model: str = ""):
        self._base_url = f"http://{host}:{port}"
        self._model = model
        self._buffer = bytearray()
        self._results: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._http_client = None

    async def start_stream(self) -> None:
        import httpx
        self._http_client = httpx.AsyncClient(timeout=60.0)
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info(f"[STT] vLLM client started -> {self._base_url} model={self._model}")

    async def send_audio(self, pcm_chunk: bytes) -> None:
        async with self._lock:
            self._buffer.extend(pcm_chunk)

    async def get_transcription(self) -> Optional[str]:
        try:
            return self._results.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop_stream(self) -> None:
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._buffer.clear()

    async def _periodic_flush(self):
        _consecutive_conn_errors = 0
        while self._running:
            await asyncio.sleep(self.BUFFER_SECONDS)
            async with self._lock:
                if len(self._buffer) < self.MIN_BUFFER_BYTES:
                    continue
                audio_data = bytes(self._buffer)
                self._buffer.clear()

            try:
                text = await self._transcribe(audio_data)
                if text:
                    await self._results.put(text)
                _consecutive_conn_errors = 0
            except asyncio.CancelledError:
                break
            except _VllmServerError:
                pass
            except Exception as exc:
                _consecutive_conn_errors += 1
                if _consecutive_conn_errors <= 3:
                    logger.warning(f"[STT] vLLM connection error: {type(exc).__name__}: {exc}")
                elif _consecutive_conn_errors == 4:
                    logger.warning("[STT] vLLM unreachable; will keep retrying silently")
                await asyncio.sleep(min(0.05 * (2 ** _consecutive_conn_errors), 0.5))

    async def _transcribe(self, pcm_data: bytes) -> Optional[str]:
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_data)
        wav_bytes = wav_buf.getvalue()

        resp = await self._http_client.post(
            f"{self._base_url}/v1/audio/transcriptions",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": self._model, "language": "en"},
        )
        resp.raise_for_status()
        result = resp.json()

        if "error" in result:
            err = result["error"]
            msg = err.get("message", "unknown")[:120]
            logger.debug(f"[STT] vLLM returned error in body: {msg}")
            raise _VllmServerError(msg)

        return result.get("text", "").strip() or None


class _VllmServerError(Exception):
    """Raised when vLLM returns HTTP 200 but with an error JSON body."""


# ---------------------------------------------------------------------------
# Parakeet WebSocket (/v1/realtime -- streaming, low-latency)
# ---------------------------------------------------------------------------

class ParakeetWsSTTClient(STTClient):
    """Persistent-WebSocket STT client for the Parakeet /v1/realtime endpoint.

    Audio chunks are streamed as base64 PCM16 via ``input_audio_buffer.append``.
    A background task commits the buffer every ``COMMIT_INTERVAL`` seconds and
    collects ``transcription.delta`` results.
    """

    COMMIT_INTERVAL = 0.25

    def __init__(self, host: str, port: int, model: str = ""):
        self._ws_url = f"ws://{host}:{port}/v1/realtime"
        self._model = model
        self._ws = None
        self._results: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None
        self._commit_task: Optional[asyncio.Task] = None
        self._has_audio = False
        self._audio_lock = asyncio.Lock()

    async def start_stream(self) -> None:
        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info(f"[STT] Parakeet WS client started -> {self._ws_url}")

    async def send_audio(self, pcm_chunk: bytes) -> None:
        if self._ws and self._running:
            b64 = base64.b64encode(pcm_chunk).decode("ascii")
            try:
                await self._ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": b64,
                }))
                async with self._audio_lock:
                    self._has_audio = True
            except Exception:
                pass

    async def get_transcription(self) -> Optional[str]:
        try:
            return self._results.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop_stream(self) -> None:
        self._running = False
        if self._commit_task:
            self._commit_task.cancel()
            try:
                await self._commit_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _ws_loop(self):
        """Maintain the WebSocket connection with fast reconnect."""
        import websockets

        backoff = 0.05
        while self._running:
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=2 ** 22,
                ) as ws:
                    self._ws = ws
                    backoff = 0.05

                    session_msg = json.loads(await ws.recv())
                    sid = session_msg.get("session", {}).get("id", "?")
                    logger.info(f"[STT] Parakeet WS connected (session={sid})")

                    self._commit_task = asyncio.create_task(self._commit_loop())

                    async for raw in ws:
                        event = json.loads(raw)
                        etype = event.get("type", "")
                        if etype == "transcription.delta":
                            text = event.get("text", "").strip()
                            if text:
                                await self._results.put(text)
                        elif etype == "error":
                            msg = event.get("error", {}).get("message", "")
                            if msg:
                                logger.debug(f"[STT] Parakeet WS server error: {msg}")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._ws = None
                if self._commit_task:
                    self._commit_task.cancel()
                if not self._running:
                    break
                logger.warning(f"[STT] Parakeet WS error: {type(exc).__name__}: {exc}; "
                               f"reconnecting in {backoff:.2f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 0.5)

    async def _commit_loop(self):
        """Periodically commit the audio buffer to trigger transcription."""
        while self._running and self._ws:
            await asyncio.sleep(self.COMMIT_INTERVAL)
            async with self._audio_lock:
                has_audio = self._has_audio
                self._has_audio = False
            if has_audio and self._ws:
                try:
                    await self._ws.send(json.dumps({
                        "type": "input_audio_buffer.commit",
                        "final": True,
                    }))
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# ElevenLabs Scribe (wss://api.elevenlabs.io/v1/speech-to-text/realtime)
# ---------------------------------------------------------------------------

class ElevenLabsSTTClient(STTClient):
    """Streaming STT via ElevenLabs Scribe v2 realtime WebSocket.

    Sends base64 PCM16 chunks as ``input_audio_chunk`` messages (commit=False).
    ElevenLabs Scribe uses its own VAD to detect speech boundaries and returns
    ``partial_transcript`` / ``committed_transcript`` automatically -- no
    explicit commit messages are needed.
    """

    COMMIT_INTERVAL = 3.0  # kept for test compat; not used at runtime
    WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

    def __init__(self, api_key: str, model: str = "scribe_v2_realtime",
                 language: str = "en"):
        self._api_key = api_key
        self._model = model
        self._language = language
        self._ws = None
        self._results: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None
        self._has_audio = False
        self._audio_lock = asyncio.Lock()
        self._chunks_sent = 0

    async def start_stream(self) -> None:
        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info(f"[STT] ElevenLabs client started (model={self._model})")

    async def send_audio(self, pcm_chunk: bytes) -> None:
        async with self._audio_lock:
            self._has_audio = True
            self._chunks_sent += 1
        if self._ws and self._running:
            b64 = base64.b64encode(pcm_chunk).decode("ascii")
            try:
                await self._ws.send(json.dumps({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": b64,
                    "commit": False,
                    "sample_rate": SAMPLE_RATE,
                }))
            except Exception:
                pass

    async def get_transcription(self) -> Optional[str]:
        try:
            return self._results.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop_stream(self) -> None:
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _ws_loop(self):
        import websockets

        url = f"{self.WS_URL}?model_id={self._model}"
        headers = {"xi-api-key": self._api_key}
        backoff = 0.5

        while self._running:
            # Wait until audio is actually flowing before opening a connection
            while self._running:
                async with self._audio_lock:
                    if self._has_audio:
                        break
                await asyncio.sleep(0.1)
            if not self._running:
                break

            try:
                async with websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=2 ** 22,
                ) as ws:
                    self._ws = ws

                    session_raw = await ws.recv()
                    session_data = json.loads(session_raw)
                    sid = session_data.get("session_id", "?")
                    logger.info(f"[STT] ElevenLabs WS connected (session={sid})")
                    backoff = 0.5

                    async for raw in ws:
                        event = json.loads(raw)
                        mtype = event.get("message_type", "")
                        if mtype == "committed_transcript":
                            text = event.get("text", "").strip()
                            if text:
                                await self._results.put(text)
                        elif mtype == "committed_transcript_with_timestamps":
                            text = event.get("text", "").strip()
                            if text:
                                await self._results.put(text)
                        elif mtype == "partial_transcript":
                            pass
                        elif mtype == "input_error":
                            logger.warning(f"[STT] ElevenLabs input error: {event}")
                        else:
                            logger.debug(f"[STT] ElevenLabs event: {mtype}")

                    code = ws.close_code
                    reason = ws.close_reason or ""
                    logger.info(f"[STT] ElevenLabs WS closed (code={code} reason={reason})")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._running:
                    break
                logger.warning(f"[STT] ElevenLabs WS error: {type(exc).__name__}: {exc}; "
                               f"reconnecting in {backoff:.1f}s")
            finally:
                self._ws = None

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_stt_client(config: Dict[str, Any]) -> STTClient:
    """Factory: create STT client from config.

    Supported protocols:
      grpc         - Riva StreamingRecognize (RNNT/TDT models)
      grpc-batch   - Riva Recognize (CTC models like Parakeet-CTC)
      http         - generic HTTP POST with raw PCM body
      vllm         - vLLM /v1/audio/transcriptions (Voxtral, Whisper, etc.)
      parakeet_ws  - Parakeet WebSocket /v1/realtime (streaming)
      elevenlabs   - ElevenLabs Scribe realtime WebSocket STT
    """
    import os

    stt_cfg = config if "protocol" in config else config.get("speech", {}).get("stt", {})
    protocol = stt_cfg.get("protocol", "grpc")
    host = stt_cfg.get("host", "stt-service")
    port = int(stt_cfg.get("port", 50051))
    model = stt_cfg.get("model", "")

    if protocol == "grpc":
        return GrpcSTTClient(host, port, model=model)
    elif protocol == "grpc-batch":
        return GrpcBatchSTTClient(host, port, model=model)
    elif protocol == "vllm":
        return VllmSTTClient(host, port, model=model)
    elif protocol == "parakeet_ws":
        return ParakeetWsSTTClient(host, port, model=model)
    elif protocol == "elevenlabs":
        api_key = stt_cfg.get("api_key") or os.environ.get("ELEVENLABS_API_KEY", "")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY required for elevenlabs STT")
        return ElevenLabsSTTClient(
            api_key=api_key,
            model=model or "scribe_v2_realtime",
        )
    elif protocol == "http":
        url = f"http://{host}:{port}/transcribe"
        return HttpSTTClient(url)
    else:
        raise ValueError(f"Unknown STT protocol: {protocol}")
