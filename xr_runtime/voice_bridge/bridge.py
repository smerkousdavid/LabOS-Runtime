"""Voice Bridge -- per-camera service connecting audio hardware to the NAT agent.

Main loop: MediaMTX audio -> FFmpeg decode -> STT -> accumulator -> wake word -> WebSocket to NAT.
Return path: NAT responses -> TTS Pusher / Dashboard.

Features:
  - Utterance accumulation: debounces short ASR fragments before processing
  - TTS barge-in: wake word or stop command interrupts TTS playback
  - Optional continuous video frame streaming to the NAT server

Message types sent to glasses:
  GENERIC              -- rich-text chat (user and agent messages)
  COMPONENTS_STATUS    -- voice/server/robot indicators
  SINGLE_STEP_PANEL_CONTENT  -- step panel (forwarded from NAT)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional

from loguru import logger

_log_dir = os.environ.get("LOG_DIR", "/app/logs")
if os.path.isdir(_log_dir):
    logger.add(os.path.join(_log_dir, "voice_bridge.log"), rotation="20 MB", retention="3 days", level="DEBUG")

from wakeword import WakeWordFilter
from stt_client import create_stt_client, STTClient
from ws_client import NATWebSocketClient
from status_manager import StatusManager


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

NAT_SERVER_URL = os.environ.get("NAT_SERVER_URL", "ws://localhost:8002/ws")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "1"))
SESSION_ID = os.environ.get("SESSION_ID", f"demo-{CAMERA_INDEX}")
MEDIAMTX_HOST = os.environ.get("MEDIAMTX_HOST", "mediamtx")
STT_HOST = os.environ.get("STT_HOST", "localhost")
STT_PORT = int(os.environ.get("STT_PORT", "50051"))
STT_PROTOCOL = os.environ.get("STT_PROTOCOL", "grpc")
STT_MODEL = os.environ.get("STT_MODEL", "")
TTS_PUSHER_URL = os.environ.get("TTS_PUSHER_URL", "http://tts-pusher:5000")
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://dashboard:5000")
SOCKET_PATH = os.environ.get("SOCKET_PATH", "/tmp/xr_service.sock")
FORWARD_AUDIO = os.environ.get("FORWARD_AUDIO", "false").lower() in ("true", "1", "yes")
FORWARD_FRAMES = os.environ.get("FORWARD_FRAMES", "false").lower() in ("true", "1", "yes")
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "480"))
FRAME_FPS = int(os.environ.get("FRAME_FPS", "15"))
WAKE_WORDS = os.environ.get("WAKE_WORDS", "stella,hey stella").split(",")
WAKE_TIMEOUT = float(os.environ.get("WAKE_TIMEOUT", "10"))
TTS_MODEL = os.environ.get("TTS_MODEL", "vibevoice")
RESET_SESSION = os.environ.get("RESET_SESSION_ON_DISCONNECT", "false").lower()
RTSP_EXTERNAL_HOST = os.environ.get("RTSP_EXTERNAL_HOST", "localhost")

AUDIO_CHUNK_MS = 100
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_SIZE = int(SAMPLE_RATE * (AUDIO_CHUNK_MS / 1000.0) * BYTES_PER_SAMPLE)

_xr_conn = None
_status: Optional[StatusManager] = None
_ww_filter: Optional[WakeWordFilter] = None
_tts_playing = False
_tts_cancel = asyncio.Event()

# Filler words / echo artefacts that ASR hallucinates from silence/noise
_NOISE_TRANSCRIPTIONS = frozenset({
    "mhm", "mm", "mmm", "hmm", "hm", "uh", "um", "ah", "oh",
    "uh-huh", "mm-hmm", "mhm mhm", "so so",
    "he", "ha", "hey", "hi",
})


def _get_xr_connection():
    global _xr_conn
    if _xr_conn is None:
        try:
            from xr_service_library import XRServiceConnection
            _xr_conn = XRServiceConnection(socket_path=SOCKET_PATH)
            _xr_conn.connect()
            logger.info(f"[Bridge] Connected to XR service at {SOCKET_PATH}")
        except Exception as exc:
            logger.warning(f"[Bridge] XR service connection failed: {exc}")
    return _xr_conn


# ---------------------------------------------------------------------------
# Utterance accumulator -- debounce short ASR fragments
# ---------------------------------------------------------------------------

class UtteranceAccumulator:
    """Buffers short STT fragments and flushes after a silence gap.

    Short fragments (single word or < MIN_CHARS) are held in the buffer.
    Longer fragments cause an immediate flush (with any buffered prefix prepended).
    A background task flushes the buffer after DEBOUNCE_SECONDS of inactivity.
    """

    MIN_CHARS = 4
    DEBOUNCE_SECONDS = 0.4

    def __init__(self):
        self._buffer: list[str] = []
        self._result_queue: asyncio.Queue[str] = asyncio.Queue()
        self._timer_handle: Optional[asyncio.TimerHandle] = None

    def feed(self, text: str) -> None:
        """Add a transcription fragment."""
        text = text.strip()
        if not text:
            return

        word_count = len(text.split())
        is_short = word_count <= 1 or len(text) < self.MIN_CHARS

        if is_short:
            self._buffer.append(text)
            self._reset_timer()
        else:
            self._buffer.append(text)
            self._flush()

    def _flush(self) -> None:
        if self._cancel_timer():
            pass
        if self._buffer:
            merged = " ".join(self._buffer)
            self._buffer.clear()
            self._result_queue.put_nowait(merged)

    def _reset_timer(self) -> None:
        self._cancel_timer()
        loop = asyncio.get_event_loop()
        self._timer_handle = loop.call_later(self.DEBOUNCE_SECONDS, self._flush)

    def _cancel_timer(self) -> bool:
        if self._timer_handle is not None:
            self._timer_handle.cancel()
            self._timer_handle = None
            return True
        return False

    def get(self) -> Optional[str]:
        """Return the next accumulated utterance, or None."""
        try:
            return self._result_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None


# ---------------------------------------------------------------------------
# Send messages to glasses via dashboard API
# ---------------------------------------------------------------------------

async def _send_to_glasses(message_type: str, payload: str):
    """POST a message to the dashboard for delivery to glasses."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(
                f"{DASHBOARD_URL}/api/send_message",
                json={"message_type": message_type, "payload": payload},
            )
    except Exception as exc:
        logger.warning(f"[Bridge] Send to glasses failed: {exc}")


async def send_generic(text: str, source: str = "Agent"):
    """Send a GENERIC rich-text message to the glasses."""
    payload = json.dumps({
        "message": {
            "type": "rich-text",
            "content": f"<b>{source}:</b> {text}",
            "source": source,
        }
    })
    await _send_to_glasses("GENERIC", payload)


# ---------------------------------------------------------------------------
# FFmpeg audio decoder
# ---------------------------------------------------------------------------

def start_audio_decoder() -> subprocess.Popen:
    idx = f"{CAMERA_INDEX:04d}"
    rtsp_url = f"rtsp://{MEDIAMTX_HOST}:8554/NB_{idx}_TX_MIC_p6S"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "pipe:1",
    ]
    logger.info(f"[Bridge] Starting audio decoder: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ---------------------------------------------------------------------------
# NAT response handlers
# ---------------------------------------------------------------------------

async def handle_agent_response(msg: dict):
    """Route agent_response: send GENERIC message to glasses and optionally TTS."""
    text = msg.get("text", "")
    tts = msg.get("tts", False)

    if text:
        await send_generic(text, source="Agent")

    if tts and text:
        await _trigger_tts(text)


async def handle_notification(msg: dict):
    text = msg.get("text", "")
    tts = msg.get("tts", False)
    if text:
        await send_generic(text, source="Agent")
    if tts and text:
        await _trigger_tts(text)


async def handle_display_update(msg: dict):
    """Forward display_update directly to glasses (already has correct message_type)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            payload_raw = msg.get("payload", "")
            if isinstance(payload_raw, (dict, list)):
                payload_str = json.dumps(payload_raw)
            else:
                payload_str = str(payload_raw)
            await client.post(
                f"{DASHBOARD_URL}/api/send_message",
                json={
                    "message_type": msg.get("message_type", "GENERIC"),
                    "payload": payload_str,
                },
            )
    except Exception as exc:
        logger.warning(f"[Bridge] Display update forward failed: {exc}")


async def handle_tts_only(msg: dict):
    text = msg.get("text", "")
    if text:
        await _trigger_tts(text)


async def handle_request_frames(msg: dict, ws_client: NATWebSocketClient):
    request_id = msg.get("request_id", "")
    count = msg.get("count", 8)
    interval_ms = msg.get("interval_ms", 1250)

    conn = _get_xr_connection()
    frames = []

    if conn:
        for i in range(count):
            try:
                frame = conn.get_latest_frame()
                if frame is not None:
                    import cv2
                    _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")
                    frames.append(b64)
            except Exception as exc:
                logger.warning(f"[Bridge] Frame capture failed: {exc}")
            if i < count - 1:
                await asyncio.sleep(interval_ms / 1000.0)

    await ws_client.send({
        "type": "frame_response",
        "request_id": request_id,
        "frames": frames,
    })


async def handle_tool_call(msg: dict):
    """Display tool call activity as a GENERIC chat message on the glasses."""
    tool_name = msg.get("tool_name", "unknown")
    summary = msg.get("summary", "")
    status = msg.get("status", "started")

    if status == "started":
        text = f"<color=#59D2FF>Tool: {tool_name}</color> -- {summary}"
    elif status == "completed":
        text = f"<color=#88CC88>Tool: {tool_name}</color> -- done"
    else:
        text = f"<color=#FF4444>Tool: {tool_name}</color> -- {status}"

    await send_generic(text, source="Tool")


async def handle_wake_timeout(msg: dict, ww_filter: WakeWordFilter):
    seconds = msg.get("seconds", 10)
    ww_filter.timeout_seconds = float(seconds)
    logger.info(f"[Bridge] Wake timeout updated to {seconds}s")


# ---------------------------------------------------------------------------
# TTS with barge-in support
# ---------------------------------------------------------------------------

async def _trigger_tts(text: str):
    """Synthesize TTS audio and deliver it to the glasses via the gRPC server.

    Supports barge-in: checks ``_tts_cancel`` between audio chunks.
    If the event is set (by the main loop detecting a wake word or stop command),
    playback stops immediately.
    """
    global _tts_playing
    logger.info(f"[TTS] Speaking: {text[:80]}")
    import httpx
    import io
    import wave

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{TTS_PUSHER_URL}/synthesize_wav",
                json={"text": text, "model": TTS_MODEL},
            )
            resp.raise_for_status()
            wav_bytes = resp.content
    except Exception as exc:
        logger.warning(f"[Bridge] TTS synthesis failed: {exc}")
        return

    conn = _get_xr_connection()
    if conn is None:
        logger.warning("[Bridge] No XR connection for TTS audio delivery")
        return

    _tts_cancel.clear()
    _tts_playing = True
    try:
        from xr_service_library.xr_types import AudioSample

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            sw = wf.getsampwidth()
            ch = wf.getnchannels()
            pcm = wf.readframes(wf.getnframes())

        chunk_duration_ms = 100
        bytes_per_frame = sw * ch
        chunk_frames = int(sr * chunk_duration_ms / 1000)
        chunk_bytes = chunk_frames * bytes_per_frame

        sent = 0
        for i in range(0, len(pcm), chunk_bytes):
            if _tts_cancel.is_set():
                logger.info(f"[TTS] Barge-in: stopped after {sent / bytes_per_frame / sr:.1f}s")
                break

            chunk = pcm[i : i + chunk_bytes]
            sample = AudioSample(audio_data=chunk, sample_rate=float(sr))
            conn.schedule_audio_transmission(sample)
            sent += len(chunk)
            await asyncio.sleep(chunk_duration_ms / 1000.0)
        else:
            logger.info(f"[TTS] Delivered {sent} bytes ({sent / bytes_per_frame / sr:.1f}s) to glasses")
    except Exception as exc:
        logger.warning(f"[Bridge] TTS audio delivery failed: {exc}")
    finally:
        await asyncio.sleep(0.3)
        _tts_playing = False
        if _ww_filter is not None:
            _ww_filter.touch()


# ---------------------------------------------------------------------------
# Continuous video frame streaming
# ---------------------------------------------------------------------------

async def _frame_stream_task(ws_client: NATWebSocketClient):
    """Background task: captures frames from XR service and pushes them over WS."""
    interval = 1.0 / max(1, FRAME_FPS)
    seq = 0
    logger.info(f"[Frames] Streaming at {FRAME_FPS} fps, {FRAME_WIDTH}x{FRAME_HEIGHT}")

    while True:
        try:
            conn = _get_xr_connection()
            if conn is None:
                await asyncio.sleep(1.0)
                continue

            frame = conn.get_latest_frame()
            if frame is None:
                await asyncio.sleep(interval)
                continue

            import cv2
            h, w = frame.shape[:2]
            if w != FRAME_WIDTH or h != FRAME_HEIGHT:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")
            seq += 1

            await ws_client.send({
                "type": "video_stream",
                "data": b64,
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "seq": seq,
            })

            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning(f"[Frames] Error: {exc}")
            await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main():
    global _status, _ww_filter

    logger.info(f"[Bridge] Starting for camera {CAMERA_INDEX}, session {SESSION_ID}")

    # Log RTSP URLs for external access (VLC, NAT server, etc.)
    idx = f"{CAMERA_INDEX:04d}"
    logger.info(f"[RTSP] Video : rtsp://{RTSP_EXTERNAL_HOST}:8554/NB_{idx}_TX_CAM_RGB")
    logger.info(f"[RTSP] Audio : rtsp://{RTSP_EXTERNAL_HOST}:8554/NB_{idx}_TX_MIC_p6S")
    logger.info(f"[RTSP] Merged: rtsp://{RTSP_EXTERNAL_HOST}:8554/NB_{idx}_TX_CAM_RGB_MIC_p6S")

    _status = StatusManager(DASHBOARD_URL)

    ww_filter = WakeWordFilter(
        wake_words=WAKE_WORDS,
        timeout_seconds=WAKE_TIMEOUT,
    )
    _ww_filter = ww_filter

    accumulator = UtteranceAccumulator()

    stt = create_stt_client({
        "host": STT_HOST,
        "port": STT_PORT,
        "protocol": STT_PROTOCOL,
        "model": STT_MODEL,
    })

    ws_client = NATWebSocketClient(
        url=NAT_SERVER_URL,
        session_id=SESSION_ID,
        camera_index=CAMERA_INDEX,
        rtsp_base=f"rtsp://{RTSP_EXTERNAL_HOST}:8554",
        on_connect=lambda: asyncio.ensure_future(_status.update(server_connection="active")),
        on_disconnect=lambda: asyncio.ensure_future(_status.update(server_connection="inactive")),
    )

    ws_client.on("agent_response", handle_agent_response)
    ws_client.on("notification", handle_notification)
    ws_client.on("display_update", handle_display_update)
    ws_client.on("tts_only", handle_tts_only)
    ws_client.on("tool_call", handle_tool_call)
    ws_client.on("request_frames", lambda msg: handle_request_frames(msg, ws_client))
    ws_client.on("wake_timeout", lambda msg: handle_wake_timeout(msg, ww_filter))

    ws_task = asyncio.create_task(ws_client.run())

    frame_task = None
    if FORWARD_FRAMES:
        frame_task = asyncio.create_task(_frame_stream_task(ws_client))

    await _status.update(voice_assistant="idle", server_connection="inactive")

    await stt.start_stream()

    audio_seq = 0
    ffmpeg_proc = None
    _retry_delay = 0.05
    prev_ww_state = "IDLE"
    _consecutive_failures = 0
    _glasses_connected = False
    _disconnect_handled = False

    try:
        while True:
            if ffmpeg_proc is None or ffmpeg_proc.poll() is not None:
                if ffmpeg_proc is not None:
                    ffmpeg_proc.stdout.close()
                    ffmpeg_proc.stderr.close()
                    _consecutive_failures += 1
                else:
                    _consecutive_failures = 0

                if _consecutive_failures >= 3 and _glasses_connected and not _disconnect_handled:
                    _glasses_connected = False
                    _disconnect_handled = True
                    logger.info("[Session] Glasses disconnected. Awaiting reset decision...")

                    if RESET_SESSION == "true":
                        logger.info("[Session] Auto-resetting session (reset_on_disconnect=true)")
                        await ws_client.reset_session()
                        await _status.update(
                            voice_assistant="idle",
                            server_connection="inactive",
                            robot_status="N/A",
                        )
                    elif RESET_SESSION == "ask":
                        logger.info("[Session] Glasses disconnected. Awaiting reset decision...")

                await asyncio.sleep(_retry_delay)
                _retry_delay = min(_retry_delay * 2, 0.5)
                ffmpeg_proc = start_audio_decoder()

            chunk = await asyncio.get_event_loop().run_in_executor(
                None, ffmpeg_proc.stdout.read, CHUNK_SIZE
            )

            if not chunk:
                ffmpeg_proc = None
                continue

            _retry_delay = 0.05
            _consecutive_failures = 0

            if not _glasses_connected:
                _glasses_connected = True
                _disconnect_handled = False
                logger.info("[Bridge] Audio stream active")

            await stt.send_audio(chunk)

            # -- STT result handling ----------------------------------------
            transcription = await stt.get_transcription()
            if transcription:
                stripped = transcription.strip().lower()

                # During TTS: only react to wake words and stop commands
                if _tts_playing:
                    if ww_filter.contains_wake_word(transcription) or WakeWordFilter.is_stop_command(transcription):
                        logger.info(f"[STT] Barge-in detected: {transcription}")
                        _tts_cancel.set()
                    else:
                        logger.debug(f"[STT] Suppressed (TTS playing): {transcription}")
                    continue

                if stripped in _NOISE_TRANSCRIPTIONS:
                    logger.debug(f"[STT] Filtered noise: {transcription}")
                    continue

                accumulator.feed(transcription)

            # -- Process accumulated utterances ------------------------------
            utterance = accumulator.get()
            if utterance:
                logger.info(f"[STT] {utterance}")
                cleaned = ww_filter.process(utterance)

                cur_state = ww_filter.state
                if cur_state != prev_ww_state:
                    va = "listening" if cur_state == "ACTIVE" else "idle"
                    logger.info(f"[WakeWord] State: {prev_ww_state} -> {cur_state}")
                    await _status.update(voice_assistant=va)
                    prev_ww_state = cur_state

                if cleaned:
                    logger.info(f"[STT] -> NAT: {cleaned}")
                    await send_generic(cleaned, source="User")
                    await ws_client.send({
                        "type": "user_message",
                        "text": cleaned,
                    })

            if FORWARD_AUDIO:
                audio_seq += 1
                b64 = base64.b64encode(chunk).decode("ascii")
                await ws_client.send({
                    "type": "audio_stream",
                    "data": b64,
                    "sample_rate": SAMPLE_RATE,
                    "seq": audio_seq,
                })

    except asyncio.CancelledError:
        pass
    finally:
        await stt.stop_stream()
        await ws_client.stop()
        ws_task.cancel()
        if frame_task:
            frame_task.cancel()
        if ffmpeg_proc and ffmpeg_proc.poll() is None:
            ffmpeg_proc.terminate()
        logger.info("[Bridge] Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
