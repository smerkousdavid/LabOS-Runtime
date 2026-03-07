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
import audioop
import json
import os
import select
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

from loguru import logger

_log_dir = os.environ.get("LOG_DIR", "/app/logs")
if os.path.isdir(_log_dir):
    logger.add(os.path.join(_log_dir, "voice_bridge.log"), rotation="20 MB", retention="3 days", level="DEBUG")

from wakeword import WakeWordFilter
from stt_client import create_stt_client, STTClient
from ws_client import NATWebSocketClient
from status_manager import StatusManager
from session_recorder import SessionRecorder


def _load_tool_filter() -> set:
    """Load suppressed tool names from tool_filter.yaml."""
    import yaml
    from pathlib import Path
    for candidate in [Path("tool_filter.yaml"), Path(__file__).parent / "tool_filter.yaml"]:
        if candidate.is_file():
            try:
                with open(candidate) as f:
                    data = yaml.safe_load(f) or {}
                names = set(data.get("suppressed_tools", []))
                if names:
                    logger.info(f"[Bridge] Loaded tool filter: {len(names)} suppressed tools")
                return names
            except Exception as exc:
                logger.warning(f"[Bridge] Failed to load tool_filter.yaml: {exc}")
    return set()


_SUPPRESSED_TOOLS: set = _load_tool_filter()


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
GEMINI_AUDIO_FORWARD = os.environ.get("GEMINI_AUDIO_FORWARD", "false").lower() in ("true", "1", "yes")
INITIAL_QR_CODE = os.environ.get("INITIAL_QR_CODE", "false").lower() in ("true", "1", "yes")
FORWARD_FRAMES = os.environ.get("FORWARD_FRAMES", "false").lower() in ("true", "1", "yes")
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "480"))
FRAME_FPS = int(os.environ.get("FRAME_FPS", "15"))
WAKE_WORDS = os.environ.get("WAKE_WORDS", "stella,hey stella").split(",")
WAKE_TIMEOUT = float(os.environ.get("WAKE_TIMEOUT", "5"))
TTS_MODEL = os.environ.get("TTS_MODEL", "vibevoice")
RESET_SESSION = os.environ.get("RESET_SESSION_ON_DISCONNECT", "true").lower()
DISCONNECT_TIMEOUT_S = int(os.environ.get("DISCONNECT_TIMEOUT_SECONDS", "60"))
RTSP_EXTERNAL_HOST = os.environ.get("RTSP_EXTERNAL_HOST", "localhost")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "en")
STT_COMMIT_INTERVAL_S = float(os.environ.get("STT_COMMIT_INTERVAL_S", "0.25"))
STT_MIN_SPEECH_DURATION_MS = int(os.environ.get("STT_MIN_SPEECH_DURATION_MS", "500"))
STT_MIN_SILENCE_DURATION_MS = int(os.environ.get("STT_MIN_SILENCE_DURATION_MS", "500"))
STT_INCLUDE_TIMESTAMPS = os.environ.get("STT_INCLUDE_TIMESTAMPS", "true").lower() in ("true", "1", "yes")
STT_FALLBACK_PROTOCOL = os.environ.get("STT_FALLBACK_PROTOCOL", "")
STT_FALLBACK_HOST = os.environ.get("STT_FALLBACK_HOST", STT_HOST)
STT_FALLBACK_PORT = int(os.environ.get("STT_FALLBACK_PORT", str(STT_PORT)))
STT_FALLBACK_MODEL = os.environ.get("STT_FALLBACK_MODEL", STT_MODEL)
STT_FALLBACK_RECOVER_AFTER_S = float(os.environ.get("STT_FALLBACK_RECOVER_AFTER_S", "30"))
STT_NOISE_CORRECTION_ENABLED = os.environ.get("STT_NOISE_CORRECTION_ENABLED", "false").lower() in ("true", "1", "yes")
ENABLE_FAST_PATH = os.environ.get("ENABLE_FAST_PATH", "false").lower() in ("true", "1", "yes")
STT_NOISE_GATE_RMS = int(os.environ.get("STT_NOISE_GATE_RMS", "120"))
STT_NOISE_TERMS = os.environ.get(
    "STT_NOISE_SUPPRESSION_TERMS",
    "mhm,mm,mmm,hmm,hm,uh,um,ah,oh,uh-huh,mm-hmm,mhm mhm,so so,he,ha,hey,hi",
)
STT_SPAM_GUARD_WINDOW_S = float(os.environ.get("STT_SPAM_GUARD_WINDOW_S", "1.0"))

STT_EP_START_HISTORY = int(os.environ.get("STT_EP_START_HISTORY", "0"))
STT_EP_START_THRESHOLD = float(os.environ.get("STT_EP_START_THRESHOLD", "0.0"))
STT_EP_STOP_HISTORY = int(os.environ.get("STT_EP_STOP_HISTORY", "0"))
STT_EP_STOP_THRESHOLD = float(os.environ.get("STT_EP_STOP_THRESHOLD", "0.0"))
STT_EP_STOP_HISTORY_EOU = int(os.environ.get("STT_EP_STOP_HISTORY_EOU", "0"))
STT_EP_STOP_THRESHOLD_EOU = float(os.environ.get("STT_EP_STOP_THRESHOLD_EOU", "0.0"))

RECORDING_ENABLED = os.environ.get("RECORDING_ENABLED", "false").lower() in ("true", "1", "yes")
RECORDINGS_PATH = os.environ.get("RECORDINGS_PATH", "/app/recordings")
RECORDING_FPS = int(os.environ.get("RECORDING_FPS", "15"))

AUDIO_CHUNK_MS = 100
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_SIZE = int(SAMPLE_RATE * (AUDIO_CHUNK_MS / 1000.0) * BYTES_PER_SAMPLE)

AUDIO_SILENCE_DISCONNECT_S = float(os.environ.get("AUDIO_SILENCE_DISCONNECT_S", "5"))
AUDIO_READ_TIMEOUT_S = float(os.environ.get("AUDIO_READ_TIMEOUT_S", "5"))

AUDIO_MAX_DRIFT_S = float(os.environ.get("AUDIO_MAX_DRIFT_S", "1.0"))
AUDIO_BYTES_PER_SEC = SAMPLE_RATE * BYTES_PER_SAMPLE
AUDIO_KEEP_AFTER_DRAIN_S = 0.15
AUDIO_BURST_THRESHOLD_S = 0.01  # reads faster than this count as "burst"
AUDIO_BURST_DRAIN_AFTER = 5  # drain after N consecutive burst reads

HEARTBEAT_INTERVAL_S = float(os.environ.get("HEARTBEAT_INTERVAL_S", "10"))
FRAME_HEARTBEAT_INTERVAL_S = 2.0
FRAME_ABSENCE_DISCONNECT_S = float(os.environ.get("FRAME_ABSENCE_DISCONNECT_S", "4"))

_xr_conn = None
_xr_msg_conn = None  # dedicated connection for sending display messages
_recorder: Optional[SessionRecorder] = None
_status: Optional[StatusManager] = None
_ww_filter: Optional[WakeWordFilter] = None
_tts_playing = False
_tts_cancel = asyncio.Event()

# ---------------------------------------------------------------------------
# Fast-path patterns: detect "stella next step" / "stella previous step"
# in partial STT to fire immediately without accumulator debounce.
# Patterns match anywhere in the text (interims include full raw speech).
# ---------------------------------------------------------------------------
import re as _re

_FAST_NEXT_RE = _re.compile(
    r"(?:stella|hey\s*stella)\b.*\b(?:next\s*step|next|advance|skip|move\s*on)\b",
    _re.IGNORECASE,
)
_FAST_PREV_RE = _re.compile(
    r"(?:stella|hey\s*stella)\b.*\b(?:previous\s*step|prev\s*step|go\s*back|previous|back)\b",
    _re.IGNORECASE,
)
_FAST_QUESTION_RE = _re.compile(
    r"\b(what|when|how|why|where|which|tell\s*me|explain|describe)\b|\?",
    _re.IGNORECASE,
)

# Filler words / echo artefacts that ASR hallucinates from silence/noise
_NOISE_TRANSCRIPTIONS = frozenset(
    item.strip().lower()
    for item in STT_NOISE_TERMS.split(",")
    if item.strip()
)

# ---------------------------------------------------------------------------
# Parakeet STT misheard wake-word corrections
# ---------------------------------------------------------------------------
_QUESTION_GUARD_RE = _re.compile(
    r"\b(what|how|why|when|where|which|should|are\s+we|will|can|do\s+we)\b|\?",
    _re.IGNORECASE,
)

_COMMAND_INTENT_RE = _re.compile(
    r"\b(next|continue|start|stop|begin|protocol|step|previous|back|skip"
    r"|advance|move\s+on|repeat|open|close|run|go|pause|resume|end)\b",
    _re.IGNORECASE,
)

_STT_CORRECTIONS: list[tuple[_re.Pattern, str]] = [
    # "he/hey scala" -> "hey stella"
    (_re.compile(r"\bhe(?:y)?\s+scala\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bi\s+scala\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bwe\s+scala\b", _re.IGNORECASE), "hey stella"),
    # "scala" alone -> "stella"
    (_re.compile(r"\bscala\b", _re.IGNORECASE), "stella"),
    # "he/hey still a" -> "hey stella"
    (_re.compile(r"\bhe'?s\s+still\s+a\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bhe\s+still\s+a\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bhey\s+still\s+a\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bi\s+still\s+a\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bwe\s+still\s+a\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bhe'?s\s+still\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bhe\s+still\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bhey\s+still\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bi\s+still\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bwe\s+still\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bit'?s\s+still\s+a\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bis\s+still\s+a\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\ba\s+star\s+a\b", _re.IGNORECASE), "hey stella"),
    (_re.compile(r"\bstellar\b", _re.IGNORECASE), "stella"),
    (_re.compile(r"\bstell\b", _re.IGNORECASE), "stella"),
    (_re.compile(r"\bstill\s+a\b", _re.IGNORECASE), "stella"),
    (_re.compile(r"\ba\s+star\b", _re.IGNORECASE), "stella"),
    (_re.compile(r"\bit'?s\s+still\b", _re.IGNORECASE), "stella"),
    (_re.compile(r"\bis\s+still\b", _re.IGNORECASE), "stella"),
]


def _correct_stt_text(text: str) -> str:
    """Fix common Parakeet mishearings of 'stella' / 'hey stella'.

    Skips correction when the text contains question indicators -- unless the
    text also contains a command-intent word (e.g. 'next', 'start', 'protocol'),
    which means it's almost certainly directed at the system.
    """
    if not text:
        return text
    if _QUESTION_GUARD_RE.search(text) and not _COMMAND_INTENT_RE.search(text):
        return text
    corrected = text
    for pattern, replacement in _STT_CORRECTIONS:
        corrected = pattern.sub(replacement, corrected)
    if corrected != text:
        logger.debug(f"[STT] Corrected: '{text}' -> '{corrected}'")
    return corrected


def _get_xr_connection(*, reconnect: bool = False):
    global _xr_conn
    if reconnect and _xr_conn is not None:
        _xr_conn = None
    if _xr_conn is None:
        try:
            from xr_service_library import XRServiceConnection
            conn = XRServiceConnection(socket_path=SOCKET_PATH)
            conn.connect()
            _xr_conn = conn
            logger.info(f"[Bridge] Connected to XR service at {SOCKET_PATH}")
        except Exception as exc:
            logger.warning(f"[Bridge] XR service connection failed: {exc}")
    return _xr_conn


def _extract_numpy_frame(frame_data) -> Optional[np.ndarray]:
    """Extract a numpy array from the VideoFrame returned by XRServiceConnection."""
    if frame_data is None:
        return None
    if hasattr(frame_data, 'frame'):
        frame_obj = frame_data.frame
        if hasattr(frame_obj, 'data') and hasattr(frame_obj, 'dtype') and hasattr(frame_obj, 'shape'):
            return np.frombuffer(frame_obj.data, dtype=frame_obj.dtype).reshape(frame_obj.shape)
        return None
    if isinstance(frame_data, np.ndarray):
        return frame_data
    if isinstance(frame_data, dict) and 'image' in frame_data:
        return frame_data['image']
    return None


# ---------------------------------------------------------------------------
# Utterance accumulator -- debounce short ASR fragments
# ---------------------------------------------------------------------------

class UtteranceAccumulator:
    """Buffers short STT fragments and flushes after a silence gap.

    Short fragments (single word or < MIN_CHARS) are held in the buffer.
    Longer fragments cause an immediate flush (with any buffered prefix prepended).
    A background task flushes the buffer after DEBOUNCE_SECONDS of inactivity.
    """

    MIN_CHARS = int(os.environ.get("STT_UTTERANCE_MIN_CHARS", "4"))
    DEBOUNCE_SECONDS = float(os.environ.get("STT_UTTERANCE_DEBOUNCE_S", "0.4"))

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
# Send messages to glasses via XR service socket (direct, per-camera)
# ---------------------------------------------------------------------------

def _get_msg_connection(*, reconnect: bool = False):
    """Return (or create) a dedicated XR connection used only for send_message."""
    global _xr_msg_conn
    if reconnect and _xr_msg_conn is not None:
        _xr_msg_conn = None
    if _xr_msg_conn is None:
        try:
            from xr_service_library import XRServiceConnection
            conn = XRServiceConnection(socket_path=SOCKET_PATH)
            conn.connect()
            _xr_msg_conn = conn
            logger.info(f"[Bridge] Message connection established at {SOCKET_PATH}")
        except Exception as exc:
            logger.warning(f"[Bridge] Message connection failed: {exc}")
    return _xr_msg_conn


def _send_to_glasses_sync(message_type: str, payload: str) -> bool:
    """Send a message directly to this camera's glasses via the XR socket.

    Returns True if the message was accepted by the gRPC server.
    """
    conn = _get_msg_connection()
    if conn is None:
        logger.warning(f"[Bridge] No msg connection -- cannot send {message_type}")
        return False
    try:
        from xr_service_library.xr_types import Message
        msg = Message(message_type=message_type, payload=payload)
        result = conn.send_message(msg)
        if not result:
            logger.warning(f"[Bridge] send_message returned falsy for {message_type}")
            return False
        logger.info(f"[Bridge] Sent {message_type} to glasses ({len(payload)} bytes)")
        return True
    except Exception as exc:
        logger.warning(f"[Bridge] Send to glasses failed ({message_type}): {exc}")
        _get_msg_connection(reconnect=True)
        return False


async def _send_to_glasses(message_type: str, payload: str) -> bool:
    """Send a message to glasses (async wrapper around sync XR socket call)."""
    return await asyncio.get_event_loop().run_in_executor(
        None, _send_to_glasses_sync, message_type, payload,
    )


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
    max_delay_us = str(int(AUDIO_MAX_DRIFT_S * 1_000_000))
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-fflags", "nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-probesize", "32768",
        "-analyzeduration", "0",
        "-max_delay", max_delay_us,
        "-rtbufsize", "128k",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-af", "aresample=async=1",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-flush_packets", "1",
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
        if _recorder and _recorder.running:
            _recorder.log_chat("Agent", text)

    if tts and text:
        await _trigger_tts(text)


async def handle_notification(msg: dict):
    text = msg.get("text", "")
    tts = msg.get("tts", False)
    if text:
        await send_generic(text, source="Agent")
        if _recorder and _recorder.running:
            _recorder.log_chat("Agent", text)
    if tts and text:
        await _trigger_tts(text)


_last_display_update: Optional[tuple[str, str]] = None


async def handle_display_update(msg: dict):
    """Forward display_update directly to glasses via XR socket."""
    global _last_display_update
    payload_raw = msg.get("payload", "")
    if isinstance(payload_raw, (dict, list)):
        payload_str = json.dumps(payload_raw)
    else:
        payload_str = str(payload_raw)
    msg_type = msg.get("message_type", "GENERIC")
    _last_display_update = (msg_type, payload_str)
    if not _glasses_connected:
        return
    await _send_to_glasses(msg_type, payload_str)


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
                raw = conn.get_latest_frame()
                frame = _extract_numpy_frame(raw)
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
    if tool_name in _SUPPRESSED_TOOLS:
        return
    summary = msg.get("summary", "")
    status = msg.get("status", "started")

    if status == "started":
        text = f"<color=#59D2FF>Tool: {tool_name}</color> -- {summary}"
    elif status == "completed":
        text = f"<color=#88CC88>Tool: {tool_name}</color> -- done"
    else:
        text = f"<color=#FF4444>Tool: {tool_name}</color> -- {status}"

    await send_generic(text, source="Tool")

    if _recorder and _recorder.running:
        _recorder.log_data(f"Tool [{status}]: {tool_name} -- {summary}", user_facing=True)


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
        # if _ww_filter is not None:
        #     _ww_filter.touch()


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

            raw = conn.get_latest_frame()
            frame = _extract_numpy_frame(raw)
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
            if "context" in str(exc).lower():
                _get_xr_connection(reconnect=True)
            await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# Recording frame capture -- pushes frames to the SessionRecorder
# ---------------------------------------------------------------------------

_recording_task: Optional[asyncio.Task] = None


async def _recording_capture_task():
    """Capture frames from the shared XR connection and push to the recorder."""
    interval = 1.0 / max(1, RECORDING_FPS)
    logger.info(f"[Recorder] Frame capture task started at {RECORDING_FPS} fps")

    while True:
        try:
            if _recorder is None or not _recorder.running:
                await asyncio.sleep(0.5)
                continue

            conn = _get_xr_connection()
            if conn is None:
                await asyncio.sleep(1.0)
                continue

            raw = conn.get_latest_frame()
            frame = _extract_numpy_frame(raw)
            if frame is not None:
                _recorder.push_frame(frame)

            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning(f"[Recorder] Frame capture error: {exc}")
            if "context" in str(exc).lower():
                _get_xr_connection(reconnect=True)
            await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# QR Code scanning
# ---------------------------------------------------------------------------

_qr_scanning_active = False
_qr_scan_current_task: Optional[asyncio.Task] = None
_qr_cooldown_until: float = 0.0  # monotonic timestamp; scanning blocked until this
_QR_SESSION_COOLDOWN_S = float(os.environ.get("QR_SESSION_COOLDOWN_S", "15"))
_rtsp_relay_proc = None
_rtsp_relay_monitor_task: Optional[asyncio.Task] = None
_glasses_connected = False
_QR_SHOW_COMMANDS = {"show qr code", "show qr"}
_QR_QUIT_COMMANDS = {"quit app"}


def _decode_qr(frame) -> list[str]:
    """Decode QR codes from a frame using pyzbar (primary) or OpenCV (fallback).

    Returns a list of decoded string payloads (may be empty).
    """
    try:
        from pyzbar.pyzbar import decode as zbar_decode, ZBarSymbol
        results = zbar_decode(frame, symbols=[ZBarSymbol.QRCODE])
        return [r.data.decode("utf-8", errors="replace") for r in results]
    except ImportError:
        pass
    except Exception as exc:
        logger.debug(f"[QR] pyzbar error, falling back to cv2: {exc}")

    import cv2
    try:
        detector = cv2.QRCodeDetector()
        retval, _ = detector.detectAndDecodeMulti(frame)
        if retval:
            return [s for s in retval if s]
    except Exception:
        data, _, _ = detector.detectAndDecode(frame)
        if data:
            return [data]
    return []


async def _qr_scan_task(ws_client: NATWebSocketClient):
    """Scan camera for QR codes at ~4 FPS, show preview on AR panel."""
    global _qr_scanning_active
    _qr_scanning_active = True
    import cv2
    interval = 0.25
    _prompt_msg = (
        "<size=16><color=#59D2FF>Point at the QR code on screen</color></size>"
        '<size=14><color=#CCCCCC>\nSay "show qr code" or "quit app" to restart session</color></size>'
    )
    _waiting_msg = (
        '<size=16><color=#888888>Waiting for camera...</color></size>'
        '<size=14><color=#CCCCCC>\nSay "show qr code" or "quit app"</color></size>'
    )

    logger.info("[QR] QR code scanning started")
    _last_sent_qr: Optional[str] = None
    _last_sent_qr_ts: float = 0.0
    _QR_COOLDOWN_S = 5.0  # ignore same QR for N seconds after sending

    while _qr_scanning_active:
        try:
            if not _glasses_connected:
                await asyncio.sleep(interval)
                continue

            if time.monotonic() < _qr_cooldown_until:
                await asyncio.sleep(interval)
                continue

            conn = _get_xr_connection()
            frame = None
            if conn is not None:
                raw = conn.get_latest_frame()
                frame = _extract_numpy_frame(raw)

            if frame is not None:
                h, w = frame.shape[:2]
                if w > 480 or h > 480:
                    scale = 480.0 / max(w, h)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                decoded_list = _decode_qr(frame)
                for data in decoded_list:
                    if (
                        data == _last_sent_qr
                        and (time.monotonic() - _last_sent_qr_ts) < _QR_COOLDOWN_S
                    ):
                        continue

                    logger.info(f"[QR] Decoded payload: {data[:200]}")
                    try:
                        payload = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        payload = data

                    # Determine if this looks like a valid LabOS QR:
                    #   - compact: dict with "h" and "s" keys
                    #   - verbose: dict with "type"=="labos_live"
                    #   - pairing code: short string (6-char or numeric)
                    #   - any dict with "t"=="ll"
                    is_valid = False
                    if isinstance(payload, dict):
                        is_valid = (
                            payload.get("t") == "ll"
                            or payload.get("type") == "labos_live"
                            or ("h" in payload and "s" in payload)
                        )
                    elif isinstance(payload, str):
                        is_valid = len(payload.strip()) >= 4

                    if is_valid:
                        global _qr_cooldown_until
                        logger.info(f"[QR] Sending QR payload to NAT")
                        await ws_client.send({
                            "type": "qr_payload",
                            "payload": payload,
                        })
                        _last_sent_qr = data
                        _last_sent_qr_ts = time.monotonic()
                        _qr_cooldown_until = time.monotonic() + _QR_SESSION_COOLDOWN_S
                        logger.info(f"[QR] Cooldown set for {_QR_SESSION_COOLDOWN_S}s after sending payload")
                        _qr_scanning_active = False
                        break
                    else:
                        logger.info(f"[QR] Ignoring unrecognized QR data: {data[:80]}")

                if not _qr_scanning_active:
                    break

                _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                preview_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")
                messages = [
                    {"type": "base64-image", "content": preview_b64},
                    {"type": "rich-text", "content": _prompt_msg},
                ]
            else:
                messages = [
                    {"type": "rich-text", "content": _prompt_msg},
                    {"type": "rich-text", "content": _waiting_msg},
                ]

            panel_payload = json.dumps({"messages": messages})
            await _send_to_glasses("SINGLE_STEP_PANEL_CONTENT", panel_payload)

            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning(f"[QR] Scan error: {exc}")
            await asyncio.sleep(1.0)

    logger.info("[QR] QR code scanning stopped")


def stop_qr_scanning():
    global _qr_scanning_active, _qr_scan_current_task
    _qr_scanning_active = False
    if _qr_scan_current_task is not None and not _qr_scan_current_task.done():
        _qr_scan_current_task.cancel()
    _qr_scan_current_task = None


def start_qr_scanning():
    global _qr_scanning_active
    _qr_scanning_active = True


def _launch_qr_scan(ws_client, *, force: bool = False) -> None:
    """Start a QR scan task, cancelling any existing one first.

    Respects the session cooldown unless *force* is True.
    """
    global _qr_scan_current_task
    if not force and time.monotonic() < _qr_cooldown_until:
        remaining = _qr_cooldown_until - time.monotonic()
        logger.info(f"[QR] Launch blocked -- cooldown active ({remaining:.1f}s remaining)")
        return
    stop_qr_scanning()
    start_qr_scanning()
    _qr_scan_current_task = asyncio.create_task(_qr_scan_task(ws_client))


# ---------------------------------------------------------------------------
# RTSP relay to LabOS server
# ---------------------------------------------------------------------------

_RTSP_RELAY_TIMEOUT_S = float(os.environ.get("RTSP_RELAY_TIMEOUT_S", "180"))
_RTSP_RELAY_RETRY_DELAY = 2.0
_rtsp_relay_ws_client: Optional[object] = None


async def _drain_stderr(proc: asyncio.subprocess.Process, label: str):
    """Read ffmpeg stderr line-by-line and log at WARNING level."""
    assert proc.stderr is not None
    while True:
        line = await proc.stderr.readline()
        if not line:
            break
        logger.warning(f"[RTSP] {label}: {line.decode(errors='replace').rstrip()}")


async def _start_rtsp_relay(
    local_rtsp_path: str, publish_rtsp: str, ws_client=None,
):
    """Start FFmpeg relay with time-based retry while the WS connection is alive."""
    global _rtsp_relay_proc, _rtsp_relay_monitor_task, _rtsp_relay_ws_client
    await _stop_rtsp_relay()
    _rtsp_relay_ws_client = ws_client

    local_url = f"rtsp://{MEDIAMTX_HOST}:8554/{local_rtsp_path}"
    max_delay_us = str(int(AUDIO_MAX_DRIFT_S * 1_000_000))
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "nobuffer+genpts",
        "-flags", "low_delay",
        "-probesize", "5000000",
        "-analyzeduration", "3000000",
        "-max_delay", max_delay_us,
        "-rtbufsize", "2M",
        "-rtsp_transport", "tcp",
        "-i", local_url,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-avoid_negative_ts", "make_zero",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        publish_rtsp,
    ]

    async def _ws_alive() -> bool:
        if _rtsp_relay_ws_client is None:
            return True
        return getattr(_rtsp_relay_ws_client, "connected", True)

    async def _run_with_retry():
        global _rtsp_relay_proc
        start = time.monotonic()
        attempt = 0
        while (time.monotonic() - start) < _RTSP_RELAY_TIMEOUT_S:
            if not await _ws_alive():
                logger.info("[RTSP] WS disconnected -- stopping relay retries")
                break
            attempt += 1
            elapsed = time.monotonic() - start
            logger.info(
                f"[RTSP] Starting relay (attempt {attempt}, "
                f"{elapsed:.0f}s/{_RTSP_RELAY_TIMEOUT_S:.0f}s): "
                f"{local_url} -> {publish_rtsp}"
            )
            try:
                _rtsp_relay_proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
            except Exception as exc:
                logger.error(f"[RTSP] Failed to spawn ffmpeg: {exc}")
                _rtsp_relay_proc = None
                await asyncio.sleep(_RTSP_RELAY_RETRY_DELAY)
                continue

            logger.info(f"[RTSP] Relay started (pid={_rtsp_relay_proc.pid})")
            stderr_task = asyncio.create_task(_drain_stderr(_rtsp_relay_proc, "ffmpeg"))

            exit_code = await _rtsp_relay_proc.wait()
            await stderr_task

            if exit_code == 0:
                logger.info("[RTSP] Relay exited cleanly (exit_code=0)")
                _rtsp_relay_proc = None
                return

            logger.warning(f"[RTSP] Relay exited (exit_code={exit_code})")
            _rtsp_relay_proc = None

            remaining = _RTSP_RELAY_TIMEOUT_S - (time.monotonic() - start)
            if remaining > _RTSP_RELAY_RETRY_DELAY:
                logger.info(
                    f"[RTSP] Retrying in {_RTSP_RELAY_RETRY_DELAY}s "
                    f"({remaining:.0f}s remaining)..."
                )
                await asyncio.sleep(_RTSP_RELAY_RETRY_DELAY)

        elapsed = time.monotonic() - start
        logger.error(
            f"[RTSP] Relay gave up after {attempt} attempts / {elapsed:.0f}s"
        )

    _rtsp_relay_monitor_task = asyncio.create_task(_run_with_retry())


async def _stop_rtsp_relay():
    """Stop the RTSP relay FFmpeg process and cancel the monitor task."""
    global _rtsp_relay_proc, _rtsp_relay_monitor_task

    if _rtsp_relay_monitor_task is not None:
        _rtsp_relay_monitor_task.cancel()
        try:
            await _rtsp_relay_monitor_task
        except asyncio.CancelledError:
            pass
        _rtsp_relay_monitor_task = None

    if _rtsp_relay_proc is not None:
        try:
            _rtsp_relay_proc.terminate()
            await asyncio.wait_for(_rtsp_relay_proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                _rtsp_relay_proc.kill()
            except ProcessLookupError:
                pass
        logger.info("[RTSP] Relay stopped")
        _rtsp_relay_proc = None


async def _handle_session_connected(msg: dict, ws_client):
    """LabOS Live session confirmed -- start RTSP relay."""
    publish_rtsp = msg.get("publish_rtsp", "")
    session_id = msg.get("session_id", "")
    logger.info(f"[Session] LabOS Live connected: {session_id}")

    stop_qr_scanning()

    if publish_rtsp:
        idx = f"{CAMERA_INDEX:04d}"
        local_path = f"NB_{idx}_TX_CAM_RGB_MIC_p6S"
        await _start_rtsp_relay(local_path, publish_rtsp, ws_client=ws_client)


async def _handle_session_cleared(ws_client):
    """LabOS Live session ended -- stop relay, return to QR scanning."""
    logger.info("[Session] LabOS Live session cleared")
    await _stop_rtsp_relay()

    if INITIAL_QR_CODE:
        _launch_qr_scan(ws_client)


async def _handle_session_connect_failed(msg: dict, ws_client):
    """LabOS Live session failed to connect -- stop relay, return to QR scanning."""
    reason = msg.get("reason", "unknown")
    logger.warning(f"[Session] LabOS Live connect failed: {reason}")
    await _stop_rtsp_relay()

    if INITIAL_QR_CODE:
        _launch_qr_scan(ws_client)


async def _restart_qr_flow(ws_client: NATWebSocketClient, *, reset_ws: bool, reason: str) -> None:
    """Stop current live session state and return to QR scanning."""
    global _last_display_update
    logger.info(f"[QR] Restart requested: {reason} reset_ws={reset_ws}")
    stop_qr_scanning()
    await _stop_rtsp_relay()
    _last_display_update = None
    if reset_ws:
        await ws_client.reset_session()
    if INITIAL_QR_CODE:
        _launch_qr_scan(ws_client)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main():
    global _status, _ww_filter, _recorder, _recording_task, _last_display_update

    logger.info(f"[Bridge] Starting for camera {CAMERA_INDEX}, session {SESSION_ID}")

    if RECORDING_ENABLED:
        _recorder = SessionRecorder(
            camera_index=CAMERA_INDEX,
            recordings_root=RECORDINGS_PATH,
            width=1280,
            height=720,
            framerate=RECORDING_FPS,
        )
        logger.info(f"[Recorder] Enabled -- recordings at {RECORDINGS_PATH}")

    # Log RTSP URLs for external access (VLC, NAT server, etc.)
    idx = f"{CAMERA_INDEX:04d}"
    logger.info(f"[RTSP] Video : rtsp://{RTSP_EXTERNAL_HOST}:8554/NB_{idx}_TX_CAM_RGB")
    logger.info(f"[RTSP] Audio : rtsp://{RTSP_EXTERNAL_HOST}:8554/NB_{idx}_TX_MIC_p6S")
    logger.info(f"[RTSP] Merged: rtsp://{RTSP_EXTERNAL_HOST}:8554/NB_{idx}_TX_CAM_RGB_MIC_p6S")

    _status = StatusManager(_send_to_glasses)

    ww_filter = WakeWordFilter(
        wake_words=WAKE_WORDS,
        timeout_seconds=WAKE_TIMEOUT,
        noise_words=_NOISE_TRANSCRIPTIONS,
    )
    _ww_filter = ww_filter

    accumulator = UtteranceAccumulator()

    _endpointing_cfg = {}
    # if STT_EP_STOP_HISTORY:
    #     _endpointing_cfg["stop_history"] = STT_EP_STOP_HISTORY
    # if STT_EP_STOP_THRESHOLD:
    #     _endpointing_cfg["stop_threshold"] = STT_EP_STOP_THRESHOLD
    # if STT_EP_STOP_HISTORY_EOU:
    #     _endpointing_cfg["stop_history_eou"] = STT_EP_STOP_HISTORY_EOU
    # if STT_EP_STOP_THRESHOLD_EOU:
    #     _endpointing_cfg["stop_threshold_eou"] = STT_EP_STOP_THRESHOLD_EOU
    # if STT_EP_START_HISTORY:
    #     _endpointing_cfg["start_history"] = STT_EP_START_HISTORY
    # if STT_EP_START_THRESHOLD:
    #     _endpointing_cfg["start_threshold"] = STT_EP_START_THRESHOLD

    stt = create_stt_client({
        "host": STT_HOST,
        "port": STT_PORT,
        "protocol": STT_PROTOCOL,
        "model": STT_MODEL,
        "language": STT_LANGUAGE,
        "commit_interval_s": STT_COMMIT_INTERVAL_S,
        "min_speech_duration_ms": STT_MIN_SPEECH_DURATION_MS,
        "min_silence_duration_ms": STT_MIN_SILENCE_DURATION_MS,
        "include_timestamps": STT_INCLUDE_TIMESTAMPS,
        "fallback_recover_after_s": STT_FALLBACK_RECOVER_AFTER_S,
        "endpointing": _endpointing_cfg or None,
        "fallback": {
            "protocol": STT_FALLBACK_PROTOCOL,
            "host": STT_FALLBACK_HOST,
            "port": STT_FALLBACK_PORT,
            "model": STT_FALLBACK_MODEL,
        },
    })

    def _on_ws_connect():
        if _glasses_connected:
            asyncio.ensure_future(_status.update(server_connection="active"))

    async def _on_ws_disconnect():
        await _status.update(server_connection="inactive")
        await _stop_rtsp_relay()

    ws_client = NATWebSocketClient(
        url=NAT_SERVER_URL,
        session_id=SESSION_ID,
        camera_index=CAMERA_INDEX,
        rtsp_base=f"rtsp://{RTSP_EXTERNAL_HOST}:8554",
        on_connect=_on_ws_connect,
        on_disconnect=lambda: asyncio.ensure_future(_on_ws_disconnect()),
    )

    ws_client.on("agent_response", handle_agent_response)
    ws_client.on("notification", handle_notification)
    ws_client.on("display_update", handle_display_update)
    ws_client.on("tts_only", handle_tts_only)
    ws_client.on("tool_call", handle_tool_call)
    ws_client.on("request_frames", lambda msg: handle_request_frames(msg, ws_client))
    ws_client.on("wake_timeout", lambda msg: handle_wake_timeout(msg, ww_filter))
    ws_client.on("session_connected", lambda msg: _handle_session_connected(msg, ws_client))
    ws_client.on("session_cleared", lambda msg: _handle_session_cleared(ws_client))
    ws_client.on("session_connect_failed", lambda msg: _handle_session_connect_failed(msg, ws_client))

    ws_task = asyncio.create_task(ws_client.run())

    if INITIAL_QR_CODE:
        _launch_qr_scan(ws_client)

    frame_task = None
    if FORWARD_FRAMES:
        frame_task = asyncio.create_task(_frame_stream_task(ws_client))

    await _status.update(voice_assistant="idle", server_connection="inactive")

    await stt.start_stream()
    logger.info(
        f"[STT] Config protocol={STT_PROTOCOL} model={STT_MODEL} "
        f"commit={STT_COMMIT_INTERVAL_S:.2f}s min_speech_ms={STT_MIN_SPEECH_DURATION_MS} "
        f"min_silence_ms={STT_MIN_SILENCE_DURATION_MS} "
        f"noise_correction={STT_NOISE_CORRECTION_ENABLED} noise_gate_rms={STT_NOISE_GATE_RMS}"
    )

    audio_seq = 0
    ffmpeg_proc = None
    global _glasses_connected
    _retry_delay = 0.05
    prev_ww_state = "IDLE"
    _glasses_connected = False
    _disconnect_handled = False
    _ever_connected = False  # True after first glasses connection
    _last_sent_utterance = ""
    _last_sent_utterance_ts = 0.0
    _last_good_chunk_ts = 0.0

    # ------------------------------------------------------------------
    # Heartbeat: periodically re-push status + cached display to glasses.
    # Catches races where the initial push arrives before the UI is ready
    # or the gRPC connection silently went stale.
    # ------------------------------------------------------------------
    async def _heartbeat_task():
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            if not _glasses_connected:
                continue
            _get_msg_connection(reconnect=True)
            await _status.force_push()
            if _last_display_update:
                mt, pl = _last_display_update
                await _send_to_glasses(mt, pl)

    asyncio.create_task(_heartbeat_task())

    # ------------------------------------------------------------------
    # Frame-based disconnect detection: polls IPC frames and triggers a
    # full disconnect / fresh-session cycle when the frame stream goes
    # away (glasses app closed) and then resumes (glasses app reopened).
    # ------------------------------------------------------------------
    _last_frame_ts: float = 0.0
    _frame_disconnect_done: bool = False

    async def _do_glasses_disconnect(reason: str):
        """Full disconnect: stop STT, clear state, reset WS session."""
        global _last_display_update, _glasses_connected
        nonlocal _frame_disconnect_done, _disconnect_handled
        if _frame_disconnect_done:
            return
        _frame_disconnect_done = True
        _glasses_connected = False
        _disconnect_handled = True

        logger.info(f"[Session] Glasses disconnected ({reason})")

        await stt.stop_stream()
        accumulator._buffer.clear()
        ww_filter.reset()
        _tts_cancel.set()

        if _recording_task and not _recording_task.done():
            _recording_task.cancel()
        if _recorder and _recorder.running:
            _recorder.stop()

        await _stop_rtsp_relay()
        stop_qr_scanning()

        _last_display_update = None

        await _status.update(
            voice_assistant="idle",
            server_connection="inactive",
            robot_status="N/A",
        )

        logger.info("[Session] Resetting WS session for clean reconnect")
        await ws_client.reset_session()

    async def _do_glasses_reconnect():
        """Full reconnect: fresh IPC + WS is auto-reconnected + QR flow."""
        global _glasses_connected
        nonlocal _frame_disconnect_done, _disconnect_handled
        _frame_disconnect_done = False
        _glasses_connected = True
        _disconnect_handled = False

        _get_xr_connection(reconnect=True)
        _get_msg_connection(reconnect=True)

        logger.info("[Session] Glasses reconnected -- starting fresh session")

        await stt.start_stream()

        _status.voice_assistant = "idle"
        _status.server_connection = "active" if ws_client.connected else "inactive"
        await _status.force_push()

        # Re-push status after UI has time to initialize
        async def _deferred_reconnect_push():
            await asyncio.sleep(3)
            if _glasses_connected:
                logger.info("[Session] Deferred reconnect push (3s)")
                _get_msg_connection(reconnect=True)
                await _status.force_push()
        asyncio.create_task(_deferred_reconnect_push())

        if INITIAL_QR_CODE:
            _launch_qr_scan(ws_client)
            logger.info("[Session] QR scanning restarted for fresh session")

    async def _frame_heartbeat_task():
        nonlocal _last_frame_ts, _frame_disconnect_done
        _frame_was_absent = False
        while True:
            await asyncio.sleep(FRAME_HEARTBEAT_INTERVAL_S)
            conn = _get_xr_connection()
            if conn is None:
                continue
            try:
                raw = await asyncio.get_event_loop().run_in_executor(
                    None, conn.get_latest_frame,
                )
                frame = _extract_numpy_frame(raw)
            except Exception:
                frame = None

            now = time.monotonic()
            if frame is not None:
                if _frame_was_absent:
                    logger.info("[Heartbeat] Frames resumed after absence")
                    await _do_glasses_reconnect()
                _frame_was_absent = False
                _last_frame_ts = now
            else:
                if (
                    _last_frame_ts > 0
                    and (now - _last_frame_ts) >= FRAME_ABSENCE_DISCONNECT_S
                    and not _frame_was_absent
                ):
                    _frame_was_absent = True
                    logger.info(
                        f"[Heartbeat] No frames for "
                        f"{now - _last_frame_ts:.1f}s -- glasses disconnected"
                    )
                    await _do_glasses_disconnect(
                        f"no frames for {now - _last_frame_ts:.1f}s"
                    )

    asyncio.create_task(_frame_heartbeat_task())

    _drift_wall_start: float = 0.0
    _drift_bytes_read: int = 0
    _burst_count: int = 0
    _last_read_start: float = 0.0
    _drift_log_interval: float = 30.0
    _last_drift_log: float = 0.0

    try:
        while True:
            if ffmpeg_proc is None or ffmpeg_proc.poll() is not None:
                _drift_wall_start = 0.0
                _drift_bytes_read = 0
                _burst_count = 0
                if ffmpeg_proc is not None:
                    ffmpeg_proc.stdout.close()
                    ffmpeg_proc.stderr.close()

                if (
                    _glasses_connected
                    and not _disconnect_handled
                    and _last_good_chunk_ts > 0
                    and (time.monotonic() - _last_good_chunk_ts) >= AUDIO_SILENCE_DISCONNECT_S
                ):
                    silence_s = time.monotonic() - _last_good_chunk_ts
                    await _do_glasses_disconnect(
                        f"no audio for {silence_s:.1f}s"
                    )
                    if _recorder:
                        _recorder.log_data(f"Glasses disconnected (no audio for {silence_s:.1f}s)")

                await asyncio.sleep(_retry_delay)
                _retry_delay = min(_retry_delay * 2, 1.5)
                ffmpeg_proc = start_audio_decoder()

            _last_read_start = time.monotonic()
            try:
                chunk = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, ffmpeg_proc.stdout.read, CHUNK_SIZE
                    ),
                    timeout=AUDIO_READ_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning("[Bridge] Audio decoder stalled -- restarting")
                if _recorder and _recorder.running:
                    _recorder.log_error("Audio decoder stalled -- restarting FFmpeg")
                try:
                    ffmpeg_proc.kill()
                    ffmpeg_proc.wait(timeout=2)
                except Exception:
                    pass
                try:
                    ffmpeg_proc.stdout.close()
                    ffmpeg_proc.stderr.close()
                except Exception:
                    pass
                ffmpeg_proc = None
                continue

            if not chunk:
                try:
                    ffmpeg_proc.wait(timeout=2)
                except Exception:
                    pass
                ffmpeg_proc = None
                continue

            read_elapsed = time.monotonic() - _last_read_start
            _last_good_chunk_ts = time.monotonic()
            _retry_delay = 0.05

            # Burst detection: if the read returned almost instantly,
            # ffmpeg had buffered data (pipeline latency accumulation).
            if read_elapsed < AUDIO_BURST_THRESHOLD_S:
                _burst_count += 1
            else:
                _burst_count = 0

            if _burst_count >= AUDIO_BURST_DRAIN_AFTER:
                fd = ffmpeg_proc.stdout.fileno()
                drained = 0
                while True:
                    ready, _, _ = select.select([fd], [], [], 0)
                    if not ready:
                        break
                    discarded = os.read(fd, 65536)
                    if not discarded:
                        break
                    drained += len(discarded)
                if drained > 0:
                    await stt.clear_buffer()
                    logger.warning(
                        f"[Bridge] Burst drain: {_burst_count} fast reads, "
                        f"discarded {drained}B ({drained/AUDIO_BYTES_PER_SEC:.2f}s)"
                    )
                _burst_count = 0
                _drift_wall_start = time.monotonic()
                _drift_bytes_read = 0
                continue

            # --- Drift detection: skip stale audio when pipeline falls behind ---
            now_mono = _last_good_chunk_ts
            if _drift_wall_start == 0.0:
                _drift_wall_start = now_mono
                _drift_bytes_read = 0
            _drift_bytes_read += len(chunk)
            audio_duration = _drift_bytes_read / AUDIO_BYTES_PER_SEC
            wall_elapsed = now_mono - _drift_wall_start
            drift = audio_duration - wall_elapsed
            if drift > AUDIO_MAX_DRIFT_S:
                # Smart partial drain: discard exactly enough bytes to land
                # back at real-time, but keep the most recent audio so the
                # STT stream stays continuous.
                keep_bytes = int(AUDIO_KEEP_AFTER_DRAIN_S * AUDIO_BYTES_PER_SEC)
                target_drain = int(drift * AUDIO_BYTES_PER_SEC) - keep_bytes
                drained = 0
                fd = ffmpeg_proc.stdout.fileno()
                while drained < target_drain:
                    ready, _, _ = select.select([fd], [], [], 0)
                    if not ready:
                        break
                    to_read = min(65536, target_drain - drained)
                    discarded = os.read(fd, to_read)
                    if not discarded:
                        break
                    drained += len(discarded)
                await stt.clear_buffer()
                _drift_wall_start = time.monotonic()
                _drift_bytes_read = keep_bytes
                logger.warning(
                    f"[Bridge] Audio drift {drift:.2f}s > {AUDIO_MAX_DRIFT_S}s "
                    f"-- drained {drained}B ({drained/AUDIO_BYTES_PER_SEC:.2f}s), "
                    f"kept ~{AUDIO_KEEP_AFTER_DRAIN_S}s tail"
                )
                continue

            now_for_log = time.monotonic()
            if now_for_log - _last_drift_log >= _drift_log_interval:
                _inner = stt
                if hasattr(_inner, '_primary'):
                    _inner = getattr(_inner, '_primary', _inner)
                stt_qsize = getattr(_inner, '_audio_queue', None)
                qd = stt_qsize.qsize() if stt_qsize is not None else "?"
                logger.info(
                    f"[Bridge] Audio stats: drift={drift:.2f}s "
                    f"read_ms={read_elapsed*1000:.0f} "
                    f"stt_queue={qd} burst={_burst_count}"
                )
                _last_drift_log = now_for_log

            if not _glasses_connected:
                # If the frame heartbeat already handled this reconnect,
                # _frame_disconnect_done will be False (cleared by _do_glasses_reconnect).
                # Otherwise, this is the first detection — do a full fresh start.
                if _frame_disconnect_done:
                    # Frame heartbeat already triggered disconnect but hasn't
                    # seen frames resume yet.  Let the frame heartbeat handle
                    # the reconnect to avoid a race.
                    continue

                _glasses_connected = True
                _disconnect_handled = False
                _frame_disconnect_done = False

                _get_xr_connection(reconnect=True)
                _get_msg_connection(reconnect=True)

                if _ever_connected:
                    _last_display_update = None
                    logger.info("[Bridge] Glasses reconnected -- resetting WS for fresh session")
                    await ws_client.reset_session()
                else:
                    logger.info("[Bridge] First glasses connection -- audio stream active")
                _ever_connected = True

                await stt.start_stream()
                _last_frame_ts = time.monotonic()

                _status.voice_assistant = "idle"
                _status.server_connection = "active" if ws_client.connected else "inactive"
                await _status.force_push()

                async def _deferred_initial_push():
                    await asyncio.sleep(3)
                    if _glasses_connected:
                        logger.info("[Bridge] Deferred initial push (3s)")
                        _get_msg_connection(reconnect=True)
                        await _status.force_push()
                asyncio.create_task(_deferred_initial_push())

                if INITIAL_QR_CODE:
                    _launch_qr_scan(ws_client)
                    logger.info("[Session] QR scanning started")

                if _recorder and not _recorder.running:
                    _recorder.start()
                    if _recording_task is None or _recording_task.done():
                        _recording_task = asyncio.create_task(_recording_capture_task())
                if _recorder and _recorder.running:
                    _recorder.log_data("Glasses connected -- audio stream active")

            if STT_NOISE_CORRECTION_ENABLED:
                try:
                    chunk_rms = audioop.rms(chunk, BYTES_PER_SAMPLE)
                except Exception:
                    chunk_rms = STT_NOISE_GATE_RMS + 1
                if chunk_rms < STT_NOISE_GATE_RMS:
                    continue

            await stt.send_audio(chunk)

            # -- Interim processing -----------------------------------------
            interim = getattr(stt, "get_last_interim", lambda: "")()
            if interim:
                interim = _correct_stt_text(interim)

            # Interim wake-word detection: if interim contains the wake word,
            # pre-activate the filter so the final text passes through even
            # when Riva's language model drops "stella" from the final.
            if interim and ww_filter.contains_wake_word(interim):
                if _tts_playing:
                    logger.info(f"[STT] Interim barge-in: {interim}")
                    _tts_cancel.set()
                    if _recorder and _recorder.running:
                        _recorder.log_data(f"TTS barge-in: {interim}", user_facing=True)
                    if hasattr(stt, "_last_interim"):
                        stt._last_interim = ""
                elif ww_filter.state != "ACTIVE":
                    logger.info(f"[WakeWord] Pre-activated from interim: {interim}")
                    ww_filter._activate()
                    await _status.update(voice_assistant="listening")
                    prev_ww_state = "ACTIVE"
                    if _recorder and _recorder.running:
                        _recorder.log_data(f"Wake word detected: {interim}", user_facing=True)
            elif interim and _tts_playing and WakeWordFilter.is_stop_command(interim):
                logger.info(f"[STT] Interim stop command: {interim}")
                _tts_cancel.set()

            # Fast-path on interims (before waiting for final)
            if ENABLE_FAST_PATH and interim and not _tts_playing:
                if not _FAST_QUESTION_RE.search(interim):
                    if _FAST_NEXT_RE.search(interim):
                        logger.info(f"[FastPath] next_step from interim: {interim}")
                        await ws_client.send({"type": "fast_command", "command": "next_step"})
                        await send_generic("next step", source="User")
                        accumulator._buffer.clear()
                        ww_filter.deactivate()
                        await _status.update(voice_assistant="idle")
                        prev_ww_state = "IDLE"
                        if hasattr(stt, "_last_interim"):
                            stt._last_interim = ""
                        _ = await stt.get_transcription()
                        continue
                    if _FAST_PREV_RE.search(interim):
                        logger.info(f"[FastPath] previous_step from interim: {interim}")
                        await ws_client.send({"type": "fast_command", "command": "previous_step"})
                        await send_generic("previous step", source="User")
                        accumulator._buffer.clear()
                        ww_filter.deactivate()
                        await _status.update(voice_assistant="idle")
                        prev_ww_state = "IDLE"
                        if hasattr(stt, "_last_interim"):
                            stt._last_interim = ""
                        _ = await stt.get_transcription()
                        continue

            # -- STT result handling (finals) --------------------------------
            transcription = await stt.get_transcription()
            if transcription:
                transcription = _correct_stt_text(transcription)
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

                # Fast path on final text too (fallback if interim didn't catch it)
                if ENABLE_FAST_PATH and not _FAST_QUESTION_RE.search(transcription):
                    if _FAST_NEXT_RE.search(transcription):
                        logger.info(f"[FastPath] next_step from final: {transcription}")
                        await ws_client.send({"type": "fast_command", "command": "next_step"})
                        await send_generic("next step", source="User")
                        accumulator._buffer.clear()
                        ww_filter.deactivate()
                        await _status.update(voice_assistant="idle")
                        prev_ww_state = "IDLE"
                        continue
                    if _FAST_PREV_RE.search(transcription):
                        logger.info(f"[FastPath] previous_step from final: {transcription}")
                        await ws_client.send({"type": "fast_command", "command": "previous_step"})
                        await send_generic("previous step", source="User")
                        accumulator._buffer.clear()
                        ww_filter.deactivate()
                        await _status.update(voice_assistant="idle")
                        prev_ww_state = "IDLE"
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
                    normalized_cmd = " ".join(cleaned.strip().lower().split())
                    if INITIAL_QR_CODE and normalized_cmd in _QR_SHOW_COMMANDS:
                        logger.info(f"[QR] Voice command received: {cleaned}")
                        await send_generic(cleaned, source="User")
                        await _restart_qr_flow(
                            ws_client, reset_ws=False, reason="voice command: show qr code",
                        )
                        await send_generic("QR scanner restarted.", source="System")
                        ww_filter.deactivate()
                        await _status.update(voice_assistant="idle")
                        prev_ww_state = "IDLE"
                        continue
                    if INITIAL_QR_CODE and normalized_cmd in _QR_QUIT_COMMANDS:
                        logger.info(f"[QR] Voice command received: {cleaned}")
                        await send_generic(cleaned, source="User")
                        await _restart_qr_flow(
                            ws_client, reset_ws=True, reason="voice command: quit app",
                        )
                        await send_generic("App session reset. QR scanner restarted.", source="System")
                        ww_filter.deactivate()
                        await _status.update(voice_assistant="idle")
                        prev_ww_state = "IDLE"
                        continue

                    now = time.monotonic()
                    if (
                        cleaned.lower() == _last_sent_utterance
                        and now - _last_sent_utterance_ts < STT_SPAM_GUARD_WINDOW_S
                    ):
                        logger.debug(f"[STT] Suppressed duplicate utterance: {cleaned}")
                        continue
                    _last_sent_utterance = cleaned.lower()
                    _last_sent_utterance_ts = now
                    logger.info(f"[STT] -> NAT: {cleaned}")
                    await send_generic(cleaned, source="User")
                    if _recorder and _recorder.running:
                        _recorder.log_chat("User", cleaned)
                        _recorder.log_data(f"User command sent: {cleaned}", user_facing=True)
                    await ws_client.send({
                        "type": "user_message",
                        "text": cleaned,
                    })
                    ww_filter.deactivate()
                    await _status.update(voice_assistant="idle")
                    prev_ww_state = "IDLE"

            if FORWARD_AUDIO:
                audio_seq += 1
                b64 = base64.b64encode(chunk).decode("ascii")
                await ws_client.send({
                    "type": "audio_stream",
                    "data": b64,
                    "sample_rate": SAMPLE_RATE,
                    "seq": audio_seq,
                })

            if GEMINI_AUDIO_FORWARD:
                b64 = base64.b64encode(chunk).decode("ascii")
                await ws_client.send({
                    "type": "audio_stream_gemini",
                    "data": b64,
                    "sample_rate": SAMPLE_RATE,
                })

    except asyncio.CancelledError:
        pass
    finally:
        if _recording_task and not _recording_task.done():
            _recording_task.cancel()
        if _recorder and _recorder.running:
            _recorder.stop()
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
