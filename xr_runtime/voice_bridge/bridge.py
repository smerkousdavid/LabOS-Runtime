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
import secrets
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


def _default_session_id() -> str:
    """Unique per-runtime session id: labos-runtime-<random_hex>-<camera_index>."""
    return f"labos-runtime-{secrets.token_hex(4)}-{CAMERA_INDEX}"


SESSION_ID = os.environ.get("SESSION_ID", _default_session_id())
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
        return False
    try:
        from xr_service_library.xr_types import Message
        msg = Message(message_type=message_type, payload=payload)
        result = conn.send_message(msg)
        if result:
            return True

        logger.warning(f"[Bridge] send_message returned falsy for {message_type}; reconnecting once")
        retry_conn = _get_msg_connection(reconnect=True)
        if retry_conn is None:
            return False

        retry_result = retry_conn.send_message(msg)
        if not retry_result:
            logger.warning(f"[Bridge] send_message retry returned falsy for {message_type}")
            return False
        return True
    except Exception as exc:
        logger.warning(f"[Bridge] Send to glasses failed: {exc}")
        if "context" in str(exc).lower():
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
_rtsp_relay_proc = None
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
    panel_resend_interval = 2.0
    last_panel_payload: Optional[str] = None
    last_panel_send_ts = 0.0
    _prompt_msg = (
        "<size=16><color=#59D2FF>Point at the QR code on screen</color></size>"
        '<size=14><color=#CCCCCC>\nSay "show qr code" or "quit app" to restart session</color></size>'
    )
    _waiting_msg = (
        '<size=16><color=#888888>Waiting for camera...</color></size>'
        '<size=14><color=#CCCCCC>\nSay "show qr code" or "quit app"</color></size>'
    )

    logger.info("[QR] QR code scanning started")

    while _qr_scanning_active:
        try:
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
                    logger.info(f"[QR] Decoded payload: {data[:200]}")
                    try:
                        payload = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        logger.info(f"[QR] Non-JSON QR detected (raw string), sending as-is")
                        payload = {"t": "ll", "raw": data}

                    if isinstance(payload, dict) and payload.get("t") == "ll":
                        session_id = payload.get("session_id", "?")
                        logger.info(f"[QR] Matched LabOS QR: session={session_id}")
                        await ws_client.send({
                            "type": "qr_payload",
                            "payload": payload,
                        })
                        _qr_scanning_active = False
                        break
                    else:
                        logger.info(f"[QR] Ignoring QR -- type={payload.get('t', 'N/A')!r} (expected 'labos_live')")

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
            now = time.monotonic()
            # Avoid flooding gRPC queue with identical panel payloads while in QR loop.
            should_send_panel = (
                panel_payload != last_panel_payload
                or (now - last_panel_send_ts) >= panel_resend_interval
            )
            if should_send_panel:
                sent = await _send_to_glasses("SINGLE_STEP_PANEL_CONTENT", panel_payload)
                if sent:
                    last_panel_payload = panel_payload
                    last_panel_send_ts = now

            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning(f"[QR] Scan error: {exc}")
            await asyncio.sleep(1.0)

    logger.info("[QR] QR code scanning stopped")


def stop_qr_scanning():
    global _qr_scanning_active
    _qr_scanning_active = False


def start_qr_scanning():
    global _qr_scanning_active
    _qr_scanning_active = True


# ---------------------------------------------------------------------------
# RTSP relay to LabOS server
# ---------------------------------------------------------------------------

async def _start_rtsp_relay(local_rtsp_path: str, publish_rtsp: str):
    """Start FFmpeg process to relay local RTSP stream to remote URL."""
    global _rtsp_relay_proc
    await _stop_rtsp_relay()

    local_url = f"rtsp://localhost:8554/{local_rtsp_path}"
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", local_url,
        "-c", "copy",
        "-f", "rtsp",
        publish_rtsp,
    ]
    logger.info(f"[RTSP] Starting relay: {local_url} -> {publish_rtsp}")
    try:
        _rtsp_relay_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info(f"[RTSP] Relay started (pid={_rtsp_relay_proc.pid})")
    except Exception as exc:
        logger.error(f"[RTSP] Failed to start relay: {exc}")
        _rtsp_relay_proc = None


async def _stop_rtsp_relay():
    """Stop the RTSP relay FFmpeg process."""
    global _rtsp_relay_proc
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
        await _start_rtsp_relay(local_path, publish_rtsp)


async def _handle_session_cleared(ws_client):
    """LabOS Live session ended -- stop relay, return to QR scanning."""
    logger.info("[Session] LabOS Live session cleared")
    await _stop_rtsp_relay()

    if INITIAL_QR_CODE:
        import asyncio as _aio
        start_qr_scanning()
        _aio.create_task(_qr_scan_task(ws_client))


async def _restart_qr_flow(ws_client: NATWebSocketClient, *, reset_ws: bool, reason: str) -> None:
    """Stop current live session state and return to QR scanning."""
    logger.info(f"[QR] Restart requested: {reason} reset_ws={reset_ws}")
    stop_qr_scanning()
    await _stop_rtsp_relay()
    if reset_ws:
        await ws_client.reset_session()
    if INITIAL_QR_CODE:
        start_qr_scanning()
        asyncio.create_task(_qr_scan_task(ws_client))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main():
    global _status, _ww_filter, _recorder, _recording_task

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

    ws_client = NATWebSocketClient(
        url=NAT_SERVER_URL,
        session_id=SESSION_ID,
        camera_index=CAMERA_INDEX,
        rtsp_base=f"rtsp://{RTSP_EXTERNAL_HOST}:8554",
        on_connect=_on_ws_connect,
        on_disconnect=lambda: asyncio.ensure_future(_status.update(server_connection="inactive")),
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

    ws_task = asyncio.create_task(ws_client.run())

    if INITIAL_QR_CODE:
        asyncio.create_task(_qr_scan_task(ws_client))

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
    _retry_delay = 0.05
    prev_ww_state = "IDLE"
    _glasses_connected = False
    _disconnect_handled = False
    _disconnect_timer: Optional[asyncio.Task] = None
    _last_sent_utterance = ""
    _last_sent_utterance_ts = 0.0
    _last_good_chunk_ts = 0.0

    async def _delayed_session_reset():
        """Wait DISCONNECT_TIMEOUT_S then reset the session if still disconnected."""
        try:
            await asyncio.sleep(DISCONNECT_TIMEOUT_S)
            if not _glasses_connected:
                logger.info(
                    f"[Session] Glasses still disconnected after {DISCONNECT_TIMEOUT_S}s "
                    "-- resetting session"
                )
                await ws_client.reset_session()
        except asyncio.CancelledError:
            logger.info("[Session] Disconnect timer cancelled (glasses reconnected)")

    try:
        while True:
            if ffmpeg_proc is None or ffmpeg_proc.poll() is not None:
                if ffmpeg_proc is not None:
                    ffmpeg_proc.stdout.close()
                    ffmpeg_proc.stderr.close()

                if (
                    _glasses_connected
                    and not _disconnect_handled
                    and _last_good_chunk_ts > 0
                    and (time.monotonic() - _last_good_chunk_ts) >= AUDIO_SILENCE_DISCONNECT_S
                ):
                    _glasses_connected = False
                    _disconnect_handled = True
                    silence_s = time.monotonic() - _last_good_chunk_ts
                    logger.info(
                        f"[Session] Glasses disconnected -- no audio for "
                        f"{silence_s:.1f}s"
                    )
                    if _recorder and _recorder.running:
                        _recorder.log_data(f"Glasses disconnected (no audio for {silence_s:.1f}s)")

                    if _recording_task and not _recording_task.done():
                        _recording_task.cancel()
                    if _recorder and _recorder.running:
                        _recorder.stop()

                    await stt.stop_stream()
                    accumulator._buffer.clear()
                    ww_filter.reset()
                    _tts_cancel.set()

                    if RESET_SESSION == "true":
                        if _disconnect_timer and not _disconnect_timer.done():
                            _disconnect_timer.cancel()
                        _disconnect_timer = asyncio.create_task(_delayed_session_reset())
                        logger.info(
                            f"[Session] Disconnect timer started ({DISCONNECT_TIMEOUT_S}s)"
                        )

                    await _status.update(
                        voice_assistant="idle",
                        server_connection="inactive",
                        robot_status="N/A",
                    )

                await asyncio.sleep(_retry_delay)
                _retry_delay = min(_retry_delay * 2, 0.5)
                ffmpeg_proc = start_audio_decoder()

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
                ffmpeg_proc.kill()
                ffmpeg_proc.stdout.close()
                ffmpeg_proc.stderr.close()
                ffmpeg_proc = None
                continue

            if not chunk:
                ffmpeg_proc = None
                continue

            _last_good_chunk_ts = time.monotonic()
            _retry_delay = 0.05

            if not _glasses_connected:
                _glasses_connected = True
                _disconnect_handled = False
                if _disconnect_timer and not _disconnect_timer.done():
                    _disconnect_timer.cancel()
                    _disconnect_timer = None
                    logger.info("[Session] Disconnect timer cancelled -- glasses reconnected")
                logger.info("[Bridge] Audio stream active -- restarting STT")
                await stt.start_stream()

                _status.voice_assistant = "idle"
                _status.server_connection = "active"
                await _status.force_push()

                if _last_display_update:
                    mt, pl = _last_display_update
                    logger.info(f"[Bridge] Replaying cached display_update ({mt})")
                    await _send_to_glasses(mt, pl)

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
