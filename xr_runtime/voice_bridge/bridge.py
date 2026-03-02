"""Voice Bridge -- per-camera service connecting audio hardware to the NAT agent.

Main loop: MediaMTX audio -> FFmpeg decode -> STT -> wake word -> WebSocket to NAT.
Return path: NAT responses -> TTS Pusher / Dashboard.

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
TTS_PUSHER_URL = os.environ.get("TTS_PUSHER_URL", "http://tts-pusher:5000")
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://dashboard:5000")
SOCKET_PATH = os.environ.get("SOCKET_PATH", "/tmp/xr_service.sock")
FORWARD_AUDIO = os.environ.get("FORWARD_AUDIO", "false").lower() in ("true", "1", "yes")
WAKE_WORDS = os.environ.get("WAKE_WORDS", "stella,hey stella").split(",")
WAKE_TIMEOUT = float(os.environ.get("WAKE_TIMEOUT", "10"))
TTS_MODEL = os.environ.get("TTS_MODEL", "vibevoice")

AUDIO_CHUNK_MS = 100
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_SIZE = int(SAMPLE_RATE * (AUDIO_CHUNK_MS / 1000.0) * BYTES_PER_SAMPLE)

_xr_conn = None
_status: Optional[StatusManager] = None


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


async def handle_wake_timeout(msg: dict, ww_filter: WakeWordFilter):
    seconds = msg.get("seconds", 10)
    ww_filter.timeout_seconds = float(seconds)
    logger.info(f"[Bridge] Wake timeout updated to {seconds}s")


async def _trigger_tts(text: str):
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{TTS_PUSHER_URL}/synthesize",
                json={"text": text, "model": TTS_MODEL, "index": CAMERA_INDEX},
            )
    except Exception as exc:
        logger.warning(f"[Bridge] TTS trigger failed: {exc}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main():
    global _status

    logger.info(f"[Bridge] Starting for camera {CAMERA_INDEX}, session {SESSION_ID}")

    _status = StatusManager(DASHBOARD_URL)

    ww_filter = WakeWordFilter(
        wake_words=WAKE_WORDS,
        timeout_seconds=WAKE_TIMEOUT,
    )

    stt = create_stt_client({
        "host": STT_HOST,
        "port": STT_PORT,
        "protocol": STT_PROTOCOL,
    })

    ws_client = NATWebSocketClient(
        url=NAT_SERVER_URL,
        session_id=SESSION_ID,
        camera_index=CAMERA_INDEX,
        rtsp_base=f"rtsp://{MEDIAMTX_HOST}:8554",
        on_connect=lambda: asyncio.ensure_future(_status.update(server_connection="active")),
        on_disconnect=lambda: asyncio.ensure_future(_status.update(server_connection="inactive")),
    )

    ws_client.on("agent_response", handle_agent_response)
    ws_client.on("notification", handle_notification)
    ws_client.on("display_update", handle_display_update)
    ws_client.on("tts_only", handle_tts_only)
    ws_client.on("request_frames", lambda msg: handle_request_frames(msg, ws_client))
    ws_client.on("wake_timeout", lambda msg: handle_wake_timeout(msg, ww_filter))

    ws_task = asyncio.create_task(ws_client.run())

    await _status.update(voice_assistant="idle", server_connection="inactive")

    await stt.start_stream()

    audio_seq = 0
    ffmpeg_proc = None
    _retry_delay = 2.0
    prev_ww_state = "IDLE"

    try:
        while True:
            if ffmpeg_proc is None or ffmpeg_proc.poll() is not None:
                if ffmpeg_proc is not None:
                    ffmpeg_proc.stdout.close()
                    ffmpeg_proc.stderr.close()
                logger.info(f"[Bridge] (Re)starting audio decoder in {_retry_delay:.0f}s")
                await asyncio.sleep(_retry_delay)
                _retry_delay = min(_retry_delay * 1.5, 30.0)
                ffmpeg_proc = start_audio_decoder()

            chunk = await asyncio.get_event_loop().run_in_executor(
                None, ffmpeg_proc.stdout.read, CHUNK_SIZE
            )

            if not chunk:
                ffmpeg_proc = None
                continue

            _retry_delay = 2.0

            await stt.send_audio(chunk)

            transcription = await stt.get_transcription()
            if transcription:
                cleaned = ww_filter.process(transcription)

                # Track wake word state changes -> COMPONENTS_STATUS
                cur_state = ww_filter.state
                if cur_state != prev_ww_state:
                    va = "listening" if cur_state == "ACTIVE" else "idle"
                    await _status.update(voice_assistant=va)
                    prev_ww_state = cur_state

                if cleaned:
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
        if ffmpeg_proc and ffmpeg_proc.poll() is None:
            ffmpeg_proc.terminate()
        logger.info("[Bridge] Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
