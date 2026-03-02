"""Shared WebSocket message types for the labos-runtime protocol.

Imported by both nat_server/ws_handler.py and xr_runtime/voice_bridge/ws_client.py.
All messages are JSON objects with a required "type" field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


# ---- XR Runtime -> NAT Server ------------------------------------------------

class UserMessage(TypedDict):
    type: Literal["user_message"]
    text: str


class FrameResponse(TypedDict):
    type: Literal["frame_response"]
    request_id: str
    frames: List[str]  # base64-encoded JPEG strings


class AudioStream(TypedDict):
    type: Literal["audio_stream"]
    data: str           # base64-encoded PCM
    sample_rate: int
    seq: int


class VideoStream(TypedDict):
    type: Literal["video_stream"]
    data: str           # base64-encoded JPEG
    width: int
    height: int
    seq: int


class StreamInfo(TypedDict):
    type: Literal["stream_info"]
    camera_index: int
    rtsp_base: str
    paths: Dict[str, str]  # {"video": "NB_0001_TX_CAM_RGB", "audio": "...", "merged": "..."}


class ProtocolPush(TypedDict):
    type: Literal["protocol_push"]
    protocols: List[Dict[str, str]]  # [{"name": "...", "content": "..."}]


class Ping(TypedDict):
    type: Literal["ping"]


# ---- NAT Server -> XR Runtime ------------------------------------------------

class AgentResponse(TypedDict):
    type: Literal["agent_response"]
    text: str
    tts: bool


class Notification(TypedDict):
    type: Literal["notification"]
    text: str
    tts: bool


class DisplayUpdate(TypedDict):
    type: Literal["display_update"]
    message_type: str
    payload: str


class RequestFrames(TypedDict):
    type: Literal["request_frames"]
    request_id: str
    count: int
    interval_ms: int


class TtsOnly(TypedDict):
    type: Literal["tts_only"]
    text: str
    priority: str  # "normal" | "high"


class WakeTimeout(TypedDict):
    type: Literal["wake_timeout"]
    seconds: int


class Pong(TypedDict):
    type: Literal["pong"]


class ToolCall(TypedDict):
    type: Literal["tool_call"]
    tool_name: str
    summary: str
    status: str  # "started" | "completed" | "failed"


# ---- Type unions for dispatch ------------------------------------------------

InboundMessage = UserMessage | FrameResponse | AudioStream | VideoStream | StreamInfo | ProtocolPush | Ping
OutboundMessage = AgentResponse | Notification | DisplayUpdate | RequestFrames | TtsOnly | WakeTimeout | Pong | ToolCall

# All valid type strings
INBOUND_TYPES = {"user_message", "frame_response", "audio_stream", "video_stream", "stream_info", "protocol_push", "ping"}
OUTBOUND_TYPES = {"agent_response", "notification", "display_update", "request_frames", "tts_only", "wake_timeout", "pong", "tool_call"}


# ---- Helpers -----------------------------------------------------------------

def make_user_message(text: str) -> UserMessage:
    return {"type": "user_message", "text": text}


def make_frame_response(request_id: str, frames: List[str]) -> FrameResponse:
    return {"type": "frame_response", "request_id": request_id, "frames": frames}


def make_audio_stream(data: str, sample_rate: int, seq: int) -> AudioStream:
    return {"type": "audio_stream", "data": data, "sample_rate": sample_rate, "seq": seq}


def make_video_stream(data: str, width: int, height: int, seq: int) -> VideoStream:
    return {"type": "video_stream", "data": data, "width": width, "height": height, "seq": seq}


def make_stream_info(camera_index: int, rtsp_base: str, paths: Dict[str, str]) -> StreamInfo:
    return {"type": "stream_info", "camera_index": camera_index, "rtsp_base": rtsp_base, "paths": paths}


def make_agent_response(text: str, tts: bool = True) -> AgentResponse:
    return {"type": "agent_response", "text": text, "tts": tts}


def make_notification(text: str, tts: bool = True) -> Notification:
    return {"type": "notification", "text": text, "tts": tts}


def make_display_update(message_type: str, payload: str) -> DisplayUpdate:
    return {"type": "display_update", "message_type": message_type, "payload": payload}


def make_request_frames(request_id: str, count: int = 8, interval_ms: int = 1250) -> RequestFrames:
    return {"type": "request_frames", "request_id": request_id, "count": count, "interval_ms": interval_ms}


def make_tts_only(text: str, priority: str = "normal") -> TtsOnly:
    return {"type": "tts_only", "text": text, "priority": priority}


def make_wake_timeout(seconds: int) -> WakeTimeout:
    return {"type": "wake_timeout", "seconds": seconds}


def make_tool_call(tool_name: str, summary: str, status: str) -> ToolCall:
    return {"type": "tool_call", "tool_name": tool_name, "summary": summary, "status": status}


def make_protocol_push(protocols: List[Dict[str, str]]) -> ProtocolPush:
    return {"type": "protocol_push", "protocols": protocols}
