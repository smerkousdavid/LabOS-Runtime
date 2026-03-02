# Voice Bridge

The glue between the XR glasses audio hardware and the agent server. Replaces the old monolithic Pipecat-based `runtime_connector` with a lightweight, focused service (~200 lines per file).

One instance runs per camera. It pulls audio from MediaMTX, transcribes via STT, filters by wake word, and routes text to the NAT server over WebSocket. On the return path, it dispatches TTS synthesis and XR display updates.

---

## Main Loop

```mermaid
graph TD
    subgraph input [Audio Input]
        MTX["MediaMTX RTSP<br/>NB_XXXX_TX_MIC_p6S"]
        FFMPEG["FFmpeg subprocess<br/>decode to s16le 16kHz mono"]
    end

    subgraph processing [Processing]
        STT["STT Client<br/>(gRPC or HTTP)"]
        WW["Wake Word Filter<br/>(IDLE / ACTIVE)"]
    end

    subgraph output [Output to NAT]
        WSC["WebSocket Client<br/>ws://nat:8002/ws"]
    end

    MTX -->|"RTSP pull"| FFMPEG
    FFMPEG -->|"PCM chunks<br/>100ms intervals"| STT
    STT -->|"transcription text"| WW
    WW -->|"filtered text<br/>(wake word stripped)"| WSC
```

## Return Path

```mermaid
graph TD
    subgraph natWS [From NAT via WebSocket]
        AR["agent_response<br/>(tts: true)"]
        DU["display_update"]
        RF["request_frames"]
        WT["wake_timeout"]
    end

    subgraph dispatch [Voice Bridge Dispatch]
        TTS["POST tts-pusher:5000/synthesize"]
        DASH["POST dashboard:5000/api/send_message"]
        FRAMES["Capture frames from XR socket"]
        TIMER["Update wake word timeout"]
    end

    AR --> TTS
    DU --> DASH
    RF --> FRAMES
    WT --> TIMER
    FRAMES -->|"frame_response via WS"| natWS
```

---

## Components

### `bridge.py` -- Main service

Per-camera entry point. Orchestrates:

1. **FFmpeg audio decoder**: Opens RTSP stream from MediaMTX, decodes to raw PCM (16kHz, mono, s16le), pipes to stdout
2. **STT streaming**: Chunks PCM at 100ms intervals, sends to configured STT client
3. **Wake word gate**: Passes transcriptions through the wake word filter
4. **WebSocket send**: Sends `user_message` to NAT server
5. **WebSocket receive loop**: Dispatches incoming messages to appropriate handlers
6. **Frame capture** (Mode 1): On `request_frames`, captures frames from XR socket via `XRServiceConnection.get_latest_frame()`, JPEG-encodes, sends back as `frame_response`
7. **Audio forwarding** (optional): When `forward_audio: true`, sends raw PCM chunks to NAT as `audio_stream` messages

### `stt_client.py` -- Pluggable STT

Two implementations behind a common interface:

```python
class STTClient(ABC):
    async def start_stream(self) -> None: ...
    async def send_audio(self, pcm_chunk: bytes) -> None: ...
    async def get_transcription(self) -> Optional[str]: ...
    async def stop_stream(self) -> None: ...
```

| Implementation | Config `speech.stt.protocol` | How it works |
|---------------|------------------------------|-------------|
| `GrpcSTTClient` | `grpc` | Streaming gRPC to Riva/NIM Parakeet. Opens a bidirectional stream, sends audio chunks, receives partial and final transcriptions. |
| `HttpSTTClient` | `http` | POSTs audio chunks to an HTTP endpoint. Simpler but higher latency. |

Factory: `create_stt_client(config)` reads `speech.stt.protocol` from config.

### `wakeword.py` -- Wake Word Filter

Simple state machine extracted and simplified from the old `runtime_connector/filters/wakeword.py`. No Pipecat dependency.

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> ACTIVE: wake word detected
    ACTIVE --> IDLE: timeout expires
    ACTIVE --> IDLE: sleep command
    ACTIVE --> ACTIVE: user speaks (reset timer)
    ACTIVE --> ACTIVE: NAT sends wake_timeout
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| Wake words | `["stella", "hey stella"]` | Trigger phrases (case-insensitive) |
| Timeout | 10 seconds | Auto-deactivate after silence |
| Sleep commands | `["thanks", "goodbye", "go to sleep"]` | Explicit deactivation phrases |

The filter is a pure function:
```python
def process(self, transcription: str) -> Optional[str]:
    """Returns the cleaned text (wake word stripped) if active, None if filtered out."""
```

### `ws_client.py` -- WebSocket Client

Resilient WebSocket client to the NAT server:

- Connects to `ws://nat-server:8002/ws?session_id=demo-{camera_index}`
- Auto-reconnect with exponential backoff (1s -> 2s -> 4s -> ... -> 30s max)
- Sends `stream_info` on connect (camera RTSP paths)
- Dispatches received messages to registered callbacks:
  - `agent_response` / `notification` -> TTS callback
  - `display_update` -> Display callback
  - `request_frames` -> Frame capture callback
  - `wake_timeout` -> Wake word timer callback

---

## Frame Capture (Mode 1)

When the NAT server needs video frames (for STELLA VLM monitoring), it sends `request_frames` over the WebSocket. The voice bridge handles this because it already mounts the `xr_socket` volume:

1. Receive `{"type": "request_frames", "request_id": "abc", "count": 8, "interval_ms": 1250}`
2. For each of `count` frames:
   - Call `XRServiceConnection.get_latest_frame()`
   - JPEG-encode at configured quality
   - Base64-encode
   - Sleep `interval_ms` between captures
3. Send back `{"type": "frame_response", "request_id": "abc", "frames": [...]}`

This only happens in video Mode 1 (WebSocket). In Modes 2/3, the NAT server reads RTSP directly and never sends `request_frames`.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NAT_SERVER_URL` | yes | WebSocket URL: `ws://nat-server:8002/ws` |
| `CAMERA_INDEX` | yes | 1-based camera index |
| `SESSION_ID` | yes | Session ID for WebSocket (e.g., `demo-1`) |
| `MEDIAMTX_HOST` | yes | MediaMTX hostname (e.g., `mediamtx`) |
| `STT_HOST` | yes | STT service hostname |
| `STT_PORT` | yes | STT service port |
| `STT_PROTOCOL` | no | `grpc` (default) or `http` |
| `TTS_PUSHER_URL` | yes | TTS pusher base URL (e.g., `http://tts-pusher:5000`) |
| `DASHBOARD_URL` | yes | Dashboard base URL (e.g., `http://dashboard:5000`) |
| `SOCKET_PATH` | no | XR socket path (default: `/tmp/xr_service.sock`) |
| `FORWARD_AUDIO` | no | `true` to forward raw audio to NAT (default: `false`) |
| `WAKE_WORDS` | no | Comma-separated wake words (default: `stella,hey stella`) |
| `WAKE_TIMEOUT` | no | Wake word timeout seconds (default: `10`) |
| `LOGURU_LEVEL` | no | Log level (default: `INFO`) |

---

## Dockerfile

Based on `python:3.11.14-slim-bookworm` with FFmpeg. Installs `websockets`, `grpcio`, `loguru`, `httpx`, `opencv-python-headless`, `numpy`, and the `xr_service_library` wheel.
