# Dashboard

Flask-based web UI and REST API for XR glasses control, frame capture, and NAT agent interaction. Extended from `ai_stream_pipeline/servers/api_message/`.

Mounts the XR socket volume to communicate directly with the glasses via `XRServiceConnection`.

---

## Endpoints

### XR Glasses Control (from ai_stream_pipeline, unchanged)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/` | -- | Web UI (HTML) |
| POST | `/api/send_message` | `{"message_type": "...", "payload": "..."}` | `{"success": true}` |
| POST | `/api/send_audio` | Multipart: `audio` file + `sample_rate` + `method` | `{"success": true, "data": {...}}` |
| GET | `/api/status` | -- | `{"status": "running", "xr_service_connected": true}` |
| GET | `/api/cameras` | -- | `{"cameras": [...], "total": N}` |
| GET | `/api/tts_models` | -- | `{"models": [...]}` (proxied from TTS Pusher) |
| POST | `/api/send_tts` | `{"text": "...", "model": "riva", "camera_id": 1}` | `{"success": true}` |

### Frame Capture (new)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/frames/<camera_id>` | Query: `count` (default 1), `quality` (default 70) | `{"frames": ["base64jpeg...", ...]}` |

Captures frames from the XR socket using `XRServiceConnection.get_latest_frame()`. Used by the dashboard UI for frame viewing and as a fallback for STELLA VLM frame acquisition.

### Agent Proxy (new)

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `/api/agent/chat` | `{"text": "...", "session_id": "demo-1"}` | `{"response": "agent reply text"}` |
| GET | `/api/agent/tools` | -- | Proxied from NAT `GET /tools/catalog` |
| PUT | `/api/agent/tools` | `{"tool_name": true/false, ...}` | Proxied from NAT `PUT /tools/catalog` |
| GET | `/api/agent/status` | -- | `{"connected": true, "nat_url": "..."}` |

The `/api/agent/chat` endpoint opens a temporary WebSocket connection to the NAT server, sends the user message, waits for the `agent_response`, and returns it as HTTP JSON. This enables the web UI to chat with the agent without maintaining its own persistent WebSocket.

---

## XR Socket Integration

The dashboard container mounts the same `xr_socket_{i}` volume as the gRPC server. It uses the `xr_service_library` Python package to:

- **Send messages**: `send_message(Message(message_type, payload))` -- used by `/api/send_message` and when forwarding `display_update` from NAT
- **Send audio**: `schedule_audio_transmission(AudioSample(...))` -- used by `/api/send_audio`
- **Capture frames**: `get_latest_frame()` -- used by `/api/frames`

---

## Web UI Tabs

The web interface (`templates/index.html`) has four tabs:

| Tab | Description |
|-----|-------------|
| **Message** | Send custom messages to XR glasses (message type + payload) |
| **Audio** | Upload WAV/PCM files, choose transmission method (streaming, blob, single chunk) |
| **TTS** | Text-to-speech with model/voice selection, camera targeting |
| **Agent** | Chat with NAT agent, view/toggle tools, connection status |

The Agent tab is new. It provides:
- A chat interface that POSTs to `/api/agent/chat`
- A tool catalog with toggle switches (GET/PUT `/api/agent/tools`)
- NAT server connection status indicator

---

## Dockerfile

Based on `python:3.11.14-slim-bookworm`. Installs:
- `flask`, `flask-cors`, `requests`, `loguru` (from ai_stream_pipeline)
- `websockets` (new, for agent proxy)
- `xr_service_library` wheel

Uses default Debian and PyPI repos (US-based CDN).
