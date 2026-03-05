# Compose

Docker Compose generation using Jinja2 templates. Creates a `compose.yaml` with per-camera service stacks and shared services, configurable for N cameras and optional components.

---

## How It Works

```
config/config.yaml
       |
       v
  scripts/configure.py  -->  .env (ports, image names, flags)
       |
       v
  compose/generate.py   -->  compose.yaml
       |
    reads .env + runtime.j2 template
```

1. `scripts/configure.py` reads `config/config.yaml` and generates `.env` with all Docker Compose variables
2. `generate.py` reads `.env` and renders `runtime.j2` with Jinja2, producing the final `compose.yaml`
3. `start.sh` runs both steps, then `docker compose -f compose.yaml up -d`

---

## Template Variables

`generate.py` passes these variables to the Jinja2 template:

| Variable | Source | Description |
|----------|--------|-------------|
| `num_cameras` | CLI arg (`sys.argv[1]`) | Number of camera/glasses pairs |
| `streaming_method` | CLI arg (`sys.argv[2]`) | `mediamtx` or `gstreamer` |
| `default_framerate` | CLI arg (`sys.argv[3]`) | Video framerate (default 30) |
| `base_ports` | hardcoded | `{GRPC_PORT: 5050, WEB_PORT: 5001}` |
| `video_mode` | `.env` | `websocket`, `rtsp_pull`, or `mediamtx_relay` |
| `mediamtx_service_name` | `.env` | MediaMTX container name |
| `rtsp_image` | `.env` | Docker image for streaming containers |
| `ENABLE_TTS` | `.env` | Whether to include TTS services |
| `ENABLE_NVR` | `.env` | Whether to include Shinobi NVR |
| `nat_server_url` | `.env` | NAT server WebSocket URL |
| `nat_mediamtx_port` | `.env` | NAT-side MediaMTX port (Mode 3, default 8654) |
| `stt_host` / `stt_port` | `.env` | STT service host and port |
| `tts_model` | `.env` (`TTS_PROVIDER`) | TTS provider name (default `vibevoice`) |
| `forward_audio` | `.env` | Forward raw audio to NAT over WebSocket |

---

## Generated Services

### Per-camera (N copies)

| Service | Image | Volumes | Ports | Description |
|---------|-------|---------|-------|-------------|
| `grpc-server-{i}` | `python:3.12.6-slim` | `xr_socket_{i}:/tmp` | `5050+i:5050` | gRPC binary for glasses |
| `video-pusher-{i}` | `labos_streaming` | `xr_socket_{i}:/tmp` | -- | FFmpeg video to MediaMTX |
| `audio-pusher-{i}` | `labos_streaming` | `xr_socket_{i}:/tmp` | -- | FFmpeg audio to MediaMTX |
| `av-merger-{i}` | `labos_streaming` | -- | -- | Remux video+audio |
| `voice-bridge-{i}` | `labos_voice_bridge` | `xr_socket_{i}:/tmp` | -- | STT + wake word + WebSocket |

### Shared (one copy)

| Service | Image | Ports | Condition |
|---------|-------|-------|-----------|
| `mediamtx` | `bluenviron/mediamtx:latest-ffmpeg` | 8554, 8888, 8889 | Always |
| `nat-server` | `labos_nat_server` | 8002 | Always |
| `dashboard` | `labos_dashboard` | 5001 | Always |
| `tts-pusher` | `labos_tts_pusher` | 5100 | `ENABLE_TTS` |
| `tts-mixer` | `labos_tts_mixer` | 5004 | `ENABLE_TTS` |
| `nat-mediamtx` | `bluenviron/mediamtx:latest-ffmpeg` | 8654 | `video_mode == mediamtx_relay` |
| `db` | `mysql:8.0` | 9906 | `ENABLE_NVR` |
| `shinobi` | `labos_nvr` | 8088 | `ENABLE_NVR` |

### Volumes

```yaml
volumes:
  xr_socket_1:
  xr_socket_2:
  # ... one per camera
```

---

## Optional Services

### TTS (gated by `ENABLE_TTS`)

When TTS is disabled, the `tts-pusher`, `tts-mixer`, and all `NB_XXXX_RX_TTS` stream paths are omitted. The voice bridge still connects to NAT but TTS responses are text-only (displayed, not spoken).

### NVR (gated by `ENABLE_NVR`)

When NVR is disabled, the `db`, `shinobi` services and all Shinobi registration steps in `start.sh` are skipped. Streams are still available via MediaMTX HLS/WebRTC for browser viewing.

### NAT-side MediaMTX (gated by `video_mode == mediamtx_relay`)

Only created when Mode 3 video routing is configured. Runs a second MediaMTX instance on port 8654 that receives relayed streams from the XR runtime's MediaMTX.

---

## Example Output (2 cameras, TTS enabled, relay mode)

```yaml
services:
  mediamtx:
    image: bluenviron/mediamtx:latest-ffmpeg
    ports: ["8554:8554", "8888:8888", "8889:8889"]
    volumes: ["./xr_runtime/streaming/mediamtx.yml:/mediamtx.yml"]

  grpc-server-1:
    image: python:3.12.6-slim
    ports: ["5050:5050"]
    volumes: [xr_socket_1:/tmp]
    # ...

  grpc-server-2:
    image: python:3.12.6-slim
    ports: ["5051:5050"]
    volumes: [xr_socket_2:/tmp]
    # ...

  video-pusher-1:
    # ... pushes to NB_0001_TX_CAM_RGB
  video-pusher-2:
    # ... pushes to NB_0002_TX_CAM_RGB

  voice-bridge-1:
    environment:
      - SESSION_ID=demo-1
      - CAMERA_INDEX=1
    # ...
  voice-bridge-2:
    environment:
      - SESSION_ID=demo-2
      - CAMERA_INDEX=2
    # ...

  nat-server:
    build: ./nat_server
    ports: ["8002:8002"]
    volumes: ["./protocols:/app/protocols"]

  nat-mediamtx:
    image: bluenviron/mediamtx:latest-ffmpeg
    ports: ["8654:8554"]
    volumes: ["./nat_server/mediamtx.yml:/mediamtx.yml"]

  dashboard:
    build: ./xr_runtime/dashboard
    ports: ["5001:5000"]

  tts-pusher:
    build: ./xr_runtime/speech/tts_pusher
    ports: ["5100:5000"]

  tts-mixer:
    build: ./xr_runtime/speech/tts_mixer
    ports: ["5004:5002"]

volumes:
  xr_socket_1:
  xr_socket_2:
```

---

## Usage

```bash
# Generate compose file (called by start.sh automatically)
python compose/generate.py 2 mediamtx 30

# Or with custom output path
python compose/generate.py 1 mediamtx 30 my-compose.yaml
```
