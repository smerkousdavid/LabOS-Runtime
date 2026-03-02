# LabOS XR Runtime

Desktop application for VITURE XR glasses-based laboratory assistance. Manages the glasses hardware connection, audio/video streaming, speech-to-text, text-to-speech, and a local web dashboard.

This is the **client-side** component. It connects to two external services:

- **[labos-nat](../labos-nat/)** -- Agent server (LLM tool-calling, protocol management)
- **[labos-models](../labos-models/)** -- GPU model hosting (STT, TTS, LLM, VLM)

```
┌─────────────────────────────────────────────┐
│  labos-runtime (this repo, desktop app)     │
│                                             │
│  VITURE Glasses ──► gRPC Server ──► MediaMTX│
│                      │                      │
│               Voice Bridge                  │
│            (STT, wake word, WS)             │
│                 │         │                 │
│            Dashboard    TTS Pusher/Mixer    │
└─────────────┬───────────┬───────────────────┘
              │           │
     ┌────────▼──┐   ┌────▼────────┐
     │ labos-nat │   │ labos-models│
     │ (agent)   │   │ (STT, TTS) │
     └───────────┘   └─────────────┘
```

## Prerequisites

- **Docker Desktop** (Windows/macOS) or Docker + Docker Compose (Linux)
- **Python 3.10+**
- **Git**

## Quick Start

```bash
# 1. Clone
git clone <repo-url> && cd labos-runtime

# 2. Install (one time)
./install.sh          # Linux/macOS
install.bat           # Windows

# 3. Configure
#    Edit config/config.yaml -- set NAT server URL, STT/TTS endpoints
#    Edit config/.env.secrets -- add API keys (if any needed locally)

# 4. Configure glasses (optional)
./update_glasses.sh   # Linux/macOS
update_glasses.bat    # Windows

# 5. Run
./run.sh              # Linux/macOS
run.bat               # Windows

# 6. Stop
./stop.sh             # Linux/macOS
stop.bat              # Windows
```

## Configuration

### `config/config.yaml`

| Section | Purpose |
|---|---|
| `nat_server.url` | WebSocket URL of the labos-nat agent server |
| `speech.stt` | STT endpoint (host, port, protocol) on labos-models |
| `speech.tts` | TTS endpoint and provider on labos-models |
| `runtime` | Camera count, streaming method, framerate |
| `dashboard` | Local dashboard port (default 5001) |
| `nvr` | Optional Shinobi NVR for video recording |
| `wakeword` | Wake words, timeout, sleep commands |

No API keys are needed on the desktop app. All model inference and web search keys live on the server side (labos-nat, labos-models).

### `config/.env.secrets`

Only needed if running model services locally (not typical for desktop use).

## Glasses Setup

The `update_glasses` utility writes the runtime's IP and port to the glasses so they know where to connect:

```bash
./update_glasses.sh
```

1. Select your network interface (WiFi/Ethernet)
2. Plug in glasses via USB
3. The script writes `sop_config.json` to the glasses
4. Unplug and power-cycle the glasses

## Message Types

The runtime sends three message types to the XR glasses:

| Type | Panel | Format |
|---|---|---|
| `GENERIC` | LLM Chat | `{"message": {"type": "rich-text", "content": "...", "source": "Agent/User"}}` |
| `SINGLE_STEP_PANEL_CONTENT` | Step Panel | `{"messages": [{"type": "rich-text/base64-image", "content": "..."}]}` |
| `COMPONENTS_STATUS` | Status Bar | `{"Voice_Assistant": "idle/listening", "Server_Connection": "active/inactive", "Robot_Status": "..."}` |

## Services

When running, Docker Compose manages these containers:

| Service | Port | Purpose |
|---|---|---|
| gRPC Server | 5050 | Glasses connection (per camera) |
| MediaMTX | 8554 | RTSP relay |
| Video Pusher | -- | gRPC frames to RTSP |
| Audio Pusher | -- | gRPC audio to RTSP |
| AV Merger | -- | Merge video + audio streams |
| Voice Bridge | -- | STT + wake word + WS to NAT |
| Dashboard | 5001 | Web UI |
| TTS Pusher | 5100 | TTS synthesis routing |
| TTS Mixer | 5004 | RTSP audio publisher |

## Logs

All service logs are written to `logs/` with 3-day retention:

```
logs/
├── grpc_1/
├── video_pusher_1/
├── audio_pusher_1/
├── av_merger_1/
├── voice_bridge_1/
├── dashboard/
├── tts_pusher/
├── tts_mixer/
└── mediamtx/
```

## Directory Structure

```
labos-runtime/
├── install.sh / install.bat      # One-time setup
├── run.sh / run.bat              # Start runtime
├── stop.sh / stop.bat            # Stop runtime
├── update_glasses.sh / .bat      # Glasses USB config
├── requirements.txt              # Host Python deps
├── config/
│   ├── config.yaml               # Main configuration
│   ├── config.yaml.example
│   └── .env.secrets
├── scripts/
│   ├── launcher.py               # Main entry point
│   ├── configure.py              # Config file generator
│   ├── update_glasses.py         # Glasses config utility
│   ├── network_utils.py          # Network interface discovery
│   └── glass_connectors/         # Pluggable device connectors
├── compose/
│   ├── runtime.j2                # Docker Compose template
│   └── generate.py               # Compose renderer
├── xr_runtime/
│   ├── grpc_server/              # Pre-built gRPC binary
│   ├── streaming/                # Video/audio pushers, merger
│   ├── voice_bridge/             # STT, wake word, WebSocket
│   ├── dashboard/                # Flask web UI
│   ├── speech/                   # TTS pusher and mixer
│   └── nvr/                      # Shinobi NVR (optional)
└── logs/                         # Per-service log files
```
