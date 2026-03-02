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

---

## Getting Started

Follow the steps below to install and run LabOS on your machine. Platform-specific commands are provided for **Windows**, **macOS**, and **Linux**.

### Step 1: Prerequisites

Install the following before proceeding.

#### Docker Desktop

Download and install from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/).

| Platform | Notes |
|---|---|
| **Windows** | Install Docker Desktop for Windows. Enable the **WSL 2 backend** during setup. |
| **macOS** | Install Docker Desktop for Mac. Choose the correct chip (Apple Silicon or Intel). |
| **Linux** | Install Docker Engine and the Compose plugin via your package manager: |

```bash
# Linux (Ubuntu/Debian)
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER   # then log out and back in

# Linux (Fedora/RHEL)
sudo dnf install -y docker-ce docker-compose-plugin
sudo systemctl enable --now docker
```

#### Git

| Platform | Command |
|---|---|
| **Windows** | `winget install Git.Git` |
| **macOS** | `brew install git` |
| **Linux** | `sudo apt install git` or `sudo dnf install git` |

#### Python 3.10+

Verify with `python3 --version`. Install from [python.org](https://www.python.org/) if needed. On Windows, use `python --version`.

#### Tailscale (VPN for remote server access)

Tailscale connects your laptop to the LabOS GPU server over a secure mesh network.

| Platform | Install |
|---|---|
| **Windows** | `winget install Tailscale.Tailscale` |
| **macOS** | `brew install tailscale` |
| **Linux** | `curl -fsSL https://tailscale.com/install.sh \| sh` |

See [tailscale.com](https://tailscale.com/) for details.

#### VITURE LabCapture App

The VITURE Pro XR glasses run a custom Unity app for camera streaming and display.

1. Update the VITURE Pro neckband firmware to **2.0.8.30211** or later (OTA update via the VITURE companion app).
2. Install the LabCapture APK from the `viture/` folder in this repository (`LabCapture0214v2.apk`). Side-load it via USB or your preferred method.
3. Launch the app **once** and grant camera + microphone permissions, then close it. This creates the required data directory on the neckband.

See [`viture/README.txt`](viture/README.txt) for full details on the VITURE setup and messaging protocol.

---

### Step 2: Verify Base Components

Before continuing, confirm everything is installed:

```bash
docker info                 # Docker daemon is running
docker compose version      # Compose v2 is available
git --version               # Git is installed
python3 --version           # Python 3.10+
tailscale --version         # Tailscale CLI is available
```

If any command fails, revisit Step 1 for that component.

---

### Step 3: Set Up Tailscale

Tailscale provides secure access to the LabOS GPU server (which runs labos-nat and labos-models).

1. **Contact David Smerkous** ([smerkd@stanford.edu](mailto:smerkd@stanford.edu)) to:
   - Be added to the LabOS tailnet (shared network)
   - Receive the `.env.secrets` file with API keys

2. **Log in** to Tailscale with the account David provides:

```bash
# Linux / macOS
sudo tailscale login

# Windows (run as Administrator)
tailscale login
```

3. **Verify connectivity** -- you should see the LabOS GPU server in the list:

```bash
tailscale status
```

The server will appear with a `100.x.x.x` IP address. This IP is used automatically by the runtime to reach the NAT agent server.

---

### Step 4: Configure the VITURE Glasses

The glasses need to know where to find the runtime on your local network.

1. **Set up a local router** -- both your laptop and the VITURE glasses must be on the same WiFi network.

2. **Turn on the VITURE glasses** and open the LabCapture app at least once, then close it. This ensures the app data directory exists.

3. **Plug the VITURE glasses into your computer** via USB. The neckband storage should appear as a removable drive.

4. **Navigate** to the following path on the glasses storage:

```
/storage/emulated/0/Android/data/com.viture.xr.labcapture/files/Config/
```

If the `Config/` folder does not exist, create it.

5. **Create or edit `sop_config.json`** in that folder with your laptop's local IP address:

```json
{
    "XR_AI_Runtime_IP": "192.168.1.100",
    "XR_AI_Runtime_PORT": "5050"
}
```

Replace `192.168.1.100` with your laptop's actual IP on the local WiFi network. You can find it with `ipconfig` (Windows), `ifconfig` (macOS), or `ip addr` (Linux).

6. **Safely eject** the glasses from your computer, then unplug the USB cable.

---

### Step 5: Install and Run the Runtime

```bash
# Clone the repository
git clone https://github.com/smerkousdavid/LabOS-Runtime
cd LabOS-Runtime

# One-time install
./install.sh          # Linux / macOS
install.bat           # Windows

# Copy the .env.secrets file provided by David
cp /path/to/env.secrets config/.env.secrets

# Start the runtime
./run.sh              # Linux / macOS
run.bat               # Windows
```

Wait for all Docker containers to build and start. This may take several minutes on the first run. For now it might ask
```txt
Update glasses connection config? :
```
Please reply with `n` as auto updating glasses will not work currently.


Once running, **open the LabCapture app** on the VITURE glasses. You should see the status indicators turn green and the assistant will greet you. If the status indicator is not green, still try talking by saying "Hey Stella, how are you?" to see if it connects and runs.

To stop the runtime:

```bash
./stop.sh             # Linux / macOS
stop.bat              # Windows
```

---

### Step 6: Using LabOS

LabOS is voice-controlled through the VITURE glasses. Say **"Hey Stella"** to activate the assistant, then speak your command. The assistant deactivates after 10 seconds of silence or when you say **"thanks"** or **"go to sleep"**.

#### Voice Commands

| Command | What it does |
|---|---|
| **"Hey Stella"** | Wake word -- activates the assistant |
| **"List protocols"** | Shows available lab protocols on the AR display |
| **"Start the PCR protocol"** | Begins a protocol by name (fuzzy matching supported) |
| **"Next step"** | Advance to the next protocol step |
| **"Go to step 3"** | Jump to a specific step number |
| **"Previous step"** | Go back one step |
| **"What am I looking at?"** | STELLA VLM analyzes the camera feed and describes what it sees |
| **"What do you see on the bench?"** | Any visual question triggers the VLM camera analysis |
| **"Provide more details"** | Shows an expanded view of the current step with images |
| **"What errors were made?"** | Reports errors from the current or last protocol run |
| **"Stop protocol"** | Ends the current protocol |
| **"Search for gel electrophoresis"** | Web search -- results displayed on AR |
| **"Thanks"** / **"Go to sleep"** | Deactivates the assistant |

#### Web Dashboard

A monitoring dashboard is available at **http://localhost:5001** while the runtime is running. It shows connection status, live transcription, and agent responses.

---

### Step 7: Writing Custom Protocols

You can add your own lab protocols by placing text files in the `protocols/` folder. Protocols placed here are automatically pushed to the NAT agent server when the runtime connects.

Supported file formats: `.txt`, `.md`, `.csv`, `.json`, `.yaml`

#### Simple Format (numbered list)

For quick or simple protocols, use a plain numbered list:

```
1. Pick up the phone
2. Move to the desk next to laptop
3. Place phone on the desk
```

#### Structured XML Format (recommended)

For complex protocols with materials, context, and error tracking, use XML:

```xml
<protocol>
  <title>Protocol Name</title>
  <goal>One-sentence description of what this protocol accomplishes.</goal>

  <materials>
    <item>Micropipettes (P20, P200) and tips</item>
    <item>5X QS buffer</item>
    <item>PCR tubes (0.2 mL)</item>
  </materials>

  <reaction_mix_per_reaction>
    <reagent name="5X QS buffer">5 uL</reagent>
    <reagent name="Forward Primer">0.5 uL</reagent>
    <total>25 uL</total>
  </reaction_mix_per_reaction>

  <protocol_steps>
    <step id="1" title="Label tubes">
      Label all tubes before adding any liquid.
      Include specific identifiers for each tube.
    </step>
    <step id="2" title="Weigh empty tubes">
      Weigh and record each tube. Wait for the balance
      reading to stabilize before writing it down.
    </step>
  </protocol_steps>

  <common_errors>
    <error>Not changing tips between samples -- causes cross-contamination.</error>
  </common_errors>

  <notes>
    <note>Check reagent volumes against original source before execution.</note>
  </notes>
</protocol>
```

See [`protocols/pilot_pcr.txt`](protocols/pilot_pcr.txt) for a complete working example.

#### How the AI uses each section

| Section | Purpose |
|---|---|
| `<title>`, `<goal>` | Provides context so the agent can answer questions about the protocol's purpose |
| `<materials>` | Helps the agent verify you have the right equipment and reagents |
| `<reaction_mix_per_reaction>` | Referenced when answering questions about volumes and concentrations |
| `<protocol_steps>` | Each `<step>` becomes a tracked step on the AR display with monitoring |
| `<common_errors>` | Surfaced during STELLA VLM monitoring to catch mistakes in real-time |
| `<notes>` | Additional context the agent can reference when answering questions |

The AI agent automatically compacts verbose step descriptions into short, action-oriented AR display text and generates enriched guidance in the background. You do not need to worry about formatting for the display -- write steps as clearly and completely as you want.

---

### Step 8: Recordings

Camera and audio streams are automatically recorded when the glasses connect (if enabled). Recordings are useful for reviewing protocol execution, training, and auditing.

Enable recording in `config/config.yaml`:

```yaml
recording:
  enabled: true           # set to false to disable
  format: fmp4            # fmp4 | mpegts
  segment_duration: 60m   # segment length before rolling to a new file
  path: "./recordings"    # host-side storage path
```

Recordings are stored at:

```
recordings/
├── NB_0001_TX_CAM_RGB/             # video-only
│   └── 2026-03-02_14-30-00.mp4
├── NB_0001_TX_MIC_p6S/             # audio-only
│   └── 2026-03-02_14-30-00.mp4
└── NB_0001_TX_CAM_RGB_MIC_p6S/     # merged audio + video
    └── 2026-03-02_14-30-00.mp4
```

Each camera produces separate video-only, audio-only, and merged recordings. Files are stored locally on the runtime machine and are gitignored.

---

## Technical Reference

Everything below is reference material for developers and advanced configuration.

### Configuration

#### `config/config.yaml`

| Section | Purpose |
|---|---|
| `nat_server.url` | WebSocket URL of the labos-nat agent server |
| `speech.stt` | STT endpoint (host, port, protocol) on labos-models |
| `speech.tts` | TTS endpoint and provider on labos-models |
| `runtime` | Camera count, streaming method, framerate |
| `dashboard` | Local dashboard port (default 5001) |
| `nvr` | Optional Shinobi NVR for video recording |
| `wakeword` | Wake words, timeout, sleep commands |

No API keys are needed on the desktop app unless using cloud STT/TTS (e.g. ElevenLabs).

#### `config/.env.secrets`

API keys for cloud services (ElevenLabs, DashScope, NGC). Copy `config/.env.secrets.example` and fill in values. Keys are injected into `.env` at configure time.

### RTSP Streams

Each camera produces three RTSP streams on MediaMTX (port 8554):

| Stream | Path pattern | Content |
|---|---|---|
| Video only | `NB_XXXX_TX_CAM_RGB` | H.264 video from glasses camera |
| Audio only | `NB_XXXX_TX_MIC_p6S` | PCM audio from glasses microphone |
| Merged | `NB_XXXX_TX_CAM_RGB_MIC_p6S` | Combined audio + video |

Where `XXXX` is the zero-padded camera index (e.g. `0001`).

The RTSP host is auto-detected at configure time. When `rtsp.external_host` is `"auto"` (the default):
1. If the NAT server is on a Tailscale network (100.x.x.x), the local Tailscale IPv4 is used
2. Otherwise, the local IP that routes to the NAT server is detected via UDP
3. Falls back to `localhost` if neither works

The resolved IP is written to `.env` as `RTSP_EXTERNAL_HOST` and sent to the NAT server on connection via `stream_info`.

### Message Types

The runtime sends three message types to the XR glasses:

| Type | Panel | Format |
|---|---|---|
| `GENERIC` | LLM Chat | `{"message": {"type": "rich-text", "content": "...", "source": "Agent/User"}}` |
| `SINGLE_STEP_PANEL_CONTENT` | Step Panel | `{"messages": [{"type": "rich-text/base64-image", "content": "..."}]}` |
| `COMPONENTS_STATUS` | Status Bar | `{"Voice_Assistant": "idle/listening", "Server_Connection": "active/inactive", "Robot_Status": "..."}` |

### Services

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

### NAT WebSocket Protocol

The runtime connects to the NAT agent server over WebSocket. All messages are JSON with a `type` field. The canonical type definitions live in `xr_runtime/voice_bridge/ws_protocol.py`.

**Connection**: `ws://<nat-host>:8002/ws?session_id=<id>`

#### Runtime -> NAT

| Type | When | Key Fields |
|---|---|---|
| `stream_info` | On connect | `camera_index`, `rtsp_base` (e.g. `rtsp://100.93.x.x:8554`), `paths` (`{video, audio, merged}`) |
| `user_message` | User speaks (wake word stripped) | `text` |
| `frame_response` | Reply to `request_frames` | `request_id`, `frames` (base64 JPEG list) |
| `audio_stream` | Optional raw audio forwarding | `data` (base64 PCM), `sample_rate`, `seq` |
| `video_stream` | Optional WS frame push | `data` (base64 JPEG), `width`, `height`, `seq` |
| `protocol_push` | On connect (after `stream_info`) | `protocols` (list of `{name, content}` from local `protocols/` folder) |
| `ping` | Keepalive | -- |

#### NAT -> Runtime

| Type | Purpose | Key Fields |
|---|---|---|
| `agent_response` | Agent reply, optional TTS | `text`, `tts` (bool) |
| `notification` | System notification, optional TTS | `text`, `tts` (bool) |
| `display_update` | Push to glasses display | `message_type` (`GENERIC` / `SINGLE_STEP_PANEL_CONTENT` / `COMPONENTS_STATUS`), `payload` (JSON string) |
| `request_frames` | Capture camera frames | `request_id`, `count`, `interval_ms` |
| `tts_only` | Speak without display | `text`, `priority` (`normal` / `high`) |
| `tool_call` | Notify runtime of tool activity | `tool_name`, `summary`, `status` (`started` / `completed` / `failed`) |
| `wake_timeout` | Override wake word timeout | `seconds` |
| `pong` | Keepalive reply | -- |

#### Consuming RTSP from the NAT server

1. On WebSocket connect, listen for the `stream_info` message
2. Build RTSP URLs: `{rtsp_base}/{paths.merged}` (e.g. `rtsp://100.93.211.91:8554/NB_0001_TX_CAM_RGB_MIC_p6S`)
3. Open with any RTSP client (OpenCV `cv2.VideoCapture`, `ffmpeg`, GStreamer)
4. Video-only and audio-only streams are available via `paths.video` and `paths.audio`

### Logs

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

### Directory Structure

```
labos-runtime/
├── install.sh / install.bat      # One-time setup
├── run.sh / run.bat              # Start runtime
├── stop.sh / stop.bat            # Stop runtime
├── update_glasses.sh / .bat      # Glasses USB config
├── requirements.txt              # Host Python deps
├── protocols/                    # Custom protocol files (pushed to NAT on connect)
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
├── viture/                       # VITURE glasses APK, config, docs
└── logs/                         # Per-service log files
```
