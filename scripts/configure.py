#!/usr/bin/env python3
"""Generate derived config files from config/config.yaml + config/.env.secrets.

Outputs:
  .env                                          Docker Compose environment variables
  xr_runtime/streaming/mediamtx.yml             MediaMTX config
  xr_runtime/speech/tts_pusher/tts_models.yaml  TTS model definitions
"""

import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config" / "config.yaml"


# ---------------------------------------------------------------------------
# Hostname resolution (so Docker containers always get IPs they can reach)
# ---------------------------------------------------------------------------

def resolve_host(hostname: str) -> str:
    """Resolve a hostname to an IPv4 address.

    Returns the original string unchanged if it's already an IP or resolution
    fails (so the rest of the pipeline keeps working with the raw value).
    """
    if not hostname or hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
        return hostname
    try:
        socket.inet_aton(hostname)
        return hostname
    except OSError:
        pass
    try:
        info = socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM)
        if info:
            ip = info[0][4][0]
            print(f"  Resolved {hostname} -> {ip}")
            return ip
    except socket.gaierror:
        pass
    return hostname


def resolve_url(url: str) -> str:
    """Replace the hostname inside a URL with its resolved IP."""
    parsed = urlparse(url)
    if not parsed.hostname:
        return url
    resolved = resolve_host(parsed.hostname)
    if resolved == parsed.hostname:
        return url
    return parsed._replace(netloc=parsed.netloc.replace(parsed.hostname, resolved, 1)).geturl()


# ---------------------------------------------------------------------------
# Secrets loader
# ---------------------------------------------------------------------------

def load_secrets(secrets_path: Path) -> dict:
    secrets = {}
    if not secrets_path.exists():
        return secrets
    with open(secrets_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                secrets[key.strip()] = value.strip()
    return secrets


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Auto-detect RTSP external host
# ---------------------------------------------------------------------------

def detect_external_host(nat_url: str) -> str:
    """Return the local IP reachable from the NAT server's network.

    Priority: Tailscale (if NAT is on 100.x.x.x) -> UDP route -> localhost.
    """
    parsed = urlparse(nat_url)
    nat_host = parsed.hostname or "localhost"

    # Tailscale: NAT address is in the CGNAT range used by Tailscale
    if nat_host.startswith("100."):
        try:
            result = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                ip = result.stdout.strip().split("\n")[0]
                if ip:
                    return ip
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Generic: UDP socket trick to discover which local IP routes to NAT host
    if nat_host not in ("localhost", "127.0.0.1"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((nat_host, parsed.port or 8002))
            local_ip = s.getsockname()[0]
            s.close()
            if local_ip and local_ip != "0.0.0.0":
                return local_ip
        except Exception:
            pass

    return "localhost"


# ---------------------------------------------------------------------------
# .env (Docker Compose runtime)
# ---------------------------------------------------------------------------

def generate_env(cfg: dict, secrets: dict) -> str:
    rt = cfg.get("runtime", {})
    sp = cfg.get("speech", {})
    dash = cfg.get("dashboard", {})
    nvr = cfg.get("nvr", {})
    nat = cfg.get("nat_server", {})

    streaming = rt.get("streaming", {})
    stt = sp.get("stt", {})
    stt_fallback = stt.get("fallback", {})
    stt_noise = stt.get("noise_correction", {})
    stt_endpointing = stt.get("endpointing", {})
    suppression_terms = stt_noise.get("suppression_terms", [])
    if isinstance(suppression_terms, (list, tuple)):
        suppression_terms_env = ",".join(str(x) for x in suppression_terms)
    else:
        suppression_terms_env = str(suppression_terms or "")

    tts = sp.get("tts", {})
    nvr_creds = nvr.get("credentials", {})
    shinobi = nvr.get("shinobi", {})

    session = cfg.get("session", {})
    nat_url = resolve_url(nat.get("url", "ws://localhost:8002/ws"))

    # Resolve STT host so Docker containers can always reach it
    stt_host = resolve_host(stt.get("host", "localhost"))

    # Resolve RTSP external host
    _rtsp_host_raw = cfg.get("rtsp", {}).get("external_host", "auto")
    if _rtsp_host_raw == "auto":
        _rtsp_host = detect_external_host(nat_url)
        print(f"  RTSP external host: {_rtsp_host} (auto-detected)")
    else:
        _rtsp_host = _rtsp_host_raw
        print(f"  RTSP external host: {_rtsp_host} (manual)")

    # Forward every key from .env.secrets into .env so Docker Compose
    # containers can pick them up.  Specific known keys get a guaranteed
    # line even if the secrets file omits them.
    _secret_keys = ["NGC_API_KEY", "DASHSCOPE_API_KEY", "ELEVENLABS_API_KEY"]
    secret_lines = []
    for k in _secret_keys:
        secret_lines.append(f"{k}={secrets.get(k, '')}")
    for k, v in secrets.items():
        if k not in _secret_keys:
            secret_lines.append(f"{k}={v}")

    lines = [
        "# Auto-generated by scripts/configure.py -- do not edit manually.",
        "# Secrets are read from config/.env.secrets at generation time.",
        "",
        "# Secrets (from config/.env.secrets)",
        *secret_lines,
        "",
        "# Runtime",
        f"NUM_CAMERAS={rt.get('num_cameras', 1)}",
        f"STREAMING_METHOD={streaming.get('method', 'mediamtx')}",
        f"DEFAULT_FRAMERATE={streaming.get('framerate', 30)}",
        f"MEDIAMTX_SERVICE_NAME={streaming.get('mediamtx_service_name', 'mediamtx')}",
        "",
        "# Ports",
        f"GRPC_PORT=5050",
        f"WEB_PORT={dash.get('port', 5001)}",
        f"TTS_PUSHER_PORT=5100",
        f"TTS_MIXER_PORT=5004",
        f"NVR_PORT={nvr.get('port', 8088)}",
        "",
        "# NAT Server (external)",
        f"NAT_SERVER_URL={nat_url}",
        "",
        "# Speech",
        f"STT_HOST={stt_host}",
        f"STT_PORT={stt.get('port', 50051)}",
        f"STT_PROTOCOL={stt.get('protocol', 'grpc')}",
        f"STT_MODEL={stt.get('model', '')}",
        f"STT_LANGUAGE={stt.get('language', 'en')}",
        f"STT_COMMIT_INTERVAL_S={stt.get('commit_interval_s', 0.25)}",
        f"STT_MIN_SPEECH_DURATION_MS={stt.get('min_speech_duration_ms', 500)}",
        f"STT_MIN_SILENCE_DURATION_MS={stt.get('min_silence_duration_ms', 500)}",
        f"STT_INCLUDE_TIMESTAMPS={'true' if stt.get('include_timestamps', True) else 'false'}",
        f"STT_FALLBACK_RECOVER_AFTER_S={stt.get('fallback_recover_after_s', 30)}",
        f"STT_FALLBACK_PROTOCOL={stt_fallback.get('protocol', '')}",
        f"STT_FALLBACK_HOST={resolve_host(stt_fallback.get('host', stt_host))}",
        f"STT_FALLBACK_PORT={stt_fallback.get('port', stt.get('port', 50051))}",
        f"STT_FALLBACK_MODEL={stt_fallback.get('model', '')}",
        f"STT_NOISE_CORRECTION_ENABLED={'true' if stt_noise.get('enabled', False) else 'false'}",
        f"STT_NOISE_GATE_RMS={stt_noise.get('gate_rms', 120)}",
        f"STT_NOISE_SUPPRESSION_TERMS={suppression_terms_env}",
        f"STT_SPAM_GUARD_WINDOW_S={stt_noise.get('spam_guard_window_s', 1.0)}",
        f"STT_EP_START_HISTORY={stt_endpointing.get('start_history', 0)}",
        f"STT_EP_START_THRESHOLD={stt_endpointing.get('start_threshold', 0.0)}",
        f"STT_EP_STOP_HISTORY={stt_endpointing.get('stop_history', 0)}",
        f"STT_EP_STOP_THRESHOLD={stt_endpointing.get('stop_threshold', 0.0)}",
        f"STT_EP_STOP_HISTORY_EOU={stt_endpointing.get('stop_history_eou', 0)}",
        f"STT_EP_STOP_THRESHOLD_EOU={stt_endpointing.get('stop_threshold_eou', 0.0)}",
        f"ENABLE_TTS={'true' if tts.get('enabled', True) else 'false'}",
        f"TTS_PROVIDER={tts.get('provider', 'vibevoice')}",
        "",
        "# TTS Providers",
        f"RIVA_HOST={tts.get('riva', {}).get('host', 'riva-server')}",
        f"RIVA_PORT={tts.get('riva', {}).get('port', 50051)}",
        f"ENABLE_RIVA_TTS={'true' if tts.get('provider') == 'riva' else 'false'}",
        f"ENABLE_DASHSCOPE_TTS={'true' if tts.get('provider') == 'qwen' else 'false'}",
        "",
        "# NVR",
        f"ENABLE_NVR={'true' if nvr.get('enabled', False) else 'false'}",
        f"MAIL={nvr_creds.get('email', 'viture@test.com')}",
        f"PASSWORD={nvr_creds.get('password', 'test1234')}",
        f"GROUP_KEY={shinobi.get('group_key', '')}",
        f"UNIQUE_ID={shinobi.get('unique_id', '')}",
        f"API_KEY={shinobi.get('api_key', '')}",
        f"STORAGE_SIZE={shinobi.get('storage_size', 300000)}",
        "",
        "# Docker Images",
        f"RTSP_IMAGE=labos_streaming:latest",
        "",
        "# Features",
        f"ENABLE_FAST_PATH={'true' if cfg.get('features', {}).get('fast_path_enabled', False) else 'false'}",
        "",
        "# Audio / Video Forwarding",
        f"FORWARD_AUDIO=false",
        f"FORWARD_FRAMES={'true' if cfg.get('frame_streaming', {}).get('enabled', False) else 'false'}",
        f"FRAME_WIDTH={cfg.get('frame_streaming', {}).get('width', 640)}",
        f"FRAME_HEIGHT={cfg.get('frame_streaming', {}).get('height', 480)}",
        f"FRAME_FPS={cfg.get('frame_streaming', {}).get('fps', 15)}",
        f"RTSP_EXTERNAL_HOST={_rtsp_host}",
        "",
        "# LabOS Live Session",
        f"INITIAL_QR_CODE={'true' if cfg.get('labos_live', {}).get('initial_qr_code', False) else 'false'}",
        "",
        "# Recording",
        f"RECORDING_ENABLED={'true' if cfg.get('recording', {}).get('enabled', False) else 'false'}",
        f"RECORDINGS_PATH={cfg.get('recording', {}).get('path', './recordings')}",
        f"RECORDING_FPS={cfg.get('recording', {}).get('framerate', 15)}",
        "",
        "# Session",
        f"RESET_SESSION_ON_DISCONNECT={session.get('reset_on_disconnect', 'false')}",
        "",
        "# Robot Runtime",
        f"ENABLE_ROBOT={'true' if cfg.get('robot', {}).get('enabled', False) else 'false'}",
        f"XARM_IP={cfg.get('robot', {}).get('xarm_ip', '192.168.1.185')}",
        f"ROBOT_SESSION_ID={cfg.get('robot', {}).get('session_id', 'robot-1')}",
        f"ROBOT_NO_VISION={'true' if cfg.get('robot', {}).get('no_vision', False) else 'false'}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MediaMTX config (XR-side)
# ---------------------------------------------------------------------------

def generate_mediamtx_config(cfg: dict) -> str:
    lines = [
        "# Auto-generated by scripts/configure.py",
        "",
        "api: yes",
        "apiAddress: 0.0.0.0:9997",
        "",
        "rtsp: yes",
        "rtspAddress: :8554",
        "",
        "hls: yes",
        "hlsAddress: :8888",
        "",
        "webrtc: yes",
        "webrtcAddress: :8889",
        "",
        "paths:",
        "  all:",
        "    source: publisher",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TTS models config
# ---------------------------------------------------------------------------

def generate_tts_models(cfg: dict) -> dict:
    tts = cfg.get("speech", {}).get("tts", {})
    riva_cfg = tts.get("riva", {})
    qwen_cfg = tts.get("qwen", {})
    vibe_cfg = tts.get("vibevoice", {})
    el_cfg = tts.get("elevenlabs", {})

    return {
        "models": {
            "riva": {
                "name": "Riva TTS",
                "provider": "riva",
                "description": "NVIDIA Riva gRPC TTS",
                "type": "grpc",
                "enabled": tts.get("provider") == "riva",
                "host": riva_cfg.get("host", "riva-server"),
                "port": riva_cfg.get("port", 50051),
                "default_voice": "English-US.Female-1",
                "sample_rate": 48000,
                "channels": 1,
                "voices": [{"name": "English-US.Female-1", "language": "en", "sample_rate": 48000}],
            },
            "qwen-tts": {
                "name": "Qwen TTS",
                "provider": "qwen",
                "description": "Alibaba DashScope Qwen TTS",
                "type": "http",
                "enabled": tts.get("provider") == "qwen",
                "base_url": qwen_cfg.get("base_url", "https://dashscope.aliyuncs.com/api/v1"),
                "api_key_env": "DASHSCOPE_API_KEY",
                "default_voice": "loongstella-v1",
                "sample_rate": 24000,
                "channels": 1,
                "voices": [{"name": "loongstella-v1", "language": "zh", "sample_rate": 24000}],
            },
            "vibevoice": {
                "name": "VibeVoice TTS",
                "provider": "vibevoice",
                "description": "VibeVoice HTTP TTS server",
                "type": "http",
                "enabled": tts.get("provider") == "vibevoice",
                "host": vibe_cfg.get("host", "tts"),
                "port": vibe_cfg.get("port", 8050),
                "default_voice": "en-Emma_woman",
                "sample_rate": 44100,
                "channels": 1,
                "voices": [{"name": "en-Emma_woman", "language": "en", "sample_rate": 44100}],
            },
            "elevenlabs": {
                "name": "ElevenLabs TTS",
                "provider": "elevenlabs",
                "description": "ElevenLabs cloud TTS (streaming)",
                "type": "http",
                "enabled": tts.get("provider") == "elevenlabs",
                "default_voice": el_cfg.get("voice_id", "pNInz6obpgDQGcFmaJgB"),
                "model_id": el_cfg.get("model_id", "eleven_multilingual_v2"),
                "output_format": "mp3_22050_32",
                "sample_rate": 22050,
                "channels": 1,
                "voices": [{"name": "Adam", "language": "en", "voice_id": "pNInz6obpgDQGcFmaJgB"}],
            },
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not CONFIG_FILE.exists():
        print(f"Error: {CONFIG_FILE} not found", file=sys.stderr)
        print("  Run install.sh/install.bat first, or copy config.yaml.example to config.yaml")
        sys.exit(1)

    print("Loading config/config.yaml ...")
    cfg = load_config()

    secrets_file = cfg.get("global", {}).get("secrets_file", "config/.env.secrets")
    secrets_path = ROOT / secrets_file
    if not secrets_path.exists():
        print(f"  Warning: {secrets_path} not found -- secrets will be empty.")
        print(f"  Copy config/.env.secrets.example to config/.env.secrets and fill in your keys.")
    secrets = load_secrets(secrets_path)

    # Warn about empty secrets that the current config needs
    stt_proto = cfg.get("speech", {}).get("stt", {}).get("protocol", "")
    tts_prov = cfg.get("speech", {}).get("tts", {}).get("provider", "")
    if stt_proto in ("elevenlabs", "elevenlabs_realtime") or tts_prov == "elevenlabs":
        if not secrets.get("ELEVENLABS_API_KEY"):
            print("  Warning: ELEVENLABS_API_KEY is empty in config/.env.secrets")

    # .env (with real secrets)
    env_content = generate_env(cfg, secrets)
    env_path = ROOT / ".env"
    env_path.write_text(env_content)
    print(f"  Generated {env_path}")

    # .env.example (secrets redacted, safe to commit)
    empty_secrets = {k: "" for k in secrets}
    example_content = generate_env(cfg, empty_secrets)
    example_content = example_content.replace(
        "# Auto-generated by scripts/configure.py -- do not edit manually.",
        "# Auto-generated by scripts/configure.py -- safe to commit.\n"
        "# Copy to .env and fill in secrets from config/.env.secrets.",
    )
    example_path = ROOT / ".env.example"
    example_path.write_text(example_content)
    print(f"  Generated {example_path}")

    # mediamtx.yml
    mtx_content = generate_mediamtx_config(cfg)
    mtx_path = ROOT / "xr_runtime" / "streaming" / "mediamtx.yml"
    mtx_path.write_text(mtx_content)
    print(f"  Generated {mtx_path}")

    # tts_models.yaml
    tts_models = generate_tts_models(cfg)
    tts_path = ROOT / "xr_runtime" / "speech" / "tts_pusher" / "tts_models.yaml"
    tts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tts_path, "w") as f:
        yaml.dump(tts_models, f, default_flow_style=False, sort_keys=False)
    print(f"  Generated {tts_path}")

    print("Done.")


if __name__ == "__main__":
    main()
