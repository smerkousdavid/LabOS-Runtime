#!/usr/bin/env python3
"""LabOS XR Runtime -- External service connectivity tests.

Tests reachability of NAT, STT, and TTS services defined in config/config.yaml.
Can be run standalone or imported by the launcher.

Usage:
    python scripts/test_models.py              # uses config/config.yaml
    python scripts/test_models.py --config path/to/config.yaml
"""

import os
import socket
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
console = Console()

TIMEOUT_SEC = 4


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    endpoint: str
    ok: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------

def _tcp_connect(host: str, port: int) -> tuple[bool, str]:
    """Attempt a raw TCP connection."""
    try:
        sock = socket.create_connection((host, port), timeout=TIMEOUT_SEC)
        sock.close()
        return True, ""
    except OSError as exc:
        return False, str(exc)


def _http_get(url: str) -> tuple[bool, str]:
    """Attempt an HTTP(S) GET (any 2xx/3xx counts as reachable)."""
    try:
        req = urllib.request.Request(url, method="GET")
        resp = urllib.request.urlopen(req, timeout=TIMEOUT_SEC)
        resp.close()
        return True, ""
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Service testers
# ---------------------------------------------------------------------------

def _test_nat(cfg: dict) -> Optional[TestResult]:
    url = cfg.get("nat_server", {}).get("url", "")
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "wss" else 8002)
    ok, detail = _tcp_connect(host, port)
    return TestResult("NAT Server", url, ok, detail)


def _test_stt(cfg: dict) -> Optional[TestResult]:
    stt = cfg.get("speech", {}).get("stt", {})
    host = stt.get("host", "")
    port = int(stt.get("port", 50051))
    protocol = stt.get("protocol", "grpc")
    model = stt.get("model", "")
    if not host:
        return None
    endpoint = f"{host}:{port} ({protocol})"

    ok, detail = _tcp_connect(host, port)
    if not ok:
        return TestResult(f"STT ({protocol})", endpoint, False, detail)

    if protocol in ("grpc", "grpc-batch"):
        ok, detail = _grpc_asr_probe(host, port, protocol, model)
    elif protocol == "vllm":
        ok, detail = _vllm_probe(host, port, model)
    elif protocol == "parakeet_ws":
        ok, detail = _parakeet_ws_probe(host, port, model)
    elif protocol == "elevenlabs":
        ok, detail = _elevenlabs_stt_probe(stt)
    return TestResult(f"STT ({protocol})", endpoint, ok, detail)


def _grpc_asr_probe(host: str, port: int, protocol: str, model: str = "") -> tuple[bool, str]:
    """Send a tiny recognition request to verify an ASR model is loaded."""
    try:
        import riva.client
    except ImportError:
        return True, "nvidia-riva-client not installed; skipped model probe"

    try:
        import struct
        silence = struct.pack("<" + "h" * 16000, *([0] * 16000))  # 1s silence

        auth = riva.client.Auth(uri=f"{host}:{port}", use_ssl=False)
        asr = riva.client.ASRService(auth)

        config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=16000,
            language_code="en-US",
            max_alternatives=1,
            audio_channel_count=1,
        )
        if model:
            config.model = model

        if protocol == "grpc-batch":
            asr.offline_recognize(silence, config)
        else:
            streaming_cfg = riva.client.StreamingRecognitionConfig(
                config=config, interim_results=False,
            )
            responses = asr.streaming_response_generator(
                audio_chunks=iter([silence]),
                streaming_config=streaming_cfg,
            )
            for _ in responses:
                pass

        return True, ""
    except Exception as exc:
        msg = str(exc)
        if "Unavailable model" in msg or "not available on server" in msg:
            return False, "port open but no ASR model loaded on server"
        return False, msg[:120]


def _vllm_probe(host: str, port: int, model: str = "") -> tuple[bool, str]:
    """Check vLLM /health and verify the model is listed in /v1/models."""
    import json

    base = f"http://{host}:{port}"
    ok, detail = _http_get(f"{base}/health")
    if not ok:
        return False, f"health check failed: {detail[:80]}"

    if model:
        try:
            req = urllib.request.Request(f"{base}/v1/models", method="GET")
            resp = urllib.request.urlopen(req, timeout=TIMEOUT_SEC)
            data = json.loads(resp.read().decode())
            resp.close()
            model_ids = [m.get("id", "") for m in data.get("data", [])]
            if model not in model_ids:
                return False, f"model '{model}' not in server (available: {model_ids})"
        except Exception as exc:
            return False, f"model list check failed: {str(exc)[:80]}"

    return True, ""


def _parakeet_ws_probe(host: str, port: int, model: str = "") -> tuple[bool, str]:
    """Check Parakeet WS /health and verify the model is listed in /v1/models."""
    import json

    base = f"http://{host}:{port}"
    ok, detail = _http_get(f"{base}/health")
    if not ok:
        return False, f"health check failed: {detail[:80]}"

    if model:
        try:
            req = urllib.request.Request(f"{base}/v1/models", method="GET")
            resp = urllib.request.urlopen(req, timeout=TIMEOUT_SEC)
            data = json.loads(resp.read().decode())
            resp.close()
            model_ids = [m.get("id", "") for m in data.get("data", [])]
            if model not in model_ids:
                return False, f"model '{model}' not in server (available: {model_ids})"
        except Exception as exc:
            return False, f"model list check failed: {str(exc)[:80]}"

    return True, ""


def _elevenlabs_stt_probe(stt_cfg: dict) -> tuple[bool, str]:
    """Verify ElevenLabs API key by hitting the /v1/models endpoint."""
    api_key = stt_cfg.get("api_key") or os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        return False, "ELEVENLABS_API_KEY not set"
    try:
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/models",
            headers={"xi-api-key": api_key},
        )
        resp = urllib.request.urlopen(req, timeout=TIMEOUT_SEC)
        resp.close()
        return True, ""
    except Exception as exc:
        return False, f"ElevenLabs API check failed: {str(exc)[:80]}"


def _elevenlabs_tts_probe(tts_cfg: dict) -> tuple[bool, str]:
    """Verify ElevenLabs API key for TTS by listing voices."""
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        return False, "ELEVENLABS_API_KEY not set"
    try:
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": api_key},
        )
        resp = urllib.request.urlopen(req, timeout=TIMEOUT_SEC)
        resp.close()
        return True, ""
    except Exception as exc:
        return False, f"ElevenLabs API check failed: {str(exc)[:80]}"


def _test_tts(cfg: dict) -> Optional[TestResult]:
    tts = cfg.get("speech", {}).get("tts", {})
    if not tts.get("enabled", False):
        return None

    provider = tts.get("provider", "vibevoice")

    if provider == "vibevoice":
        vibe = tts.get("vibevoice", {})
        host = vibe.get("host", "")
        port = int(vibe.get("port", 8050))
        if not host:
            return None
        url = f"http://{host}:{port}"
        ok, detail = _tcp_connect(host, port)
        return TestResult("TTS (VibeVoice)", url, ok, detail)

    if provider == "riva":
        riva = tts.get("riva", {})
        host = riva.get("host", "riva-server")
        port = int(riva.get("port", 50051))
        endpoint = f"{host}:{port} (gRPC)"
        ok, detail = _tcp_connect(host, port)
        return TestResult("TTS (Riva)", endpoint, ok, detail)

    if provider == "qwen":
        qwen = tts.get("qwen", {})
        base_url = qwen.get("base_url", "")
        if not base_url:
            return None
        ok, detail = _http_get(base_url)
        return TestResult("TTS (Qwen)", base_url, ok, detail)

    if provider == "elevenlabs":
        ok, detail = _elevenlabs_tts_probe(tts)
        return TestResult("TTS (ElevenLabs)", "api.elevenlabs.io", ok, detail)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tests(cfg: dict) -> list[TestResult]:
    """Run all connectivity tests and return results."""
    tests = [_test_nat, _test_stt, _test_tts]
    results: list[TestResult] = []
    for fn in tests:
        r = fn(cfg)
        if r is not None:
            results.append(r)
    return results


def print_results(results: list[TestResult], *, console: Console = console) -> None:
    """Pretty-print test results."""
    if not results:
        console.print("[dim]  No external services configured.[/dim]")
        return

    max_name = max(len(r.name) for r in results)
    lines = Text()

    for r in results:
        if r.ok:
            mark = Text(" ✓ ", style="bold green")
        else:
            mark = Text(" ✗ ", style="bold red")
        name = Text(f" {r.name:<{max_name}}  ", style="bold")
        endpoint = Text(r.endpoint, style="dim")
        lines.append(mark)
        lines.append(name)
        lines.append(endpoint)
        if not r.ok and r.detail:
            lines.append(Text(f"  ({r.detail})", style="dim red"))
        lines.append("\n")

    passed = sum(1 for r in results if r.ok)
    total = len(results)
    if passed == total:
        summary = Text(f"  All {total} services ready.", style="green")
    else:
        summary = Text(f"  {total - passed} of {total} services failed.", style="yellow")
    lines.append("\n")
    lines.append(summary)

    console.print(Panel(lines, title="Connectivity Check", border_style="blue", padding=(0, 1)))


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test external service connectivity")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "config.yaml"),
                        help="Path to config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        console.print(f"[red]Config not found:[/red] {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Load secrets so API keys are available in os.environ
    secrets_file = cfg.get("global", {}).get("secrets_file", "config/.env.secrets")
    secrets_path = ROOT / secrets_file
    if secrets_path.exists():
        with open(secrets_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if value and key not in os.environ:
                    os.environ[key] = value

    console.print()
    results = run_tests(cfg)
    print_results(results)
    console.print()

    failed = sum(1 for r in results if not r.ok)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
