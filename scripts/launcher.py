#!/usr/bin/env python3
"""LabOS XR Runtime -- Central launcher.

Cross-platform Python entry point that replaces start.sh.
Called by run.sh / run.bat.
"""

import argparse
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import yaml
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: str | None = None, capture: bool = False, **kw):
    """Run a subprocess, print on failure."""
    r = subprocess.run(cmd, cwd=cwd or str(ROOT), capture_output=capture, text=True, **kw)
    if r.returncode != 0 and not capture:
        console.print(f"[red]Command failed:[/red] {' '.join(cmd)}")
    return r


def load_config() -> dict:
    cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        console.print("[red]config/config.yaml not found. Run install.sh first.[/red]")
        sys.exit(1)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def count_running_containers() -> int:
    r = run(["docker", "compose", "-f", "compose.yaml", "ps", "-q"], capture=True)
    if r.returncode != 0:
        return 0
    return len([l for l in r.stdout.strip().splitlines() if l.strip()])


def docker_compose_up(rebuild: bool = False):
    cmd = ["docker", "compose", "-f", "compose.yaml", "up", "-d"]
    if rebuild:
        cmd += ["--build", "--force-recreate"]
    run(cmd)


def docker_compose_down():
    run(["docker", "compose", "-f", "compose.yaml", "down", "--remove-orphans"])


# ---------------------------------------------------------------------------
# Service health polling
# ---------------------------------------------------------------------------

def wait_for_service(name: str, url: str, timeout: int = 30) -> bool:
    """Poll an HTTP endpoint until it returns 200."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.urlopen(url, timeout=3)
            if req.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def wait_for_xr_connection(dashboard_port: int, timeout: int = 60) -> bool:
    """Poll dashboard /api/status until xr_service_connected is true."""
    import urllib.request
    import json as _json
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(
                f"http://localhost:{dashboard_port}/api/status", timeout=3
            )
            data = _json.loads(resp.read())
            if data.get("xr_service_connected"):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def send_initial_status(dashboard_port: int, retries: int = 5) -> bool:
    """Send the initial COMPONENTS_STATUS to glasses via the dashboard, with retries."""
    import urllib.request
    import json as _json
    status_payload = _json.dumps({
        "Voice_Assistant": "idle",
        "Server_Connection": "inactive",
        "Robot_Status": "N/A",
    })
    data = _json.dumps({
        "message_type": "COMPONENTS_STATUS",
        "payload": status_payload,
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"http://localhost:{dashboard_port}/api/send_message",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def watch_grpc_logs(camera: int = 1, timeout: int = 0):
    """Watch gRPC server logs for client connection events.

    With timeout=0 this is non-blocking (one check).
    """
    r = run(
        ["docker", "compose", "-f", "compose.yaml", "logs", f"grpc-server-{camera}",
         "--tail", "50"],
        capture=True,
    )
    if r.returncode != 0:
        return False
    return "stream connected from" in r.stdout


# ---------------------------------------------------------------------------
# Log directory setup
# ---------------------------------------------------------------------------

def create_log_dirs(num_cameras: int, cfg: dict | None = None):
    dirs = ["mediamtx", "dashboard", "tts_pusher", "tts_mixer"]
    for i in range(1, num_cameras + 1):
        dirs += [f"grpc_{i}", f"video_pusher_{i}", f"audio_pusher_{i}",
                 f"av_merger_{i}", f"voice_bridge_{i}"]
    if cfg and cfg.get("robot", {}).get("enabled", False):
        dirs.append("robot_runtime")
    for d in dirs:
        (ROOT / "logs" / d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Live voice-bridge log tail
# ---------------------------------------------------------------------------

_BRIDGE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\[STT\] -> NAT: (.+)"),         "bold green"),
    (re.compile(r"\[STT\] Barge-in detected: (.+)"), "bold red"),
    (re.compile(r"\[STT\] Interim: (.+)"),         "dim"),
    (re.compile(r"\[STT\] (.+)"),                  "white"),
    (re.compile(r"\[WakeWord\] State: (.+)"),      "bold yellow"),
    (re.compile(r"\[TTS\] Speaking: (.+)"),         "bold cyan"),
    (re.compile(r"\[Session\] (.+)"),               "bold magenta"),
]


def _format_bridge_line(raw: str) -> str | None:
    """Parse a voice-bridge log line and return rich-formatted text, or None."""
    for pattern, style in _BRIDGE_PATTERNS:
        m = pattern.search(raw)
        if m:
            tag = pattern.pattern.split(r"\]")[0].replace("\\[", "") + "]"
            return f"[{style}]{tag} {m.group(1)}[/{style}]"
    return None


def _tail_voice_bridge(con: Console, reset_mode: str):
    """Tail docker compose logs for voice-bridge-1, pretty-print STT/TTS events.

    Also watches for glasses disconnect and handles session reset logic.
    """
    proc = subprocess.Popen(
        ["docker", "compose", "-f", "compose.yaml", "logs", "-f", "--tail", "0",
         "voice-bridge-1"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    glasses_was_connected = True

    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break

            formatted = _format_bridge_line(line)
            if formatted:
                con.print(formatted)

            # Detect glasses disconnect via repeated audio decoder restarts
            if "[Session] Glasses disconnected" in line:
                glasses_was_connected = False
                con.print("[bold magenta]Glasses disconnected.[/bold magenta]")

                if reset_mode == "ask":
                    answer = con.input(
                        "[yellow]Reset session (close NAT connection)? [y/N]:[/yellow] "
                    ).strip().lower()
                    if answer in ("y", "yes"):
                        con.print("[dim]Restarting voice bridge ...[/dim]")
                        run(["docker", "compose", "-f", "compose.yaml",
                             "restart", "voice-bridge-1"], capture=True)
                        con.print("[green]Voice bridge restarted (session reset).[/green]")

            # Detect glasses reconnection
            if "[Bridge] Audio stream active" in line and not glasses_was_connected:
                glasses_was_connected = True
                con.print("[bold green]Glasses reconnected.[/bold green]")

    except KeyboardInterrupt:
        con.print("\n[dim]Detached from live feed. Services continue in background.[/dim]")
    finally:
        proc.terminate()
        proc.wait(timeout=3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LabOS XR Runtime Launcher")
    parser.add_argument("--cameras", type=int, default=None, help="Number of cameras")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS")
    parser.add_argument("--no-nvr", action="store_true", help="Disable NVR/Shinobi")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild images")
    parser.add_argument("--update-glasses", action="store_true", help="Run glasses config utility")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument("--skip-checks", action="store_true", help="Skip connectivity checks")
    args = parser.parse_args()

    console.print()
    console.print(Panel("[bold]LabOS XR Runtime[/bold]", style="blue"))
    console.print()

    # ── Stop mode ─────────────────────────────────────────────────────────
    if args.stop:
        console.print("Stopping services ...")
        docker_compose_down()
        console.print("[green]All services stopped.[/green]")
        return

    # ── Load config ───────────────────────────────────────────────────────
    cfg = load_config()
    num_cameras = args.cameras or cfg.get("runtime", {}).get("num_cameras", 1)
    nat_url = cfg.get("nat_server", {}).get("url", "ws://localhost:8002/ws")

    # ── Load secrets into environment before any API checks ─────────────
    secrets_file = cfg.get("global", {}).get("secrets_file", "config/.env.secrets")
    secrets_path = ROOT / secrets_file
    if secrets_path.exists():
        with open(secrets_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if v and k not in os.environ:
                    os.environ[k] = v

    # ── Connectivity checks ───────────────────────────────────────────────
    if not args.skip_checks:
        from test_models import run_tests, print_results
        results = run_tests(cfg)
        print_results(results, console=console)
        console.print()

        failed = [r for r in results if not r.ok]
        if failed:
            answer = console.input(
                f"[yellow]{len(failed)} service(s) unreachable.[/yellow] Continue anyway? [y/N]: "
            ).strip().lower()
            if answer not in ("y", "yes"):
                console.print("[dim]Aborted.[/dim]")
                return

    # ── Glasses update ────────────────────────────────────────────────────
    if args.update_glasses:
        from update_glasses import run_update
        run_update()
    else:
        answer = console.input("Update glasses connection config? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            from update_glasses import run_update
            run_update()

    # ── Generate configs ──────────────────────────────────────────────────
    console.print("Generating configuration files ...")
    from configure import main as configure_main
    configure_main()

    # ── Source .env for compose generation ─────────────────────────────────
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ[k.strip()] = v.strip()

    if args.no_tts:
        os.environ["ENABLE_TTS"] = "false"
    if args.no_nvr:
        os.environ["ENABLE_NVR"] = "false"

    # ── Stop any running containers first ─────────────────────────────────
    running = count_running_containers()
    if running > 0:
        console.print(f"[dim]Stopping {running} running container(s) ...[/dim]")
        docker_compose_down()

    # ── Build streaming image ──────────────────────────────────────────────
    console.print("Building Docker images ...")
    streaming_df = ROOT / "xr_runtime" / "streaming" / "Dockerfile"
    if streaming_df.exists():
        run(["docker", "build", "-f", str(streaming_df), "-t", "labos_streaming:latest",
             str(ROOT / "xr_runtime")])

    # ── Create log directories ────────────────────────────────────────────
    create_log_dirs(num_cameras, cfg)

    # ── Generate compose.yaml ─────────────────────────────────────────────
    streaming = cfg.get("runtime", {}).get("streaming", {})
    method = streaming.get("method", "mediamtx")
    framerate = streaming.get("framerate", 30)
    console.print("Generating docker-compose.yaml ...")
    run([sys.executable, str(ROOT / "compose" / "generate.py"),
         str(num_cameras), method, str(framerate), str(ROOT / "compose.yaml")])

    # ── Start services ────────────────────────────────────────────────────
    console.print("Starting services ...")
    docker_compose_up(rebuild=True)

    # ── Wait for readiness ────────────────────────────────────────────────
    console.print()
    console.print("Waiting for services to start ...")

    dashboard_port = cfg.get("dashboard", {}).get("port", 5001)
    dash_ok = wait_for_service("Dashboard", f"http://localhost:{dashboard_port}/api/status", timeout=30)

    # ── Status table ──────────────────────────────────────────────────────
    table = Table(title="Service Status")
    table.add_column("Service", style="bold")
    table.add_column("Status")
    table.add_column("URL")

    table.add_row("Dashboard", "[green]ready[/green]" if dash_ok else "[yellow]starting...[/yellow]",
                  f"http://localhost:{dashboard_port}")
    table.add_row("gRPC Server", "[green]running[/green]", f"0.0.0.0:5050")
    table.add_row("MediaMTX", "[green]running[/green]", "rtsp://localhost:8554")
    table.add_row("NAT Server", "[cyan]external[/cyan]", nat_url)

    console.print(table)
    console.print()

    # ── Monitor for glasses connection ────────────────────────────────────
    console.print("Waiting for glasses to connect (Ctrl+C to run in background) ...")
    glasses_connected = False
    try:
        for _ in range(60):
            if watch_grpc_logs():
                console.print("[green bold]XR glasses connected![/green bold]")
                glasses_connected = True
                break
            time.sleep(2)
        else:
            console.print("[dim]No glasses detected yet. The runtime is running in the background.[/dim]")
    except KeyboardInterrupt:
        pass

    console.print()
    console.print("[green]Runtime is active.[/green] Services running in Docker.")
    console.print(f"  Dashboard:  http://localhost:{dashboard_port}")
    console.print(f"  Glasses:    connect to port 5050")
    console.print(f"  NAT Agent:  {nat_url}")
    console.print(f"  Logs:       {ROOT / 'logs'}/")
    console.print()

    # ── Send initial status once XR socket is ready ─────────────────────
    if glasses_connected:
        console.print("[dim]Waiting for dashboard XR socket connection ...[/dim]")
        if wait_for_xr_connection(dashboard_port, timeout=30):
            if send_initial_status(dashboard_port):
                console.print("[green]Initial COMPONENTS_STATUS sent to glasses.[/green]")
            else:
                console.print("[yellow]Warning: failed to send initial status.[/yellow]")
        else:
            console.print("[yellow]Warning: dashboard XR socket not connected yet.[/yellow]")

    # ── Live voice bridge monitoring ──────────────────────────────────────
    reset_mode = cfg.get("session", {}).get("reset_on_disconnect", "false")

    if glasses_connected:
        console.print(Panel("[bold]Live Voice Bridge Feed[/bold]  (Ctrl+C to detach)",
                            style="blue", padding=(0, 1)))
        _tail_voice_bridge(console, reset_mode)
    else:
        console.print("Run [bold]./stop.sh[/bold] or [bold]python scripts/launcher.py --stop[/bold] to shut down.")

    console.print()


if __name__ == "__main__":
    main()
