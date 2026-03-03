#!/usr/bin/env python3
"""Generate Docker Compose file from Jinja2 template.

Usage: python generate.py <num_cameras> <streaming_method> <default_framerate> [output_path]
"""

import os
import sys

import jinja2


def main():
    if len(sys.argv) < 4:
        print("Usage: python generate.py <num_cameras> <streaming_method> <default_framerate> [output_path]")
        sys.exit(1)

    num_cameras = int(sys.argv[1])
    streaming_method = sys.argv[2].lower()
    default_framerate = int(sys.argv[3])
    output_path = sys.argv[4] if len(sys.argv) > 4 else "compose.yaml"

    if streaming_method not in ("gstreamer", "mediamtx"):
        print("Error: streaming_method must be 'gstreamer' or 'mediamtx'")
        sys.exit(1)

    base_ports = {
        "GRPC_PORT": 5050,
        "WEB_PORT": 5001,
    }

    nat_url = os.environ.get("NAT_SERVER_URL", "ws://localhost:8002/ws")
    stt_host = os.environ.get("STT_HOST", "localhost")

    # Inside Docker containers "localhost" refers to the container itself,
    # not the host machine.  Rewrite to the Docker-provided alias so that
    # services running on the host are reachable from inside containers.
    def _docker_host(val: str) -> str:
        return val.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")

    context = {
        "num_cameras": num_cameras,
        "streaming_method": streaming_method,
        "default_framerate": default_framerate,
        "base_ports": base_ports,
        "mediamtx_service_name": os.environ.get("MEDIAMTX_SERVICE_NAME", "mediamtx"),
        "rtsp_image": os.environ.get("RTSP_IMAGE", "labos_streaming:latest"),
        "ENABLE_TTS": os.environ.get("ENABLE_TTS", "false").lower() in ("1", "true", "yes"),
        "ENABLE_NVR": os.environ.get("ENABLE_NVR", "false").lower() in ("1", "true", "yes"),
        "nat_server_url": _docker_host(nat_url),
        "nat_ws_url": _docker_host(nat_url),
        "stt_host": _docker_host(stt_host),
        "stt_port": os.environ.get("STT_PORT", "50051"),
        "stt_protocol": os.environ.get("STT_PROTOCOL", "grpc"),
        "stt_model": os.environ.get("STT_MODEL", ""),
        "tts_pusher_url": "http://tts-pusher:5000",
        "dashboard_url": "http://dashboard:5000",
        "forward_audio": os.environ.get("FORWARD_AUDIO", "false"),
        "forward_frames": os.environ.get("FORWARD_FRAMES", "false"),
        "frame_width": os.environ.get("FRAME_WIDTH", "640"),
        "frame_height": os.environ.get("FRAME_HEIGHT", "480"),
        "frame_fps": os.environ.get("FRAME_FPS", "15"),
        "rtsp_external_host": _docker_host(os.environ.get("RTSP_EXTERNAL_HOST", "localhost")),
        "tts_model": os.environ.get("TTS_PROVIDER", "vibevoice"),
        "reset_session": os.environ.get("RESET_SESSION_ON_DISCONNECT", "false"),
        "recording_enabled": os.environ.get("RECORDING_ENABLED", "false").lower() in ("1", "true", "yes"),
        "recordings_path": os.environ.get("RECORDINGS_PATH", "./recordings"),
        "ENABLE_ROBOT": os.environ.get("ENABLE_ROBOT", "false").lower() in ("1", "true", "yes"),
        "xarm_ip": os.environ.get("XARM_IP", "192.168.1.185"),
        "robot_session_id": os.environ.get("ROBOT_SESSION_ID", "robot-1"),
        "robot_no_vision": os.environ.get("ROBOT_NO_VISION", "false"),
    }

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(__file__) or "."))
    template = env.get_template("runtime.j2")
    compose = template.render(context)

    with open(output_path, "w") as f:
        f.write(compose)

    print(
        f"Generated {output_path} with {num_cameras} cameras, "
        f"method={streaming_method}, TTS={'on' if context['ENABLE_TTS'] else 'off'}"
    )


if __name__ == "__main__":
    main()
