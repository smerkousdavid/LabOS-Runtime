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

    context = {
        "num_cameras": num_cameras,
        "streaming_method": streaming_method,
        "default_framerate": default_framerate,
        "base_ports": base_ports,
        "mediamtx_service_name": os.environ.get("MEDIAMTX_SERVICE_NAME", "mediamtx"),
        "rtsp_image": os.environ.get("RTSP_IMAGE", "labos_streaming:latest"),
        "ENABLE_TTS": os.environ.get("ENABLE_TTS", "false").lower() in ("1", "true", "yes"),
        "ENABLE_NVR": os.environ.get("ENABLE_NVR", "false").lower() in ("1", "true", "yes"),
        "nat_server_url": nat_url,
        "nat_ws_url": nat_url,
        "stt_host": os.environ.get("STT_HOST", "localhost"),
        "stt_port": os.environ.get("STT_PORT", "50051"),
        "stt_protocol": os.environ.get("STT_PROTOCOL", "grpc"),
        "tts_pusher_url": "http://tts-pusher:5000",
        "dashboard_url": "http://dashboard:5000",
        "forward_audio": os.environ.get("FORWARD_AUDIO", "false"),
        "tts_model": os.environ.get("TTS_PROVIDER", "vibevoice"),
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
