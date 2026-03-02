import sys
# Configure Loguru FIRST, before any imports
from loguru import logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level="INFO",
    filter=lambda record: not (
        record["name"].startswith("xr_service_library") and
        record["level"].name == "WARNING"
    )
)

import argparse
import threading
import time
import subprocess
import numpy as np
import cv2

# Add parent directory to path to import xr_service_library
sys.path.append('../../')
from xr_service_library import XRServiceConnection
from xr_service_library.xr_types import VideoFrame, Message, AudioSample, Blob


class MessageSender:
    def __init__(self, socket_path='/tmp/xr_service.sock'):
        """Initialize RTSP pusher that pushes frames from XR Service to MediaMTX via FFmpeg"""
        self.socket_path = socket_path
        # XR service client

        self.client = XRServiceConnection(socket_path=socket_path)
        self.client.connect()

def main():
    parser = argparse.ArgumentParser(description='MessageSender args')
    parser.add_argument('--socket-path', type=str, required=True,
                        help='Path to the Unix domain socket for the XR Service')
    parser.add_argument('--message_type', type=str, required=True,
                        help='Message type to send to XR Service')
    parser.add_argument('--message_payload', type=str, required=True,
                        help='Message payload to send to XR Service')

    args = parser.parse_args()

    msgSender = MessageSender(
        socket_path=args.socket_path
    )
    msg = Message(message_type=args.message_type, payload=args.message_payload)
    msgSender.client.send_message(msg)

if __name__ == '__main__':
    main()