import sys
import time as _time
from loguru import logger

_xr_lib_last_log: dict = {}
_XR_LIB_THROTTLE_SEC = 15.0

def _xr_lib_filter(record):
    """Throttle noisy xr_service_library messages to once per 15 seconds."""
    name = record["name"]
    if not name.startswith("xr_service_library"):
        return True
    if name.startswith("xr_service_library.performance_metrics"):
        key = "perf_metrics"
    elif "No frames available" in record["message"]:
        key = "no_frames"
    else:
        return True
    now = _time.time()
    last = _xr_lib_last_log.get(key, 0.0)
    if now - last < _XR_LIB_THROTTLE_SEC:
        return False
    _xr_lib_last_log[key] = now
    return True

import os as _os
logger.remove()
logger.add(sys.stderr, level="INFO", filter=_xr_lib_filter)

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


class RTSPPusher:
    def __init__(self, socket_path='/tmp/xr_service.sock', mediamtx_url='rtsp://mediamtx:8554/cam1', width=1280, height=720, framerate=30):
        """Initialize RTSP pusher that pushes frames from XR Service to MediaMTX via FFmpeg"""
        self.socket_path = socket_path
        self.mediamtx_url = mediamtx_url
        self.width = width
        self.height = height

        # XR service client
        self.framerate = framerate
        self.client = XRServiceConnection(socket_path=socket_path)
        self.client.connect()

        # Runtime state
        self.is_running = False           # overall service running
        self.pushing = False              # whether ffmpeg is currently pushing
        self.ffmpeg_process = None

        # Shared frame state
        self._frame_lock = threading.Lock()
        self._latest_frame = None                     # np.ndarray, resized to (H, W, 3), BGR
        self._latest_checksum = None                  # int fingerprint of latest frame
        self._last_frame_received_time = 0.0          # last time any frame was received
        self._last_change_time = 0.0                  # last time the frame content changed

        # Threads
        self.fetcher_thread = None
        self.pusher_thread = None
        self._stderr_thread = None

        # Controls
        self._stop_event = threading.Event()

        logger.info(f"RTSP pusher initialized - pushing to {mediamtx_url}")

    def _extract_numpy_frame(self, frame_data):
        """Extract a numpy array from various possible frame_data structures."""
        if hasattr(frame_data, 'frame'):
            frame_obj = frame_data.frame
            if hasattr(frame_obj, 'data') and hasattr(frame_obj, 'dtype') and hasattr(frame_obj, 'shape'):
                return np.frombuffer(frame_obj.data, dtype=frame_obj.dtype).reshape(frame_obj.shape)
            else:
                logger.warning(f"Frame object missing required attributes: {dir(frame_obj)}")
                return None
        elif isinstance(frame_data, np.ndarray):
            return frame_data
        elif isinstance(frame_data, dict) and 'image' in frame_data:
            return frame_data['image']
        else:
            logger.warning(f"Unexpected frame data type: {type(frame_data)}")
            return None

    def _fingerprint(self, frame: np.ndarray) -> int:
        """Fast, lightweight fingerprint to detect content change without heavy hashing."""
        # Sample every 64th byte to keep it cheap
        view = frame.view(np.uint8).ravel()
        sample = view[::64]
        # Use 64-bit accumulation and mix in size to reduce collisions
        return int(np.uint64(sample.sum()) ^ np.uint64(sample.size))

    def _start_ffmpeg(self) -> bool:
        """Start ffmpeg process to push to MediaMTX"""
        # Build RTSP push pipeline
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
            '-use_wallclock_as_timestamps', '1',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f'{self.width}x{self.height}', '-r', f'{self.framerate}',
            '-i', '-', '-vf', 'format=yuv420p',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
            '-crf', '23',
            '-keyint_min', f'{self.framerate}', '-g', f'{self.framerate}', '-sc_threshold', '0',
            '-rtsp_transport', 'tcp',
            '-f', 'rtsp', self.mediamtx_url
        ]

        try:
            # Avoid deadlock by draining stderr in a thread
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            logger.info("FFmpeg process started")
            # Start stderr drain thread
            self._stderr_thread = threading.Thread(target=self._drain_ffmpeg_stderr, name="ffmpeg-stderr", daemon=True)
            self._stderr_thread.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            self.ffmpeg_process = None
            return False

    def _stop_ffmpeg(self):
        """Stop ffmpeg process safely"""
        proc = self.ffmpeg_process
        self.ffmpeg_process = None
        if proc:
            try:
                if proc.stdin:
                    try:
                        proc.stdin.flush()
                    except Exception:
                        pass
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.pushing = False
        logger.info("FFmpeg process stopped")

    def _drain_ffmpeg_stderr(self):
        """Continuously read ffmpeg stderr to avoid blocking; log noteworthy lines."""
        proc = self.ffmpeg_process
        if not proc or not proc.stderr:
            return
        try:
            for raw in iter(proc.stderr.readline, b''):
                if not raw:
                    break
                line = raw.decode(errors='ignore').strip()
                low = line.lower()
                if any(w in low for w in ['error', 'fail', 'invalid']):
                    logger.error(f"ffmpeg: {line}")
                elif any(w in low for w in ['rtsp', 'connection', 'timeout', 'broken pipe']):
                    logger.warning(f"ffmpeg: {line}")
        except Exception:
            pass

    def _frame_fetcher(self):
        """Thread function to continuously fetch frames from XR Service"""
        logger.info("Frame fetcher thread started")
        no_frame_count = 0
        last_checksum = None
        last_refresh_time = 0.0
        refresh_backoff_sec = 30.0  # minimum gap between context refresh attempts
        consecutive_timeouts = 0

        while not self._stop_event.is_set():
            try:
                frame_data = self.client.get_latest_frame()
            except TimeoutError:
                consecutive_timeouts += 1
                logger.warning(f"XR frame request timed out (#{consecutive_timeouts})")
                time.sleep(0.1)
                # If we keep timing out for a while, try to reconnect the XR context
                if consecutive_timeouts >= 10 and (time.time() - last_refresh_time) > refresh_backoff_sec:
                    try:
                        logger.info("Reconnecting XR service after repeated timeouts...")
                        last_refresh_time = time.time()
                        # Best-effort reconnect
                        self.client.connect()
                    except Exception as re:
                        logger.error(f"Error reconnecting XR service: {re}")
                continue
            except Exception as e:
                logger.error(f"Error fetching frame from XR service: {e}")
                time.sleep(0.1)
                continue

            consecutive_timeouts = 0

            if frame_data is not None:
                try:
                    frame = self._extract_numpy_frame(frame_data)
                    if frame is not None:
                        # Ensure BGR uint8 and target size
                        if frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8, copy=False)
                        if frame.ndim == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height))

                        checksum = self._fingerprint(frame)
                        now = time.time()

                        with self._frame_lock:
                            self._latest_frame = frame
                            self._latest_checksum = checksum
                            self._last_frame_received_time = now
                            if last_checksum != checksum:
                                self._last_change_time = now

                        last_checksum = checksum
                        no_frame_count = 0
                    else:
                        no_frame_count += 1
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    no_frame_count += 1
            else:
                no_frame_count += 1
                if no_frame_count % 200 == 1:
                    logger.debug("No frames available yet")

            time.sleep(0.01)

        logger.info("Frame fetcher thread exiting")

    def _has_recent_new_frames(self, recent_within_sec: float = 1.0) -> bool:
        """Return True if we have seen a content change within the recent window."""
        with self._frame_lock:
            last_change = self._last_change_time
            last_any = self._last_frame_received_time
        now = time.time()
        # Must have received frames and a change recently
        return (now - last_any) <= recent_within_sec and (now - last_change) <= recent_within_sec

    def _get_latest_frame_snapshot(self):
        """Return a safe copy of the latest frame and its checksum, or (None, None)."""
        with self._frame_lock:
            frame = self._latest_frame
            checksum = self._latest_checksum
        if frame is None:
            return None, None
        # Copy to ensure consistency while writing to ffmpeg
        return frame.copy(), checksum

    def _push_manager(self):
        """Thread that manages the FFmpeg push lifecycle based on frame availability."""
        logger.info("Push manager thread started")

        # Stop pushing if no change for this many seconds
        stale_timeout_sec = 10.0
        _last_waiting_log = 0.0

        while not self._stop_event.is_set():
            # Wait for genuinely new frames before starting pushing
            if not self._has_recent_new_frames(recent_within_sec=1.0):
                # Ensure ffmpeg is stopped while waiting
                if self.pushing:
                    logger.info("Pausing push: no recent changes")
                    self._stop_ffmpeg()
                now = time.time()
                if now - _last_waiting_log >= 15.0:
                    logger.info("Waiting for new frames...")
                    _last_waiting_log = now
                time.sleep(0.2)
                continue

            # We have new frames; start pushing if not already
            if not self.pushing:
                if not self._start_ffmpeg():
                    logger.error("Failed to start FFmpeg; retrying in 1s")
                    time.sleep(1.0)
                    continue
                self.pushing = True
                logger.info("Started pushing to RTSP")

            # Inner loop: write frames while pushing
            last_pushed_checksum = None
            last_change_observed_time = time.time()

            while self.pushing and not self._stop_event.is_set():
                # Check ffmpeg health
                if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
                    logger.warning("FFmpeg exited; will restart when frames are ready")
                    self.pushing = False
                    break

                # Get latest frame snapshot
                frame, checksum = self._get_latest_frame_snapshot()
                now = time.time()

                if frame is None:
                    time.sleep(0.01)
                    continue
                else:
                    # Track content change time
                    if last_pushed_checksum != checksum:
                        last_change_observed_time = now
                        last_pushed_checksum = checksum

                    # Ensure contiguous BGR24 bytes
                    frame_c = np.ascontiguousarray(frame)
                    try:
                        self.ffmpeg_process.stdin.write(frame_c.tobytes())
                    except BrokenPipeError:
                        logger.error("FFmpeg stdin broken pipe")
                        self.pushing = False
                        break
                    except Exception as e:
                        logger.error(f"Error writing frame to FFmpeg: {e}")
                        self.pushing = False
                        break

                # Check for staleness: stop pushing after 10s without content change
                if (now - last_change_observed_time) > stale_timeout_sec:
                    logger.info("Stopping push due to stale content (no changes)")
                    self._stop_ffmpeg()
                    break

                # Target FPS pacing when frames are available
                time.sleep(1.0 / self.framerate)

        # Ensure ffmpeg is stopped on exit
        if self.pushing:
            self._stop_ffmpeg()

        logger.info("Push manager thread exiting")

    def start(self):
        """Start the RTSP pusher service (frame fetcher + push manager)"""
        if self.is_running:
            return True

        self._stop_event.clear()
        self.is_running = True

        # Start frame fetcher thread
        self.fetcher_thread = threading.Thread(target=self._frame_fetcher, name="frame-fetcher", daemon=True)
        self.fetcher_thread.start()

        # Start push manager thread
        self.pusher_thread = threading.Thread(target=self._push_manager, name="push-manager", daemon=True)
        self.pusher_thread.start()

        logger.info("RTSP pusher started")
        return True

    def stop(self):
        """Stop the RTSP pusher service"""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()

        # Stop threads
        if self.fetcher_thread and self.fetcher_thread.is_alive():
            self.fetcher_thread.join(timeout=1.0)
        if self.pusher_thread and self.pusher_thread.is_alive():
            self.pusher_thread.join(timeout=1.0)

        # Stop ffmpeg if still running
        self._stop_ffmpeg()

        logger.info("RTSP pusher stopped")


def main():
    parser = argparse.ArgumentParser(description='RTSP pusher for MediaMTX')
    parser.add_argument('--socket-path', type=str, required=True,
                        help='Path to the Unix domain socket for the XR Service')
    parser.add_argument('--mediamtx-url', type=str, required=True,
                        help='MediaMTX RTSP URL to push to (e.g., rtsp://mediamtx:8554/cam1)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Output video width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Output video height (default: 720)')
    parser.add_argument('--framerate', type=int, default=30,
                        help='Output video framerate (default: 30)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Log level (default: INFO)')

    args = parser.parse_args()

    pusher = RTSPPusher(
        socket_path=args.socket_path,
        mediamtx_url=args.mediamtx_url,
        width=args.width,
        height=args.height,
        framerate=args.framerate
    )

    # Re-configure loguru AFTER xr_service_library (which calls logger.configure in __init__)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, filter=_xr_lib_filter)
    _log_dir2 = _os.environ.get("LOG_DIR", "/logs")
    if _os.path.isdir(_log_dir2):
        logger.add(_os.path.join(_log_dir2, "rtsp_pusher.log"), rotation="20 MB", retention="3 days", level="DEBUG", filter=_xr_lib_filter)

    try:
        if not pusher.start():
            logger.error("Failed to start pusher")
            sys.exit(1)

        # Keep main thread alive
        logger.info(f"Pushing to {args.mediamtx_url} with resolution {args.width}x{args.height}")
        while True:
            # msg = Message(message_type='AI_RESULT', payload='{"step": "1", "status": "completed"}')
            # pusher.client.send_message(msg)
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error in main thread: {e}")
    finally:
        pusher.stop()


if __name__ == '__main__':
    main()