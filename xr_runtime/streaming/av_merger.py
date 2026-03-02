import sys
import os as _os
from loguru import logger
logger.remove()
_xr_filter = lambda record: not (
    record["name"].startswith("xr_service_library") and
    record["level"].name == "WARNING"
)
logger.add(sys.stderr, level="INFO", filter=_xr_filter)

import argparse
import threading
import time
import subprocess
import sys


class AVMerger:
    def __init__(self, video_url, audio_url, output_url):
        """Initialize AV merger that pulls video and audio streams from MediaMTX and merges them"""
        self.video_url = video_url
        self.audio_url = audio_url
        self.output_url = output_url

        # Runtime state
        self.is_running = False           # overall service running
        self.pushing = False              # whether ffmpeg is currently pushing
        self.ffmpeg_process = None

        # Threads
        self.merger_thread = None
        self._stderr_thread = None

        # Controls
        self._stop_event = threading.Event()

        logger.info(f"AV merger initialized - merging {video_url} + {audio_url} → {output_url}")

    def _check_stream_available(self, url, timeout=3):
        """Check if RTSP stream is available using ffprobe"""
        try:
            result = subprocess.run(
                ['ffprobe', '-rtsp_transport', 'tcp',
                 '-v', 'quiet', '-print_format', 'json', '-show_streams', url],
                timeout=timeout, capture_output=True, text=True
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False

    def _wait_for_streams(self, timeout=30):
        """Wait for both input streams to be available"""
        logger.info("Waiting for input streams to be available...")
        start_time = time.time()

        while time.time() - start_time < timeout and not self._stop_event.is_set():
            video_ok = self._check_stream_available(self.video_url)
            audio_ok = self._check_stream_available(self.audio_url)

            if video_ok and audio_ok:
                logger.info("Both input streams are available")
                return True
            elif video_ok:
                logger.debug("Video stream ready, waiting for audio...")
            elif audio_ok:
                logger.debug("Audio stream ready, waiting for video...")
            else:
                logger.debug("Waiting for both streams...")

            time.sleep(1.0)

        if self._stop_event.is_set():
            return False

        logger.error(f"Timeout waiting for streams after {timeout}s")
        return False

    def _start_ffmpeg(self) -> bool:
        """Start ffmpeg process to merge video and audio streams"""
        # Build RTSP merge pipeline
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'info',
            '-fflags', 'nobuffer',  # Minimize buffering
            '-flags', 'low_delay',  # Low delay mode
            '-rtsp_transport', 'tcp',

            # Input 1: Video stream
            '-i', self.video_url,

            # Input 2: Audio stream
            '-i', self.audio_url,

            # Map streams (video from input 0, audio from input 1)
            '-map', '0:v:0',  # Video from first input
            '-map', '1:a:0',  # Audio from second input

            # Copy codecs (no re-encoding!)
            '-c:v', 'copy',
            '-c:a', 'copy',

            # Output settings
            '-rtsp_transport', 'tcp',
            '-f', 'rtsp', self.output_url
        ]

        try:
            logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.DEVNULL,  # No stdin needed
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            logger.info("FFmpeg merger process started")
            # Start stderr drain thread
            self._stderr_thread = threading.Thread(target=self._drain_ffmpeg_stderr, name="ffmpeg-stderr", daemon=True)
            self._stderr_thread.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start FFmpeg merger: {e}")
            self.ffmpeg_process = None
            return False

    def _stop_ffmpeg(self):
        """Stop ffmpeg process safely"""
        proc = self.ffmpeg_process
        self.ffmpeg_process = None
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
            except Exception:
                pass
        self.pushing = False
        logger.info("FFmpeg merger process stopped")

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

    def _merger_manager(self):
        """Thread that manages the FFmpeg merger lifecycle"""
        logger.info("AV merger manager thread started")

        while not self._stop_event.is_set():
            # Wait for input streams to be available
            if not self._wait_for_streams():
                if self._stop_event.is_set():
                    break
                logger.warning("Failed to find input streams, retrying in 5s...")
                time.sleep(5.0)
                continue

            # Start the merger if not already running
            if not self.pushing:
                if not self._start_ffmpeg():
                    logger.error("Failed to start FFmpeg merger; retrying in 5s")
                    time.sleep(5.0)
                    continue
                self.pushing = True
                logger.info("Started AV merging to RTSP")

            # Monitor the merger process
            while self.pushing and not self._stop_event.is_set():
                if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
                    logger.warning("FFmpeg merger exited; will restart when streams are ready")
                    self.pushing = False
                    break

                # Check streams are still available every 10 seconds
                time.sleep(10.0)
                if not self._check_stream_available(self.video_url, timeout=2) or \
                   not self._check_stream_available(self.audio_url, timeout=2):
                    logger.warning("Input stream(s) became unavailable; restarting merger")
                    self._stop_ffmpeg()
                    break

        # Ensure ffmpeg is stopped on exit
        if self.pushing:
            self._stop_ffmpeg()

        logger.info("AV merger manager thread exiting")

    def start(self):
        """Start the AV merger service"""
        if self.is_running:
            return True

        self._stop_event.clear()
        self.is_running = True

        # Start merger manager thread
        self.merger_thread = threading.Thread(target=self._merger_manager, name="av-merger", daemon=True)
        self.merger_thread.start()

        logger.info("AV merger started")
        return True

    def stop(self):
        """Stop the AV merger service"""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()

        # Stop thread
        if self.merger_thread and self.merger_thread.is_alive():
            self.merger_thread.join(timeout=1.0)

        # Stop ffmpeg if still running
        self._stop_ffmpeg()

        logger.info("AV merger stopped")


def main():
    parser = argparse.ArgumentParser(description='RTSP AV merger for MediaMTX')
    parser.add_argument('--video-url', type=str, required=True,
                        help='RTSP URL of video stream (e.g., rtsp://mediamtx:8554/cam1_video)')
    parser.add_argument('--audio-url', type=str, required=True,
                        help='RTSP URL of audio stream (e.g., rtsp://mediamtx:8554/cam1_audio)')
    parser.add_argument('--output-url', type=str, required=True,
                        help='RTSP URL to push merged stream to (e.g., rtsp://mediamtx:8554/cam1)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Log level (default: INFO)')

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level, filter=_xr_filter)
    _log_dir2 = _os.environ.get("LOG_DIR", "/logs")
    if _os.path.isdir(_log_dir2):
        logger.add(_os.path.join(_log_dir2, "av_merger.log"), rotation="20 MB", retention="3 days", level="DEBUG", filter=_xr_filter)

    merger = AVMerger(
        video_url=args.video_url,
        audio_url=args.audio_url,
        output_url=args.output_url
    )

    try:
        if not merger.start():
            logger.error("Failed to start AV merger")
            sys.exit(1)

        # Keep main thread alive
        logger.info(f"Merging {args.video_url} + {args.audio_url} → {args.output_url}")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error in main thread: {e}")
    finally:
        merger.stop()


if __name__ == '__main__':
    main()