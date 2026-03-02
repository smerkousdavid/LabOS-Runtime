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
    elif record["level"].name == "WARNING":
        key = "xr_warn"
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
from collections import deque

sys.path.append('../../')
from xr_service_library import XRServiceConnection
from xr_service_library.xr_types import AudioSample


class AudioPusher:
    def __init__(self, socket_path, audio_url, sample_rate=44100, channels=1, sample_format='s16le', bitrate='96k', codec='opus'):
        self.socket_path = socket_path
        self.audio_url = audio_url
        self.initial_sample_rate = int(sample_rate)
        self.sample_rate = None  # Will be detected from first audio sample
        self.channels = int(channels)
        self.sample_format = sample_format
        self.bitrate = bitrate
        self.codec = codec.lower() if codec else 'opus'

        # XR client
        self.client = XRServiceConnection(socket_path=socket_path)
        self.client.connect()

        # Runtime
        self.is_running = False
        self.pushing = False
        self.ffmpeg_process = None
        self.ffmpeg_started = False

        # Buffer and locks
        self._buffer_lock = threading.Lock()
        self._buffer = bytearray()
        self._last_received_time = 0.0
        self._last_change_time = 0.0

        # Threads
        self.fetcher_thread = None
        self.pusher_thread = None
        self.stderr_thread = None

        self._stop_event = threading.Event()

        logger.info(f"Audio pusher initialized - pushing to {audio_url}")

    def _start_ffmpeg(self):
        # Ensure sample_rate is set before starting ffmpeg
        if self.sample_rate is None:
            logger.warning("Cannot start ffmpeg: sample_rate not yet detected from audio samples")
            return False

        # Build ffmpeg command depending on codec
        base_cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'info',
            '-fflags', 'nobuffer',
            '-f', self.sample_format, '-ar', str(self.sample_rate), '-ac', str(self.channels),
            '-i', '-',
            '-af', 'aresample=resampler=soxr:ocl=48000:async=1:min_hard_comp=0.100000:first_pts=0,alimiter=limit=0.98',
        ]

        if self.codec == 'opus':
            # Opus prefers 48k internally; ffmpeg will resample if needed.
            # Note: not all ffmpeg builds expose all libopus encoder options; use -packet_loss for robustness
            codec_cmd = [
                '-c:a', 'libopus', '-b:a', self.bitrate, '-vbr', '1', '-fec', '1', '-packet_loss', '10', '-frame_duration', '20', '-application', 'audio',
            ]
        else:
            codec_cmd = [
                '-c:a', 'aac', '-profile:a', 'aac_low', '-b:a', self.bitrate,
            ]

        ffmpeg_cmd = base_cmd + codec_cmd + [
            '-rtsp_transport', 'tcp',
            '-f', 'rtsp', self.audio_url
        ]

        try:
            logger.debug("FFmpeg command: %s", ' '.join(ffmpeg_cmd))
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            logger.info(f"FFmpeg process started for audio: codec={self.codec}, sample_rate={self.sample_rate}, channels={self.channels}, format={self.sample_format}")
            self.ffmpeg_started = True
            self.stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
            self.stderr_thread.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start FFmpeg for audio ({self.codec}): {e}")
            self.ffmpeg_process = None
            # If we failed with opus, fall back to aac automatically
            if self.codec == 'opus':
                logger.warning("Falling back to AAC due to ffmpeg/codec error when using Opus")
                self.codec = 'aac'
                return self._start_ffmpeg()
            return False

    def _stop_ffmpeg(self, reason=""):
        proc = self.ffmpeg_process
        self.ffmpeg_process = None
        self.ffmpeg_started = False
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
        log_msg = "FFmpeg process stopped for audio"
        if reason:
            log_msg += f" ({reason})"
        logger.info(log_msg)

    def _drain_stderr(self):
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
            return

    def _audio_fetcher(self):
        logger.info("Audio fetcher thread started")
        first_non_silent_logged = False
        while not self._stop_event.is_set():
            try:
                samples = self.client.get_incoming_audio_samples()
            except Exception as e:
                logger.error(f"Error fetching audio samples: {e}")
                time.sleep(0.05)
                continue

            if samples:
                now = time.time()
                with self._buffer_lock:
                    for s in samples:
                        if isinstance(s, AudioSample):
                            # Detect and set sample_rate from first valid sample
                            if hasattr(s, 'sample_rate') and s.sample_rate:
                                detected_rate = int(s.sample_rate)
                                if self.sample_rate is None:
                                    self.sample_rate = detected_rate
                                    logger.info(f"Detected audio sample_rate: {self.sample_rate} Hz")
                                elif self.sample_rate != detected_rate:
                                    logger.warning(f"Sample rate changed from {self.sample_rate} to {detected_rate}! Restarting ffmpeg required.")
                                    self.sample_rate = detected_rate
                                    # Signal pusher to restart by stopping current ffmpeg
                                    if self.ffmpeg_started:
                                        logger.info("Triggering ffmpeg restart due to sample_rate change")
                                        self._stop_ffmpeg("sample_rate changed")
                            
                            # Log first non-silent chunk for diagnostics
                            if not first_non_silent_logged and len(s.audio_data) > 16:
                                zero_count = sum(1 for b in s.audio_data[:100] if b == 0)
                                if zero_count < 90:  # Less than 90% zeros
                                    logger.info(f"First non-silent audio chunk: size={len(s.audio_data)} bytes, sample_rate={s.sample_rate if hasattr(s, 'sample_rate') else 'N/A'}, first_16_bytes={s.audio_data[:16].hex()}")
                                    first_non_silent_logged = True
                            
                            self._buffer.extend(s.audio_data)
                    self._last_received_time = now
                    self._last_change_time = now
            else:
                # sleep a tiny bit
                time.sleep(0.02)

        logger.info("Audio fetcher thread exiting")

    def _has_recent_audio(self, recent_within_sec=1.0):
        now = time.time()
        with self._buffer_lock:
            last_recv = self._last_received_time
            buf_len = len(self._buffer)
        return (now - last_recv) <= recent_within_sec and buf_len > 0

    def _get_buffer_snapshot(self, max_bytes=4096):
        with self._buffer_lock:
            if not self._buffer:
                return b''
            # Ensure we always return a number of bytes that's a multiple
            # of the frame size (channels * bytes_per_sample) to avoid
            # splitting samples which causes crackling.
            bytes_per_sample = 2  # s16le => 2 bytes per sample
            frame_bytes = int(self.channels) * bytes_per_sample
            n = min(len(self._buffer), max_bytes)
            # Round down to nearest multiple of frame_bytes
            n = n - (n % frame_bytes)
            if n <= 0:
                return b''
            chunk = bytes(self._buffer[:n])
            del self._buffer[:n]
            return chunk

    def _push_manager(self):
        logger.info("Audio push manager started")
        # allow longer silence before we kill ffmpeg; we keep ffmpeg alive and write silence
        stale_timeout_sec = 60.0
        bytes_written_total = 0
        manager_start_time = time.time()
        while not self._stop_event.is_set():
            # Wait for sample_rate detection, but fallback to initial hint after 3s
            if self.sample_rate is None:
                if time.time() - manager_start_time > 3.0:
                    # fall back to initial hint
                    self.sample_rate = int(self.initial_sample_rate)
                    logger.warning(f"Falling back to initial sample_rate hint: {self.sample_rate} Hz")
                else:
                    time.sleep(0.1)
                    continue

            if not self._has_recent_audio(recent_within_sec=1.0):
                # No recent audio; keep ffmpeg alive and let writer inject silence
                if not self.pushing:
                    # start ffmpeg so it stays alive during silence
                    if not self._start_ffmpeg():
                        logger.warning("Failed to start ffmpeg for audio during silent state; retrying in 1s")
                        time.sleep(1.0)
                        continue
                    self.pushing = True
                    bytes_written_total = 0
                # if pushing already, we will write silence in the writer loop
                time.sleep(0.1)
                # fall through to writer loop

            if not self.pushing:
                if not self._start_ffmpeg():
                    logger.warning("Failed to start ffmpeg for audio; retrying in 1s")
                    time.sleep(1.0)
                    continue
                self.pushing = True
                bytes_written_total = 0

            last_change_observed_time = time.time()
            last_chunk_len = 0
            # calculate a steady write size corresponding to ~20ms of audio
            # bytes_per_sample = 2 (s16le)
            bytes_per_sample = 2
            frame_bytes = int(self.channels) * bytes_per_sample
            write_size = max(256, int(self.sample_rate * self.channels * bytes_per_sample * 0.02))
            # make write_size a multiple of frame_bytes
            write_size = write_size - (write_size % frame_bytes)
            if write_size <= 0:
                write_size = frame_bytes

            # steady writer loop: write fixed-size blocks at roughly 20ms intervals
            while self.pushing and not self._stop_event.is_set():
                if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
                    logger.warning("FFmpeg exited unexpectedly; will restart when audio is ready")
                    self.pushing = False
                    break

                start_write = time.time()
                # try to get an aligned chunk of write_size
                chunk = self._get_buffer_snapshot(write_size)
                now = time.time()
                if chunk:
                    last_change_observed_time = now
                    try:
                        self.ffmpeg_process.stdin.write(chunk)
                        bytes_written_total += len(chunk)
                        # occasional debug
                        if bytes_written_total % (self.sample_rate * bytes_per_sample * self.channels) == 0:
                            logger.debug(f"Wrote {bytes_written_total} bytes to ffmpeg")
                    except BrokenPipeError:
                        logger.error("FFmpeg stdin broken pipe for audio")
                        self.pushing = False
                        break
                    except Exception as e:
                        logger.error(f"Error writing audio to FFmpeg: {e}")
                        self.pushing = False
                        break
                else:
                    # Not enough data yet: write silence to keep encoder clock alive
                    silent_chunk = b'\x00' * write_size
                    try:
                        self.ffmpeg_process.stdin.write(silent_chunk)
                        bytes_written_total += len(silent_chunk)
                    except BrokenPipeError:
                        logger.error("FFmpeg stdin broken pipe for audio (while writing silence)")
                        self.pushing = False
                        break
                    except Exception as e:
                        logger.error(f"Error writing silence to FFmpeg: {e}")
                        self.pushing = False
                        break

                # enforce ~20ms spacing between writes to smooth bursts
                elapsed = time.time() - start_write
                sleep_time = max(0.0, 0.02 - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Stop only if we haven't received real audio for stale_timeout_sec
                if (now - self._last_received_time) > stale_timeout_sec:
                    logger.info(f"Stopping audio push due to stale audio (no data for {stale_timeout_sec}s)")
                    self._stop_ffmpeg("stale audio")
                    break

        if self.pushing:
            self._stop_ffmpeg("manager exit")

        logger.info("Audio push manager exiting")

    def start(self):
        if self.is_running:
            return True
        self._stop_event.clear()
        self.is_running = True
        self.fetcher_thread = threading.Thread(target=self._audio_fetcher, name="audio-fetcher", daemon=True)
        self.fetcher_thread.start()
        self.pusher_thread = threading.Thread(target=self._push_manager, name="audio-pusher", daemon=True)
        self.pusher_thread.start()
        logger.info("Audio pusher started")
        return True

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        self._stop_event.set()
        if self.fetcher_thread and self.fetcher_thread.is_alive():
            self.fetcher_thread.join(timeout=1.0)
        if self.pusher_thread and self.pusher_thread.is_alive():
            self.pusher_thread.join(timeout=1.0)
        self._stop_ffmpeg()
        logger.info("Audio pusher stopped")


def main():
    parser = argparse.ArgumentParser(description='RTSP audio pusher for MediaMTX')
    parser.add_argument('--socket-path', type=str, required=True, help='Path to the Unix domain socket for the XR Service')
    parser.add_argument('--audio-url', type=str, required=True, help='MediaMTX RTSP URL to push to (e.g., rtsp://mediamtx:8554/mic1)')
    parser.add_argument('--sample-rate', type=int, default=44100, help='Initial sample rate hint (will auto-detect from audio; default: 44100)')
    parser.add_argument('--channels', type=int, default=1, help='Number of audio channels (default: 1)')
    parser.add_argument('--pcm-format', type=str, default='s16le', help='PCM sample format for ffmpeg input (default: s16le)')
    parser.add_argument('--bitrate', type=str, default='128k', help='Audio bitrate for ffmpeg encoding (default: 128k)')
    parser.add_argument('--codec', type=str, default='opus', help='Audio codec to use for encoding (opus|aac)')
    parser.add_argument('--log-level', type=str, default='INFO', help='Log level (default: INFO)')

    args = parser.parse_args()
    pusher = AudioPusher(
        socket_path=args.socket_path,
        audio_url=args.audio_url,
        sample_rate=args.sample_rate,
        channels=args.channels,
        sample_format=args.pcm_format,
        bitrate=args.bitrate,
        codec=args.codec
    )

    # Re-configure loguru AFTER xr_service_library (which calls logger.configure in __init__)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, filter=_xr_lib_filter)
    _log_dir2 = _os.environ.get("LOG_DIR", "/logs")
    if _os.path.isdir(_log_dir2):
        logger.add(_os.path.join(_log_dir2, "audio_pusher.log"), rotation="20 MB", retention="3 days", level="DEBUG", filter=_xr_lib_filter)

    try:
        if not pusher.start():
            logger.error("Failed to start audio pusher")
            sys.exit(1)
        logger.info(f"Pushing audio to {args.audio_url} with sample rate {args.sample_rate} and {args.channels} channels")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error in main thread for audio pusher: {e}")
    finally:
        pusher.stop()


if __name__ == '__main__':
    main()
