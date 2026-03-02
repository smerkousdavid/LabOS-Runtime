#!/usr/bin/env python3
"""TTS Mixer: Publishes continuous silent RTSP stream and plays uploaded WAVs on request.

Endpoints:
- POST /play  - accept WAV bytes and play them on the RTSP stream (with index parameter for per-camera routing)
- GET /status - return running status and queue size for all mixers
"""
import argparse
import io
import os
import queue
import subprocess
import threading
import time
import tempfile
from flask import Flask, request, jsonify
from loguru import logger

logger.remove()
logger.add(lambda msg: print(msg, end=''), level='INFO')
_log_dir = os.environ.get("LOG_DIR", "/app/logs")
if os.path.isdir(_log_dir):
    logger.add(os.path.join(_log_dir, "tts_mixer.log"), rotation="20 MB", retention="3 days", level="DEBUG")

app = Flask(__name__)

SAMPLE_RATE = 48000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_FRAMES = 1024
CHUNK_BYTES = CHUNK_FRAMES * SAMPLE_WIDTH * CHANNELS
WRITE_SLEEP = CHUNK_FRAMES / SAMPLE_RATE


class TTSPublisherMixer:
    def __init__(self, fifo_path: str, rtsp_url: str):
        self.fifo = fifo_path
        self.rtsp_url = rtsp_url
        self.queue = queue.Queue(maxsize=512)
        self.running = threading.Event()
        self.running.set()
        self.ffmpeg_proc = None
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.monitor_thread = threading.Thread(target=self._monitor_ffmpeg, daemon=True)
        self.keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self.last_audio_time = time.time()  # Track when audio was last played
        self.last_write_time = time.time()  # Track when we last wrote to FIFO

    def start(self):
        logger.info(f'Starting TTS mixer for {self.rtsp_url}...')
        # ensure FIFO
        try:
            if os.path.exists(self.fifo):
                os.unlink(self.fifo)
            os.mkfifo(self.fifo)
        except Exception as e:
            logger.warning('Could not create FIFO: ' + str(e))
        self._ensure_ffmpeg()
        self.writer_thread.start()
        self.monitor_thread.start()
        self.keepalive_thread.start()

    def stop(self):
        logger.info(f'Stopping TTS mixer for {self.rtsp_url}')
        self.running.clear()
        try:
            if self.ffmpeg_proc:
                self.ffmpeg_proc.terminate()
        except Exception:
            pass

    def _ensure_ffmpeg(self):
        if self.ffmpeg_proc and self.ffmpeg_proc.poll() is None:
            return
        cmd = [
            'ffmpeg', '-fflags', '+nobuffer+genpts+igndts', '-re', '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', self.fifo,
            '-c:a', 'aac', '-b:a', '64k', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-flush_packets', '1', '-max_delay', '0', '-muxdelay', '0',  # Prevent buffering delays
            '-f', 'rtsp', '-rtsp_transport', 'tcp', self.rtsp_url
        ]
        logger.info('Starting ffmpeg: ' + ' '.join(cmd))
        self.ffmpeg_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _monitor_ffmpeg(self):
        while self.running.is_set():
            if not self.ffmpeg_proc or self.ffmpeg_proc.poll() is not None:
                logger.warning('ffmpeg not running, restarting')
                self._ensure_ffmpeg()
            # Check if we haven't written anything for 2 minutes (stale stream)
            elif time.time() - self.last_write_time > 120:
                logger.warning('No writes for 2 minutes, restarting ffmpeg to prevent stale stream')
                if self.ffmpeg_proc:
                    self.ffmpeg_proc.terminate()
                    try:
                        self.ffmpeg_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.ffmpeg_proc.kill()
                self._ensure_ffmpeg()
            time.sleep(5)  # Check every 5 seconds instead of 1

    def _keepalive_loop(self):
        """Send periodic keepalive packets to prevent RTSP timeout"""
        while self.running.is_set():
            # Send a very quiet audio blip every 30 seconds if no audio in last 60 seconds
            if time.time() - self.last_audio_time > 60:
                # Create a very quiet 100ms blip (barely audible keepalive)
                keepalive_samples = int(SAMPLE_RATE * 0.1)  # 100ms
                quiet_blip = b'\x01\x00' * keepalive_samples  # Very quiet 16-bit samples
                try:
                    self.queue.put(quiet_blip, timeout=1)
                    logger.debug('Sent keepalive audio blip')
                except queue.Full:
                    logger.warning('Keepalive queue full, skipping')
            time.sleep(30)  # Check every 30 seconds

    def _writer_loop(self):
        while self.running.is_set():
            try:
                logger.info('Opening FIFO for writing: ' + self.fifo)
                with open(self.fifo, 'wb', buffering=0) as f:
                    logger.info('FIFO opened')
                    while self.running.is_set():
                        try:
                            try:
                                data = self.queue.get_nowait()
                            except queue.Empty:
                                data = None
                            if data is None:
                                f.write(b'\x00' * CHUNK_BYTES)
                                time.sleep(WRITE_SLEEP)
                            else:
                                self.last_audio_time = time.time()
                                buf = io.BytesIO(data)
                                while True:
                                    chunk = buf.read(CHUNK_BYTES)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                                    self.last_write_time = time.time()  # Track last write time
                                    # os.fsync(f.fileno())  # Removed as it causes OSError on pipes
                                    time.sleep(WRITE_SLEEP)
                        except BrokenPipeError:
                            logger.warning('Broken pipe to ffmpeg; will reopen')
                            break
                        except Exception:
                            logger.exception('Error in writer loop')
                            break
            except Exception:
                logger.exception('Could not open FIFO for write; retrying')
            time.sleep(0.5)

    def play_wav(self, wav_bytes: bytes):
        # decode to raw s16le using ffmpeg and enqueue the raw bytes
        logger.info('Decoding and enqueuing WAV')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            f.write(wav_bytes)
            temp_path = f.name
        cmd = [
            'ffmpeg', '-i', temp_path, '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS), '-hide_banner', '-loglevel', 'error', '-'
        ]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            while True:
                chunk = p.stdout.read(CHUNK_BYTES)
                if not chunk:
                    break
                # put into queue; will block if queue is full
                self.queue.put(chunk)
            p.wait(timeout=5)
        except Exception:
            logger.exception('Decoding failed')
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def status(self):
        ffmpeg_alive = self.ffmpeg_proc and self.ffmpeg_proc.poll() is None
        time_since_write = time.time() - self.last_write_time
        healthy = ffmpeg_alive and time_since_write < 120  # Healthy if ffmpeg running and wrote within 2 minutes
        return {
            'running': self.running.is_set(),
            'queued': self.queue.qsize(),
            'rtsp_url': self.rtsp_url,
            'ffmpeg_alive': ffmpeg_alive,
            'last_write_seconds': int(time_since_write),
            'healthy': healthy
        }


class TTSMixerManager:
    def __init__(self, num_cameras: int, mediamtx_service_name: str = 'mediamtx_glasses'):
        self.num_cameras = num_cameras
        self.mediamtx_service_name = mediamtx_service_name
        self.mixers = {}
        for i in range(1, num_cameras + 1):
            index_str = f"{i:04d}"
            fifo_path = f'/tmp/tts_mixer_{i}.pcm'
            rtsp_url = f'rtsp://{mediamtx_service_name}:8554/NB_{index_str}_RX_TTS'
            mixer = TTSPublisherMixer(fifo_path, rtsp_url)
            self.mixers[i] = mixer

    def start_all(self):
        for mixer in self.mixers.values():
            mixer.start()

    def stop_all(self):
        for mixer in self.mixers.values():
            mixer.stop()

    def play_wav_on_index(self, index: int, wav_bytes: bytes):
        if index not in self.mixers:
            raise ValueError(f'Invalid camera index {index}. Must be 1-{self.num_cameras}')
        self.mixers[index].play_wav(wav_bytes)

    def get_status_all(self):
        return {f'camera_{i}': mixer.status() for i, mixer in self.mixers.items()}


# Global manager instance
mixer_manager = None


@app.route('/play', methods=['POST'])
def play():
    try:
        index = request.args.get('index', default=1, type=int)
        if 'file' in request.files:
            wav_bytes = request.files['file'].read()
        else:
            wav_bytes = request.get_data()
        if not wav_bytes:
            return jsonify({'error': 'no data'}), 400
        mixer_manager.play_wav_on_index(index, wav_bytes)
        return jsonify({'status': 'ok', 'index': index})
    except Exception as e:
        logger.exception('play endpoint error')
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    return jsonify(mixer_manager.get_status_all())


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring"""
    all_status = mixer_manager.get_status_all()
    all_healthy = all(status['healthy'] for status in all_status.values())
    return jsonify({
        'healthy': all_healthy,
        'details': all_status
    }), 200 if all_healthy else 503


@app.route('/inject_test_tone', methods=['POST'])
def inject_test_tone():
    """Inject a test tone for debugging stream issues"""
    try:
        index = request.args.get('index', default=1, type=int)
        duration = request.args.get('duration', default=1.0, type=float)  # seconds
        
        # Generate a 440Hz sine wave test tone
        import math
        frequency = 440
        num_samples = int(SAMPLE_RATE * duration)
        test_tone = bytearray()
        
        for i in range(num_samples):
            # Generate sine wave sample (-32768 to 32767)
            sample = int(16384 * math.sin(2 * math.pi * frequency * i / SAMPLE_RATE))
            # Convert to 16-bit little-endian
            test_tone.extend(sample.to_bytes(2, 'little', signed=True))
        
        mixer_manager.play_wav_on_index(index, bytes(test_tone))
        return jsonify({'status': 'ok', 'index': index, 'duration': duration, 'frequency': frequency})
    except Exception as e:
        logger.exception('Test tone injection failed')
        return jsonify({'error': str(e)}), 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cameras', default=os.environ.get('NUM_CAMERAS', '1'), type=int)
    parser.add_argument('--mediamtx-service-name', default=os.environ.get('MEDIAMTX_SERVICE_NAME', 'mediamtx_glasses'))
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=5002, type=int)
    args = parser.parse_args()
    global mixer_manager
    mixer_manager = TTSMixerManager(args.num_cameras, args.mediamtx_service_name)
    mixer_manager.start_all()
    logger.info(f'Started tts_mixer with {args.num_cameras} cameras, publishing to {args.mediamtx_service_name}:8554/NB_XXXX_RX_TTS [1-{args.num_cameras}] and serving play on {args.port}')
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        mixer_manager.stop_all()


if __name__ == '__main__':
    main()
