#!/usr/bin/env python3
"""Multi-Model TTS HTTP server that routes requests to different TTS providers."""

import os
import sys
import tempfile
import subprocess
import argparse
import yaml
from flask import Flask, request, jsonify
import requests
import time
import uuid
from loguru import logger
from tts_provider import TTSProviderFactory

logger.remove()
logger.add(sys.stderr, level="INFO")
_log_dir = os.environ.get("LOG_DIR", "/app/logs")
if os.path.isdir(_log_dir):
    logger.add(os.path.join(_log_dir, "tts_pusher.log"), rotation="20 MB", retention="3 days", level="DEBUG")

app = Flask(__name__)

# Global variables
tts_models = {}
available_models = []


def load_tts_models():
    """Load TTS model configurations from YAML file."""
    global tts_models, available_models

    config_path = os.environ.get('TTS_MODELS_CONFIG', '/app/tts_models.yaml')

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        tts_models = {}
        available_models = []

        for model_key, model_config in config.get('models', {}).items():
            if TTSProviderFactory.is_enabled(model_config):
                try:
                    provider = TTSProviderFactory.create_provider(model_config)
                    tts_models[model_key] = {
                        'provider': provider,
                        'config': model_config
                    }
                    available_models.append(model_key)
                    logger.info(f"Loaded TTS model: {model_config['name']} ({model_config['provider']})")
                except Exception as e:
                    logger.warning(f"Failed to load TTS model {model_key}: {e}")

        if not available_models:
            logger.warning("No TTS models are enabled or available")

    except Exception as e:
        logger.error(f"Failed to load TTS models configuration: {e}")
        tts_models = {}
        available_models = []


def synthesize_text(model: str, text: str, voice: str = None, **kwargs) -> bytes:
    """Synthesize text using specified model."""
    if model not in tts_models:
        raise ValueError(f"Model '{model}' not available. Available models: {available_models}")

    provider = tts_models[model]['provider']
    return provider.synthesize(text, voice, **kwargs)


@app.route('/synthesize', methods=['POST'])
def synthesize_route():
    data = request.get_json() or {}
    text = data.get('text')
    if not text:
        return jsonify({'error': 'missing text'}), 400

    model = data.get('model', available_models[0] if available_models else None)
    if not model:
        return jsonify({'error': 'no TTS models available'}), 503

    if model not in available_models:
        return jsonify({'error': f'model "{model}" not available. Available: {available_models}'}), 400

    voice = data.get('voice')
    index = data.get('index', 1)  # Default to camera 1 if not specified
    request_id = uuid.uuid4().hex[:8]

    model_config = tts_models[model]['config']
    logger.info(f"[req:{request_id}] Synthesize request: model={model} text='{text[:30]}' voice={voice} index={index}")

    TTS_MIXER_URL = os.environ.get('TTS_MIXER_URL', 'http://tts-mixer:5002')
    debug_save_wav = os.environ.get('DEBUG_SAVE_WAV', 'false').lower() in ('1', 'true', 'yes')

    # Synthesize to bytes
    try:
        audio_bytes = synthesize_text(model, text, voice)
    except Exception as e:
        logger.exception(f"[{model}] synthesis failed")
        return jsonify({'error': f'{model} synthesis failed: {str(e)}'}), 500

    # Save temp wav file
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp_wav.write(audio_bytes)
    tmp_wav.flush()
    tmp_wav.close()

    if debug_save_wav:
        os.makedirs('/app/logs', exist_ok=True)
        dbg_path = f'/app/logs/last_synth_{model}.wav'
        open(dbg_path, 'wb').write(audio_bytes)
        logger.info(f'DEBUG: saved wav debug copy: {dbg_path} size={os.path.getsize(dbg_path)}')

    # Inspect WAV header to compute duration and properties for debug
    try:
        import io as _io
        import wave as _wave
        wav_buf = _io.BytesIO(audio_bytes)
        with _wave.open(wav_buf, 'rb') as _wf:
            _frames = _wf.getnframes()
            _rate = _wf.getframerate()
            _channels = _wf.getnchannels()
            _sampwidth = _wf.getsampwidth()
            _duration = float(_frames) / float(_rate) if _rate else 0.0
        logger.info(f"[{model}] Synthesized WAV properties: duration={_duration:.3f}s frames={_frames} rate={_rate}Hz channels={_channels} sampwidth={_sampwidth} bytes={len(audio_bytes)}")
    except Exception:
        logger.exception(f'Failed to read synthesized WAV for debugging')

    # First, try to send the synthesized WAV to the tts_mixer service (preferred)
    try:
        resp = requests.post(f"{TTS_MIXER_URL}/play?index={index}", data=audio_bytes, headers={'Content-Type': 'application/octet-stream'}, timeout=10)
        if resp.status_code == 200:
            logger.info(f"[req:{request_id}] Posted WAV to tts_mixer at {TTS_MIXER_URL}/play?index={index}")
            index_str = f"{index:04d}"
            return jsonify({'status': 'ok', 'rtsp_url': f"rtsp://localhost:8554/NB_{index_str}_RX_TTS", 'model': model, 'index': index})
        else:
            logger.warning(f"[req:{request_id}] tts_mixer returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.warning(f"[req:{request_id}] Failed to contact tts_mixer at {TTS_MIXER_URL}: {e}")
        # Fall through to direct push fallback

    # Fallback: Push directly to RTSP using ffmpeg with stdin piping (no temp file). Force codec/sample-rate and capture ffmpeg output.
    # Try push with a single retry to handle transient broken pipes when mediamtx switches publisher
    index_str = f"{index:04d}"
    rtsp_path = f"rtsp://localhost:8554/NB_{index_str}_RX_TTS"
    try:
        ff_cmd = [
            'ffmpeg', '-y', '-f', 'wav', '-i', 'pipe:0',
            '-c:a', 'aac', '-b:a', '64k', '-ar', '48000', '-ac', '1',
            '-loglevel', 'info', '-nostdin',
            '-f', 'rtsp', '-rtsp_transport', 'tcp', rtsp_path
        ]
        logger.info(f"[req:{request_id}] FFMPEG: {' '.join(ff_cmd)} -> {rtsp_path}")
        attempt = 0
        while True:
            try:
                start_time = time.time()
                proc = subprocess.run(ff_cmd, input=audio_bytes, check=True, capture_output=True, timeout=30)
                elapsed = time.time() - start_time
                break
            except subprocess.CalledProcessError as e:
                stderr_text = ''
                try:
                    stderr_text = e.stderr.decode()
                except Exception:
                    stderr_text = str(e)
                logger.warning(f"[req:{request_id}] ffmpeg attempt {attempt+1} failed: {stderr_text}")
                if attempt == 0 and ("Broken pipe" in stderr_text or '-32' in stderr_text):
                    attempt += 1
                    time.sleep(0.2)
                    continue
                raise
        # Log ffmpeg stdout/stderr even on success for full visibility
        try:
            out = proc.stdout.decode(errors='ignore')
            err = proc.stderr.decode(errors='ignore')
        except Exception:
            out = '<unreadable stdout>'
            err = '<unreadable stderr>'
        if out:
            logger.info(f"[req:{request_id}] ffmpeg stdout:\n{out}")
        if err:
            logger.info(f"[req:{request_id}] ffmpeg stderr:\n{err}")
        logger.info(f"[req:{request_id}] Pushed audio to {rtsp_path} (elapsed {elapsed:.3f}s)")
    except subprocess.CalledProcessError as e:
        # ffmpeg returned non-zero
        stderr = None
        try:
            stderr = e.stderr.decode()
        except Exception:
            stderr = str(e)
        logger.error(f"[req:{request_id}] ffmpeg failed (CalledProcessError): {stderr}")
        # No mixer to resume; nothing to do here
        return jsonify({'error': 'failed to push to RTSP', 'details': stderr}), 500
    except Exception as e:
        logger.exception(f"[req:{request_id}] Unexpected error while running ffmpeg")
        # No mixer to resume; nothing to do here
        return jsonify({'error': 'ffmpeg unexpected error', 'details': str(e)}), 500
    finally:
        # remove temp file if it exists
        try:
            if os.path.exists(tmp_wav.name):
                os.unlink(tmp_wav.name)
        except Exception:
            pass

    # No resume action required for mixer-based pipeline

    return jsonify({'status': 'ok', 'rtsp_url': rtsp_path, 'model': model, 'index': index})


@app.route('/models', methods=['GET'])
def list_models():
    """Return list of available TTS models."""
    models_info = []
    for model_key, model_data in tts_models.items():
        config = model_data['config']
        voices = [voice['name'] for voice in config.get('voices', [])]
        models_info.append({
            'id': model_key,
            'name': config['name'],
            'provider': config['provider'],
            'description': config['description'],
            'voices': voices,
            'default_voice': config.get('default_voice')
        })

    return jsonify({
        'models': models_info,
        'available_models': available_models
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=5000, type=int)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    # Load TTS models on startup
    load_tts_models()

    if not available_models:
        logger.error("No TTS models available. Please check your configuration and environment variables.")
        sys.exit(1)

    logger.info(f"Available TTS models: {', '.join(available_models)}")

    # Run server
    logger.info(f"Starting Multi-Model TTS HTTP server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
