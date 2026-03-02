#!/usr/bin/env python3
"""
Message API Server

A Flask-based REST API server that provides endpoints for sending messages
to the XR Service via Unix domain socket connection.
"""

import sys
import os
from loguru import logger
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests

# Configure Loguru FIRST, before any imports
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    filter=lambda record: not (
        record["name"].startswith("xr_service_library") and
        record["level"].name == "WARNING"
    )
)
_log_dir = os.environ.get("LOG_DIR", "/app/logs")
if os.path.isdir(_log_dir):
    logger.add(os.path.join(_log_dir, "dashboard.log"), rotation="20 MB", retention="3 days", level="DEBUG")

# Try to import from installed package first, then from local path
try:
    from xr_service_library import XRServiceConnection
    from xr_service_library.xr_types import Message, AudioSample, Blob
except ImportError:
    # Fallback for local development
    sys.path.append('../../')
    from xr_service_library import XRServiceConnection
    from xr_service_library.xr_types import Message, AudioSample, Blob

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Global configuration
CAMERA_PORTS = {}  # Will be set at runtime
NUM_CAMERAS = 0    # Will be set at runtime

class MessageAPIServer:
    def __init__(self, socket_path='/tmp/xr_service.sock'):
        """Initialize Message API Server"""
        self.socket_path = socket_path
        self.client = None
        
    def connect_to_xr_service(self):
        """Connect to XR Service"""
        try:
            if self.client is None:
                self.client = XRServiceConnection(socket_path=self.socket_path)
                self.client.connect()
                logger.info(f"Connected to XR Service at {self.socket_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to XR Service: {e}")
            self.client = None
            return False
        return True
    
    def send_message(self, message_type, payload):
        """Send message to XR Service"""
        if not self.connect_to_xr_service():
            return False, "Failed to connect to XR Service"
        
        try:
            msg = Message(message_type=message_type, payload=payload)
            self.client.send_message(msg)
            logger.info(f"Sent message: type={message_type}, payload={payload}")
            return True, "Message sent successfully"
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False, f"Failed to send message: {str(e)}"
    
    def send_audio(self, audio_data, sample_rate):
        """Send audio sample to XR Service (single chunk)
        
        Note: This uses schedule_audio_transmission() which requires the GRPC server
        to actively poll get_incoming_audio_samples() and stream to clients.
        For large audio files, use send_audio_streaming() instead.
        """
        if not self.connect_to_xr_service():
            return False, "Failed to connect to XR Service"
        
        try:
            # Limit single chunk size to prevent buffer overflow (max 1MB)
            max_chunk_size = 1024 * 1024  # 1MB
            if len(audio_data) > max_chunk_size:
                logger.warning(f"Audio data too large ({len(audio_data)} bytes), truncating to {max_chunk_size} bytes. Consider using streaming mode.")
                audio_data = audio_data[:max_chunk_size]
            
            audio_sample = AudioSample(audio_data=audio_data, sample_rate=float(sample_rate))
            result = self.client.schedule_audio_transmission(audio_sample)
            
            if result:
                logger.info(f"Sent audio via schedule_audio_transmission: {len(audio_data)} bytes at {sample_rate} Hz")
                return True, f"Audio scheduled successfully ({len(audio_data)} bytes at {sample_rate} Hz)"
            else:
                logger.error(f"Failed to schedule audio transmission")
                return False, "Failed to schedule audio transmission"
                
        except Exception as e:
            logger.error(f"Failed to send audio: {e}")
            return False, f"Failed to send audio: {str(e)}"
    
    def send_audio_as_blob(self, audio_data, sample_rate):
        """Send audio as a blob message (alternative method)
        
        This sends audio as a binary blob which may be handled differently by the GRPC server.
        """
        if not self.connect_to_xr_service():
            return False, "Failed to connect to XR Service"
        
        try:
            # Create a blob with audio data and metadata in the type field
            blob = Blob(message_type=f"audio:{sample_rate}", payload=audio_data)
            self.client.send_blob(blob)
            logger.info(f"Sent audio as blob: {len(audio_data)} bytes at {sample_rate} Hz")
            return True, f"Audio blob sent successfully ({len(audio_data)} bytes at {sample_rate} Hz)"
        except Exception as e:
            logger.error(f"Failed to send audio blob: {e}")
            return False, f"Failed to send audio blob: {str(e)}"
    
    def send_audio_streaming(self, audio_data, sample_rate, chunk_duration_ms=100):
        """Send audio in streaming chunks (like TTS)
        
        This mimics TTS streaming by sending audio in small chunks with proper timing.
        Chunk size is calculated based on sample rate and duration.
        
        Args:
            audio_data: Raw audio bytes (PCM 16-bit)
            sample_rate: Sample rate in Hz
            chunk_duration_ms: Duration of each chunk in milliseconds (default 100ms)
        """
        if not self.connect_to_xr_service():
            return False, "Failed to connect to XR Service"
        
        try:
            import time
            
            # Calculate chunk size: sample_rate * (duration_ms / 1000) * 2 bytes per sample (16-bit PCM)
            bytes_per_sample = 2  # 16-bit PCM = 2 bytes
            chunk_size = int(sample_rate * (chunk_duration_ms / 1000.0) * bytes_per_sample)
            
            # Ensure chunk size is even (for 16-bit samples)
            if chunk_size % 2 != 0:
                chunk_size += 1
            
            total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
            logger.info(f"Streaming audio: {len(audio_data)} bytes at {sample_rate} Hz in {total_chunks} chunks of {chunk_size} bytes ({chunk_duration_ms}ms each)")
            
            # Send audio in chunks with timing control
            chunk_count = 0
            failed_chunks = 0
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                chunk_count += 1
                
                try:
                    # Send chunk as AudioSample
                    audio_sample = AudioSample(audio_data=chunk, sample_rate=float(sample_rate))
                    result = self.client.schedule_audio_transmission(audio_sample)
                    
                    if not result:
                        failed_chunks += 1
                        logger.warning(f"Failed to send chunk {chunk_count}/{total_chunks}")
                    
                    # Add delay to prevent overwhelming the buffer
                    # Sleep for a fraction of the chunk duration to allow processing
                    time.sleep(chunk_duration_ms / 1000.0 * 0.5)  # Sleep for 50% of chunk duration
                    
                    if chunk_count % 10 == 0:  # Log every 10 chunks
                        logger.info(f"Sent chunk {chunk_count}/{total_chunks}")
                        
                except Exception as chunk_error:
                    failed_chunks += 1
                    logger.error(f"Error sending chunk {chunk_count}/{total_chunks}: {chunk_error}")
                    # Continue with next chunk instead of failing completely
            
            if failed_chunks > 0:
                logger.warning(f"Streaming completed with {failed_chunks} failed chunks out of {total_chunks}")
                return True, f"Audio streamed with warnings ({total_chunks - failed_chunks}/{total_chunks} chunks successful, {len(audio_data)} bytes at {sample_rate} Hz)"
            else:
                logger.info(f"Streaming complete: sent {total_chunks} chunks successfully")
                return True, f"Audio streamed successfully ({total_chunks} chunks, {len(audio_data)} bytes at {sample_rate} Hz)"
            
        except Exception as e:
            logger.error(f"Failed to stream audio: {e}")
            return False, f"Failed to stream audio: {str(e)}"

# Global message server instance
message_server = MessageAPIServer()

@app.route('/')
def index():
    """Serve the web UI"""
    return render_template('index.html')

@app.route('/api/send_message', methods=['POST'])
def api_send_message():
    """API endpoint to send a message"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        message_type = data.get('message_type')
        payload = data.get('payload')
        
        if not message_type:
            return jsonify({'error': 'message_type is required'}), 400
        
        if not payload:
            return jsonify({'error': 'payload is required'}), 400
        
        success, result = message_server.send_message(message_type, payload)
        
        if success:
            return jsonify({
                'success': True,
                'message': result,
                'data': {
                    'message_type': message_type,
                    'payload': payload
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result
            }), 500
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/send_audio', methods=['POST'])
def api_send_audio():
    """API endpoint to send audio via schedule_audio_transmission"""
    try:
        # Check if audio file is in the request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Get sample rate and method from form data
        user_sample_rate = float(request.form.get('sample_rate', 16000))
        method = request.form.get('method', 'audio')
        
        # Read audio data
        audio_data = audio_file.read()
        
        if not audio_data:
            return jsonify({'error': 'Audio file is empty'}), 400
        
        # Auto-detect WAV file format and extract PCM data
        actual_sample_rate = user_sample_rate
        pcm_data = audio_data
        num_channels = 1
        
        if audio_file.filename.lower().endswith('.wav'):
            try:
                import struct
                
                # Parse WAV header
                if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                    # Find fmt chunk to get audio format info
                    pos = 12
                    fmt_data = None
                    
                    while pos < len(audio_data) - 8:
                        chunk_id = audio_data[pos:pos+4]
                        chunk_size = struct.unpack('<I', audio_data[pos+4:pos+8])[0]
                        
                        if chunk_id == b'fmt ':
                            fmt_data = audio_data[pos+8:pos+8+chunk_size]
                        elif chunk_id == b'data':
                            # Extract PCM data (skip WAV header)
                            pcm_data = audio_data[pos+8:pos+8+chunk_size]
                            break
                        
                        pos += 8 + chunk_size
                    
                    if fmt_data and len(fmt_data) >= 16:
                        # Parse format chunk
                        num_channels = struct.unpack('<H', fmt_data[2:4])[0]
                        actual_sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                        bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
                        
                        logger.info(f"Detected WAV file: {actual_sample_rate} Hz, {num_channels} channel(s), {bits_per_sample}-bit, PCM data size: {len(pcm_data)} bytes")
                        
                        # Convert stereo to mono if needed
                        if num_channels == 2:
                            logger.warning(f"Converting stereo to mono (averaging channels)")
                            # Convert stereo to mono by averaging left and right channels
                            import array
                            stereo_samples = array.array('h', pcm_data)  # 16-bit signed integers
                            mono_samples = array.array('h')
                            
                            for i in range(0, len(stereo_samples), 2):
                                if i + 1 < len(stereo_samples):
                                    # Average left and right channels
                                    avg = (stereo_samples[i] + stereo_samples[i + 1]) // 2
                                    mono_samples.append(avg)
                            
                            pcm_data = mono_samples.tobytes()
                            logger.info(f"Converted to mono: {len(pcm_data)} bytes")
                        
                        # If user specified different sample rate, warn them
                        if abs(actual_sample_rate - user_sample_rate) > 100:
                            logger.warning(f"Sample rate mismatch! WAV file: {actual_sample_rate} Hz, User specified: {user_sample_rate} Hz. Using WAV file rate.")
                        
            except Exception as e:
                logger.error(f"Failed to parse WAV file: {e}, using raw data")
                actual_sample_rate = user_sample_rate
                pcm_data = audio_data
        
        # Use the detected sample rate
        sample_rate = actual_sample_rate
        
        # Choose method based on parameter
        if method == 'blob':
            success, result = message_server.send_audio_as_blob(pcm_data, sample_rate)
        elif method == 'streaming':
            # Get chunk duration from form (default 100ms)
            chunk_duration = int(request.form.get('chunk_duration', 100))
            success, result = message_server.send_audio_streaming(pcm_data, sample_rate, chunk_duration)
        else:
            success, result = message_server.send_audio(pcm_data, sample_rate)
        
        if success:
            return jsonify({
                'success': True,
                'message': result,
                'data': {
                    'audio_size': len(pcm_data),
                    'sample_rate': sample_rate,
                    'detected_rate': actual_sample_rate,
                    'user_specified_rate': user_sample_rate,
                    'filename': audio_file.filename,
                    'method': method
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result
            }), 500
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint to check server status"""
    try:
        connected = message_server.connect_to_xr_service()
        return jsonify({
            'status': 'running',
            'xr_service_connected': connected,
            'socket_path': message_server.socket_path
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/cameras', methods=['GET'])
def api_cameras():
    """API endpoint to get available devices"""
    try:
        cameras = []
        for camera_id in range(1, NUM_CAMERAS + 1):
            index_str = f"{camera_id:04d}"
            app_port = 5050 + (camera_id - 1)
            cameras.append({
                'id': camera_id,
                'port': CAMERA_PORTS.get(camera_id, 5000 + camera_id),
                'appPort': app_port,
                'streams': {
                    'merged': f'rtsp://localhost:8554/NB_{index_str}_TX_CAM_RGB_MIC_p6S',
                    'video': f'rtsp://localhost:8554/NB_{index_str}_TX_CAM_RGB',
                    'audio': f'rtsp://localhost:8554/NB_{index_str}_TX_MIC_p6S',
                    'tts': f'rtsp://localhost:8554/NB_{index_str}_RX_TTS'
                }
            })
        return jsonify({
            'success': True,
            'cameras': cameras,
            'total': NUM_CAMERAS
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tts_models', methods=['GET'])
def api_tts_models():
    """API endpoint to get available TTS models"""
    try:
        # Fetch TTS models from the TTS pusher service
        tts_url = "http://tts-pusher:5000/models"
        response = requests.get(tts_url, timeout=5)
        response.raise_for_status()
        
        models_data = response.json()
        return jsonify({
            'success': True,
            'models': models_data.get('models', []),
            'available_models': models_data.get('available_models', [])
        })
    except requests.RequestException as e:
        logger.error(f"Failed to fetch TTS models: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch TTS models: {str(e)}'
        }), 500
    except Exception as e:
        logger.error(f"Error fetching TTS models: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/send_tts', methods=['POST'])
def api_send_tts():
    """API endpoint to send TTS audio"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text')
        model = data.get('model', os.environ.get('TTS_MODEL', 'vibevoice'))
        language = data.get('language', 'en')
        voice = data.get('voice', 'default')
        camera_id = data.get('camera_id', 1)
        
        if not text:
            return jsonify({'error': 'text is required'}), 400
        
        # Log TTS request
        logger.info(f"TTS request: text='{text[:50]}...', model={model}, language={language}, voice={voice}, device={camera_id}")
        
        # Send as message to XR Service indicating TTS generation
        import json
        import requests
        
        # Trigger TTS generation via TTS Pusher service
        try:
            tts_url = "http://tts-pusher:5000/synthesize"
            # Map 'default' voice to None or specific voice if needed
            voice_param = voice if voice != 'default' else None
            
            requests.post(tts_url, json={
                "text": text,
                "model": model,
                "index": camera_id,
                "voice": voice_param,
                "language": language
            }, timeout=5)
            logger.info(f"Triggered TTS generation at {tts_url}")
        except Exception as e:
            logger.error(f"Failed to trigger TTS generation: {e}")
            # We continue even if TTS generation fails, as the message to XR service might be enough for some use cases
            # or we could return an error. For now, we log it.

        success, result = message_server.send_message(
            message_type='tts',
            payload=json.dumps({
                'text': text,
                'model': model,
                'language': language,
                'voice': voice,
                'camera_id': camera_id
            })
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': f'TTS generated successfully: {text[:50]}...',
                'data': {
                    'text': text,
                    'model': model,
                    'language': language,
                    'voice': voice,
                    'camera_id': camera_id
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result
            }), 500
            
    except Exception as e:
        logger.error(f"API error in TTS: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


# ---------------------------------------------------------------------------
# New endpoints: Frame capture, Agent proxy
# ---------------------------------------------------------------------------

NAT_HTTP_URL = os.environ.get('NAT_HTTP_URL', 'http://nat-server:8002')
NAT_WS_URL = os.environ.get('NAT_WS_URL', 'ws://nat-server:8002/ws')

@app.route('/api/frames/<int:camera_id>', methods=['GET'])
def api_frames(camera_id):
    """Capture frames from XR socket for VLM or debugging."""
    try:
        import base64
        import numpy as np

        count = int(request.args.get('count', 1))
        quality = int(request.args.get('quality', 70))
        count = min(count, 16)

        if not message_server.connect_to_xr_service():
            return jsonify({'success': False, 'error': 'XR service not connected'}), 503

        frames = []
        for _ in range(count):
            try:
                frame = message_server.client.get_latest_frame()
                if frame is not None:
                    import cv2
                    _, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    b64 = base64.b64encode(jpeg_buf.tobytes()).decode('ascii')
                    frames.append(b64)
            except Exception as e:
                logger.warning(f"Frame capture error: {e}")

        return jsonify({'success': True, 'frames': frames, 'count': len(frames)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/agent/chat', methods=['POST'])
def api_agent_chat():
    """Send text to NAT agent via a temporary WebSocket connection."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'text is required'}), 400

        text = data['text']
        session_id = data.get('session_id', 'dashboard-1')

        import asyncio
        import websockets
        import json as json_mod

        async def _chat():
            uri = f"{NAT_WS_URL}?session_id={session_id}"
            async with websockets.connect(uri, close_timeout=2) as ws:
                await ws.send(json_mod.dumps({'type': 'user_message', 'text': text}))
                raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
                return json_mod.loads(raw)

        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(_chat())
        finally:
            loop.close()

        return jsonify({'success': True, 'response': resp.get('text', ''), 'raw': resp})
    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/agent/tools', methods=['GET'])
def api_agent_tools_get():
    """Proxy to NAT GET /tools/catalog."""
    try:
        resp = requests.get(f"{NAT_HTTP_URL}/tools/catalog", timeout=5)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 503


@app.route('/api/agent/tools', methods=['PUT'])
def api_agent_tools_put():
    """Proxy to NAT PUT /tools/catalog."""
    try:
        data = request.get_json()
        resp = requests.put(f"{NAT_HTTP_URL}/tools/catalog", json=data, timeout=5)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 503


@app.route('/api/agent/status', methods=['GET'])
def api_agent_status():
    """Check NAT server health."""
    try:
        resp = requests.get(f"{NAT_HTTP_URL}/health", timeout=3)
        return jsonify({'connected': True, 'nat_url': NAT_HTTP_URL, 'nat_health': resp.json()})
    except Exception:
        return jsonify({'connected': False, 'nat_url': NAT_HTTP_URL})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Message API Server')
    parser.add_argument('--socket-path', type=str, default='/tmp/xr_service.sock',
                        help='Path to the Unix domain socket for the XR Service')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to bind the server to')
    parser.add_argument('--num-cameras', type=int, default=1,
                        help='Total number of cameras available')
    
    args = parser.parse_args()
    
    # Set global configuration
    NUM_CAMERAS = args.num_cameras
    for i in range(1, NUM_CAMERAS + 1):
        CAMERA_PORTS[i] = 5000 + i
    
    # Initialize the message server with the socket path
    message_server.socket_path = args.socket_path
    
    logger.info(f"Starting Message API Server on {args.host}:{args.port}")
    logger.info(f"XR Service socket path: {args.socket_path}")
    logger.info(f"Total cameras: {NUM_CAMERAS}")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
