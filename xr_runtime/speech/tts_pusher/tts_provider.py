#!/usr/bin/env python3
"""Unified TTS Provider Interface for multiple TTS services."""

import os
import io
import wave
import base64
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from loguru import logger


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config['name']
        self.provider = config['provider']
        self.sample_rate = config.get('sample_rate', 48000)
        self.channels = config.get('channels', 1)

    @abstractmethod
    def synthesize(self, text: str, voice: str = None, **kwargs) -> bytes:
        """Synthesize text to WAV bytes.

        Args:
            text: Text to synthesize
            voice: Voice name to use (optional, uses default if not provided)
            **kwargs: Additional provider-specific parameters

        Returns:
            WAV audio data as bytes
        """
        pass

    @abstractmethod
    def get_available_voices(self) -> List[Dict]:
        """Return list of available voices for this provider."""
        pass

    def _create_wav_from_pcm(self, pcm_data: bytes, sample_rate: int = None, channels: int = None) -> bytes:
        """Create WAV file from raw PCM data."""
        if sample_rate is None:
            sample_rate = self.sample_rate
        if channels is None:
            channels = self.channels

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        return wav_buffer.getvalue()

    def _normalize_audio_format(self, wav_bytes: bytes, target_sample_rate: int = 48000, target_channels: int = 1) -> bytes:
        """Normalize audio to target format using ffmpeg."""
        if self.sample_rate == target_sample_rate and self.channels == target_channels:
            logger.info(f"Audio already at target format ({target_sample_rate}Hz/{target_channels}ch), skipping normalization")
            return wav_bytes

        logger.info(f"Normalizing audio from {self.sample_rate}Hz/{self.channels}ch to {target_sample_rate}Hz/{target_channels}ch")
        
        # Use ffmpeg to convert sample rate/channels
        cmd = [
            'ffmpeg', '-f', 'wav', '-i', 'pipe:0',
            '-ar', str(target_sample_rate), '-ac', str(target_channels),
            '-f', 'wav', '-hide_banner', '-loglevel', 'error', 'pipe:1'
        ]

        try:
            proc = subprocess.run(cmd, input=wav_bytes, capture_output=True, check=True, timeout=10)
            output = proc.stdout
            if not output or len(output) < 44:  # WAV header is at least 44 bytes
                logger.error(f"ffmpeg produced invalid output: {len(output)} bytes")
                return wav_bytes
            logger.info(f"Normalization successful: {len(wav_bytes)} bytes → {len(output)} bytes")
            return output
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio normalization failed: ffmpeg returned {e.returncode}")
            logger.error(f"stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
            return wav_bytes  # Return original if conversion fails
        except subprocess.TimeoutExpired:
            logger.error(f"Audio normalization timeout (ffmpeg exceeded 10s)")
            return wav_bytes
        except Exception as e:
            logger.error(f"Audio normalization error: {e}")
            return wav_bytes


class RivaProvider(TTSProvider):
    """Riva TTS Provider using gRPC."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import riva.client
        endpoint = os.environ.get(config.get('endpoint_env'), config.get('default_endpoint'))
        self.auth = riva.client.Auth(None, False, endpoint, None)
        self.service = riva.client.SpeechSynthesisService(self.auth)

    def synthesize(self, text: str, voice: str = None, **kwargs) -> bytes:
        """Synthesize text using Riva."""
        if voice is None:
            voice = self.config.get('default_voice')

        # Extract language from voice config
        voice_config = next((v for v in self.config['voices'] if v['name'] == voice), None)
        language = voice_config['language'] if voice_config else 'en-US'

        try:
            resp = self.service.synthesize(
                text=text,
                language_code=language,
                voice_name=voice
            )
            # Convert PCM to WAV
            wav_bytes = self._create_wav_from_pcm(resp.audio)
            # Normalize to standard format
            return self._normalize_audio_format(wav_bytes)
        except Exception as e:
            logger.error(f"Riva synthesis failed: {e}")
            raise

    def get_available_voices(self) -> List[Dict]:
        """Return available voices."""
        return self.config['voices']


class QwenProvider(TTSProvider):
    """Qwen TTS Provider using DashScope API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import dashscope
        api_key = os.environ.get(config.get('api_key_env'))
        if not api_key:
            raise ValueError(f"API key not found in environment variable {config.get('api_key_env')}")

        dashscope.api_key = api_key
        if config.get('base_url'):
            dashscope.base_http_api_url = config['base_url']
        self._dashscope = dashscope

    def synthesize(self, text: str, voice: str = None, **kwargs) -> bytes:
        """Synthesize text using Qwen TTS."""
        if voice is None:
            voice = self.config.get('default_voice')

        try:
            response = self._dashscope.MultiModalConversation.call(
                model="qwen3-tts-flash",
                text=text,
                voice=voice,
                language_type=self.config.get('language_type', 'Chinese'),
                stream=True
            )

            # Collect all audio chunks
            audio_chunks = []
            chunk_count = 0
            for chunk in response:
                if chunk.output is not None and chunk.output.audio is not None:
                    if chunk.output.audio.data is not None:
                        # Decode base64 audio data
                        audio_data = base64.b64decode(chunk.output.audio.data)
                        audio_chunks.append(audio_data)
                        chunk_count += 1
                        logger.debug(f"QwenProvider: Received chunk {chunk_count}, size={len(audio_data)} bytes")

            logger.info(f"QwenProvider: Received {chunk_count} audio chunks from Qwen API")
            
            if not audio_chunks:
                raise ValueError("No audio data received from Qwen TTS")

            # Concatenate all chunks
            combined_audio = b''.join(audio_chunks)
            logger.info(f"QwenProvider: Total PCM data: {len(combined_audio)} bytes")

            # Qwen returns raw PCM at the configured sample rate (24000Hz)
            # Always create WAV with the correct sample rate from config, then normalize to target
            logger.info(f"QwenProvider: Creating WAV from raw PCM (sample_rate={self.sample_rate}Hz, channels={self.channels}, pcm_bytes={len(combined_audio)})")
            wav_bytes = self._create_wav_from_pcm(combined_audio, sample_rate=self.sample_rate, channels=self.channels)
            logger.info(f"QwenProvider: Created WAV: {len(wav_bytes)} bytes, header: {wav_bytes[:4] if len(wav_bytes) >= 4 else 'N/A'}")
            
            # Always normalize to 48000Hz/1ch for consistency with Riva and RTSP requirements
            # This ensures tts_mixer receives properly formatted audio
            normalized = self._normalize_audio_format(wav_bytes, target_sample_rate=48000, target_channels=1)
            
            logger.info(f"QwenProvider: Normalized audio to 48000Hz/1ch (size={len(normalized)} bytes)")
            return normalized

        except Exception as e:
            # Detect DNS/NameResolution and connection problems and provide a helpful message
            err_msg = str(e)
            if 'Failed to resolve' in err_msg or 'Temporary failure in name resolution' in err_msg or 'Name or service not known' in err_msg:
                raise ConnectionError(
                    "Network/DNS resolution error when contacting DashScope API. "
                    "Check container outbound DNS and network connectivity, and that the host can resolve 'dashscope.aliyuncs.com'."
                ) from e
            raise

    def get_available_voices(self) -> List[Dict]:
        """Return available voices."""
        return self.config['voices']


class VibevoiceProvider(TTSProvider):
    """VibeVoice TTS Provider using HTTP."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import httpx
        host = config.get('host', 'tts')
        port = config.get('port', 8050)
        self._base_url = f"http://{host}:{port}"
        self._client = httpx.Client(base_url=self._base_url, timeout=60.0)

    def synthesize(self, text: str, voice: str = None, **kwargs) -> bytes:
        if voice is None:
            voice = self.config.get('default_voice', 'en-Emma_woman')
        try:
            resp = self._client.post("/synthesize", json={"text": text, "speaker_name": voice})
            resp.raise_for_status()
            wav_bytes = resp.content
            return self._normalize_audio_format(wav_bytes, target_sample_rate=48000, target_channels=1)
        except Exception as e:
            logger.error(f"VibeVoice synthesis failed: {e}")
            raise

    def get_available_voices(self) -> List[Dict]:
        return self.config.get('voices', [])


class TTSProviderFactory:
    """Factory for creating TTS providers."""

    @staticmethod
    def create_provider(model_config: Dict[str, Any]) -> TTSProvider:
        """Create a TTS provider instance based on configuration."""
        provider_type = model_config.get('type')
        provider_name = model_config.get('provider', '')

        if provider_type == 'grpc':
            return RivaProvider(model_config)
        elif provider_name == 'vibevoice':
            return VibevoiceProvider(model_config)
        elif provider_type == 'http':
            return QwenProvider(model_config)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    @staticmethod
    def is_enabled(model_config: Dict[str, Any]) -> bool:
        """Check if a model is enabled.

        Priority: enabled_env (environment variable gate) > enabled (static boolean).
        """
        enabled_env = model_config.get('enabled_env')
        if enabled_env:
            return os.environ.get(enabled_env, '').lower() in ('true', '1', 'yes')
        enabled = model_config.get('enabled')
        if enabled is not None:
            return bool(enabled)
        return False