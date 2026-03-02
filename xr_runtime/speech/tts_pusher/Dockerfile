FROM python:3.11.14-slim-bookworm

WORKDIR /app
COPY tts_pusher.py /app/
COPY tts_provider.py /app/
COPY tts_models.yaml /app/
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg gcc libsndfile1-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir flask requests loguru pyyaml httpx elevenlabs && \
    pip install --no-cache-dir nvidia-riva-client dashscope 2>/dev/null || true
CMD ["python3", "/app/tts_pusher.py"]
