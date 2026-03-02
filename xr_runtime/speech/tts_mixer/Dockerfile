FROM python:3.11.14-slim-bookworm

WORKDIR /app
COPY tts_mixer.py /app/
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir flask loguru
CMD ["python3", "/app/tts_mixer.py"]
