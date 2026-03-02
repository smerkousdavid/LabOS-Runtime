FROM python:3.11.14-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY voice_bridge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY xr_service_library/xr_service_library-0.1-py3-none-any.whl /tmp/
RUN pip install --no-cache-dir /tmp/xr_service_library-0.1-py3-none-any.whl \
    && rm /tmp/xr_service_library-0.1-py3-none-any.whl

COPY voice_bridge/ /app/

CMD ["python", "bridge.py"]
