FROM python:3.11.14-slim-bookworm

RUN apt-get update && apt-get install -y ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-bad1.0-dev \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-rtsp \
        python3-gi \
        python3-gst-1.0 \
        python3-dev \
        libgirepository1.0-dev \
        libcairo2-dev \
        pkg-config \
        gir1.2-gtk-3.0 \
        libgstrtspserver-1.0-dev \
        gir1.2-gst-rtsp-server-1.0 \
        curl \
        build-essential \
        git \
        && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools wheel && pip install poetry

ENV PATH="/root/.local/bin:$PATH"

# Build context is ./xr_runtime/ -- mirror the directory structure so
# the relative path in pyproject.toml (../xr_service_library/) resolves.
COPY xr_service_library/ /home/xr_service_library/
COPY streaming/ /home/streaming/

WORKDIR /home/streaming

RUN poetry lock
RUN poetry install

EXPOSE 8554
