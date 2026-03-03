FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps for OpenCV headless, RealSense USB, and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the robot source tree (aira package, configs, protocols, locations, etc.)
COPY aira/ /app/aira/
COPY configs/ /app/configs/
COPY protocols/ /app/protocols/
COPY locations/ /app/locations/
COPY robot_runtime.py /app/
COPY mcp_server.py /app/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "robot_runtime.py"]
