FROM python:3.11.14-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=60

# Copy requirements and install Python dependencies
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install XR service library from wheel (available in build context via docker-compose)
COPY xr_service_library/xr_service_library-0.1-py3-none-any.whl .
RUN pip install --no-cache-dir ./xr_service_library-0.1-py3-none-any.whl \
    && rm -f xr_service_library-0.1-py3-none-any.whl

# Copy the dashboard server and template/static directories
COPY dashboard/server.py .
COPY dashboard/templates ./templates
COPY dashboard/static ./static

# Expose the port
EXPOSE 5000

# Run the dashboard server
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "5000"]
