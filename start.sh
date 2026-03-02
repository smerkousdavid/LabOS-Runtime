#!/bin/bash
# =============================================================================
# labos-runtime start script
#
# Orchestrates: config generation, image builds, compose generation,
# Docker Compose up, Shinobi registration, and connection info printout.
#
# Idempotent: only rebuilds images and recreates containers when needed.
# Already-running containers are left untouched.
#
# Usage:
#   ./start.sh                          # defaults from .env / architecture.yaml
#   ./start.sh --cameras 2 --tts        # 2 cameras, TTS enabled
#   ./start.sh --no-nvr --no-tts        # disable NVR and TTS
#   ./start.sh --rebuild                # force image rebuild + container recreate
# =============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR"

# Source .env if it exists
[ -f .env ] && export $(grep -v "^#" .env | grep -v "^$" | xargs)

# ===== ARGUMENT PARSING =====
NUM_CAMERAS=${NUM_CAMERAS:-1}
STREAMING_METHOD=${STREAMING_METHOD:-mediamtx}
DEFAULT_FRAMERATE=${DEFAULT_FRAMERATE:-30}
ENABLE_TTS=${ENABLE_TTS:-true}
ENABLE_NVR=${ENABLE_NVR:-true}
FORCE_REBUILD=false

for i in $(seq 1 $#); do
    arg="${!i}"
    case "$arg" in
        --mediamtx)    STREAMING_METHOD="mediamtx" ;;
        --gstreamer)   STREAMING_METHOD="gstreamer" ;;
        --tts)         ENABLE_TTS="true" ;;
        --no-tts)      ENABLE_TTS="false" ;;
        --no-nvr)      ENABLE_NVR="false" ;;
        --rebuild)     FORCE_REBUILD=true ;;
        --cameras)
            next=$((i+1))
            NUM_CAMERAS="${!next}"
            ;;
    esac
done

echo "============================================"
echo "  labos-runtime"
echo "============================================"
echo "  Cameras:          $NUM_CAMERAS"
echo "  Streaming:        $STREAMING_METHOD"
echo "  Framerate:        $DEFAULT_FRAMERATE fps"
echo "  TTS:              $ENABLE_TTS"
echo "  NVR:              $ENABLE_NVR"
echo "  Video mode:       ${VIDEO_MODE:-websocket}"
echo "  Force rebuild:    $FORCE_REBUILD"
echo "============================================"
echo ""

# ===== GENERATE CONFIG =====
echo "Generating configuration files..."
python3 configure.py
echo ""

# ===== CHECK EXISTING STATE =====
RUNNING_COUNT=0
if [ -f compose.yaml ]; then
    RUNNING_COUNT=$(docker compose -f compose.yaml ps --status running -q 2>/dev/null | wc -l)
fi

if [ "$RUNNING_COUNT" -gt 0 ] && [ "$FORCE_REBUILD" = false ]; then
    echo "Found $RUNNING_COUNT running container(s)."
    echo "Skipping image build and full teardown (use --rebuild to force)."
    echo ""
    SKIP_BUILD=true
else
    SKIP_BUILD=false
    # Clean up any stopped/stale containers
    echo "Removing stale containers..."
    docker compose -f compose.yaml down --remove-orphans > /dev/null 2>&1 || true
fi

# ===== BUILD STREAMING IMAGE =====
if [ "$SKIP_BUILD" = false ]; then
    if [ ! -f ./xr_runtime/streaming/Dockerfile ]; then
        echo "Warning: streaming Dockerfile not found, skipping image build"
    else
        echo "Building streaming image (labos_streaming:latest)..."
        docker build -f ./xr_runtime/streaming/Dockerfile -t labos_streaming:latest ./xr_runtime/ || {
            echo "Warning: Streaming image build failed. Continuing anyway."
        }
    fi
fi
export RTSP_IMAGE="${RTSP_IMAGE:-labos_streaming:latest}"

# ===== CREATE LOG DIRECTORIES =====
mkdir -p logs/nat_server logs/dashboard logs/tts_pusher logs/tts_mixer logs/mediamtx
for i in $(seq 1 $NUM_CAMERAS); do
    mkdir -p logs/grpc_$i logs/video_pusher_$i logs/audio_pusher_$i logs/av_merger_$i logs/voice_bridge_$i
done

# ===== GENERATE DOCKER COMPOSE =====
echo "Generating Docker Compose file..."
export NUM_CAMERAS STREAMING_METHOD DEFAULT_FRAMERATE ENABLE_TTS ENABLE_NVR
export MEDIAMTX_SERVICE_NAME="${MEDIAMTX_SERVICE_NAME:-mediamtx}"

pip install jinja2 > /dev/null 2>&1 || true
python3 compose/generate.py "$NUM_CAMERAS" "$STREAMING_METHOD" "$DEFAULT_FRAMERATE" compose.yaml

# ===== START DOCKER COMPOSE =====
# `docker compose up -d` is already idempotent: it only recreates containers
# whose config has changed and starts containers that aren't running.
echo ""
echo "Starting Docker Compose..."
if [ "$FORCE_REBUILD" = true ]; then
    docker compose -f compose.yaml up -d --build --force-recreate
else
    docker compose -f compose.yaml up -d
fi
echo "Waiting for services to initialize..."
sleep 5

# ===== WAIT FOR NAT SERVER =====
echo "Waiting for NAT server..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "  NAT server is healthy"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  Warning: NAT server not responding after 30s"
    fi
    sleep 1
done

# ===== NVR SETUP =====
if [ "$ENABLE_NVR" = "true" ]; then
    echo ""
    echo "Setting up Shinobi NVR..."

    # Generate SQL if the script exists
    if [ -f ./xr_runtime/nvr/mysql-init/generate_create_sql.sh ]; then
        source ./xr_runtime/nvr/mysql-init/generate_create_sql.sh
        generate_create_sql 2>/dev/null || true
    fi

    sleep 5
    python3 ./xr_runtime/nvr/delete_rtsp.py "${API_KEY}" "${GROUP_KEY}" 2>/dev/null || true
    python3 ./xr_runtime/nvr/add_rtsp_to_shinobi.py "$NUM_CAMERAS" "$STREAMING_METHOD" "watch" "5" "$DEFAULT_FRAMERATE" 2>/dev/null || true
fi

# ===== DETECT IP =====
get_wifi_ip() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ip=$(hostname -I | awk '{print $1}')
        [[ -z "$ip" ]] && ip="localhost"
    else
        ip="localhost"
    fi
    echo "$ip"
}
WIFI_IP=$(get_wifi_ip)

# ===== DISPLAY INFO =====
echo ""
echo -e "\033[1;34m================= RTSP Streams =================\033[0m"
for i in $(seq 1 $NUM_CAMERAS); do
    index=$(printf "%04d" $i)
    echo -e "  \033[1;32mCamera $i:\033[0m"
    echo -e "    Merged:  rtsp://$WIFI_IP:8554/NB_${index}_TX_CAM_RGB_MIC_p6S"
    echo -e "    Video:   rtsp://$WIFI_IP:8554/NB_${index}_TX_CAM_RGB"
    echo -e "    Audio:   rtsp://$WIFI_IP:8554/NB_${index}_TX_MIC_p6S"
    if [ "$ENABLE_TTS" = "true" ]; then
        echo -e "    TTS:     rtsp://$WIFI_IP:8554/NB_${index}_RX_TTS"
    fi
done
echo -e "\033[1;34m================================================\033[0m"

echo ""
echo -e "\033[1;33m============================================\033[0m"
echo -e "\033[1;32mEnter this IP on each pair of glasses:\033[0m"
echo -e "\033[1;36m  $WIFI_IP\033[0m"
echo -e "\033[1;33m============================================\033[0m"

echo ""
echo -e "\033[1;34m================= Services =================\033[0m"
echo -e "  Dashboard:     http://localhost:5001"
echo -e "  NAT Server:    ws://localhost:8002/ws"
echo -e "  NAT Health:    http://localhost:8002/health"
if [ "$ENABLE_NVR" = "true" ]; then
    echo -e "  Shinobi NVR:   http://localhost:8088"
    echo -e "    Email:       ${MAIL:-viture@test.com}"
    echo -e "    Password:    ${PASSWORD:-test1234}"
fi
echo -e "  MediaMTX HLS:  http://localhost:8890"
echo -e "  MediaMTX API:  http://localhost:9997"
echo -e "\033[1;34m============================================\033[0m"
echo ""
