#!/usr/bin/env bash
# LabOS XR Runtime -- One-time installer (Linux / macOS)
set -e

cd "$(dirname "$0")"
ROOT="$(pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; exit 1; }

echo ""
echo "========================================="
echo "  LabOS XR Runtime -- Installer"
echo "========================================="
echo ""

# ── Prerequisites ─────────────────────────────────────────────────────────────

echo "Checking prerequisites ..."

command -v git >/dev/null 2>&1 || fail "git is not installed. Install it from https://git-scm.com"
ok "git"

command -v docker >/dev/null 2>&1 || fail "docker is not installed. Install Docker Desktop from https://docker.com"
ok "docker"

docker compose version >/dev/null 2>&1 || fail "'docker compose' plugin not found. Update Docker Desktop or install compose v2."
ok "docker compose"

PYTHON=""
for p in python3 python; do
    if command -v "$p" >/dev/null 2>&1; then
        ver=$("$p" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$("$p" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        minor=$("$p" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$p"
            break
        fi
    fi
done
[ -n "$PYTHON" ] || fail "Python >= 3.10 is required. Install from https://python.org"
ok "Python $ver ($PYTHON)"

echo ""

# ── Virtual environment ───────────────────────────────────────────────────────

echo "Setting up Python virtual environment ..."
if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
    ok "Created .venv/"
else
    ok ".venv/ already exists"
fi

source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Installed Python dependencies"

echo ""

# ── Config files ──────────────────────────────────────────────────────────────

echo "Setting up configuration files ..."

if [ ! -f "config/.env.secrets" ]; then
    if [ -f "config/.env.secrets.example" ]; then
        cp config/.env.secrets.example config/.env.secrets
        warn "Created config/.env.secrets -- please fill in your API keys"
    else
        touch config/.env.secrets
        warn "Created empty config/.env.secrets"
    fi
else
    ok "config/.env.secrets exists"
fi

if [ ! -f "config/config.yaml" ]; then
    if [ -f "config/config.yaml.example" ]; then
        cp config/config.yaml.example config/config.yaml
        warn "Created config/config.yaml -- please review and edit"
    fi
else
    ok "config/config.yaml exists"
fi

echo ""

# ── Docker image build ────────────────────────────────────────────────────────

echo "Building Docker images ..."

if [ -f "./xr_runtime/streaming/Dockerfile" ]; then
    docker build -f ./xr_runtime/streaming/Dockerfile -t labos_streaming:latest ./xr_runtime/ 2>&1 | tail -1
    ok "Built labos_streaming image"
else
    warn "Streaming Dockerfile not found -- skipping image build"
fi

echo ""

# ── Done ──────────────────────────────────────────────────────────────────────

echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit config/config.yaml   (set NAT server URL, STT/TTS endpoints)"
echo "  2. Edit config/.env.secrets   (add API keys if needed)"
echo "  3. Run: ./run.sh              (start the runtime)"
echo "  4. Optional: ./update_glasses.sh  (configure glasses USB)"
echo ""
