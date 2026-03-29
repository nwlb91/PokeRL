#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# PokeRL Training Launcher for Linux / macOS / Cloud VMs
# Starts Pokemon Showdown server, then launches training.
# Press Ctrl+C to stop.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHOWDOWN_DIR="${SCRIPT_DIR}/pokemon-showdown"
SHOWDOWN_PORT="${SHOWDOWN_PORT:-8000}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/training.log}"

# ----------------------------------------------------------
# Step 1: Clone Pokemon Showdown if not present
# ----------------------------------------------------------
if [ ! -f "${SHOWDOWN_DIR}/index.js" ]; then
    if [ -d "${SHOWDOWN_DIR}" ]; then
        echo "[launcher] Incomplete Pokemon Showdown found. Removing and re-cloning..."
        rm -rf "${SHOWDOWN_DIR}"
    fi
    echo "[launcher] Cloning Pokemon Showdown..."
    git clone --depth 1 https://github.com/smogon/pokemon-showdown.git "${SHOWDOWN_DIR}"
    echo "[launcher] Installing Showdown dependencies..."
    cd "${SHOWDOWN_DIR}"
    npm install
    node build
    cd "${SCRIPT_DIR}"
fi

# ----------------------------------------------------------
# Step 2: Create logs directory
# ----------------------------------------------------------
mkdir -p "${SCRIPT_DIR}/logs"

# ----------------------------------------------------------
# Step 3: Start Pokemon Showdown in the background
# ----------------------------------------------------------
echo "[launcher] Starting Pokemon Showdown on port ${SHOWDOWN_PORT}..."
node "${SHOWDOWN_DIR}/index.js" start --no-security --port "${SHOWDOWN_PORT}" &
SHOWDOWN_PID=$!

# Wait for Showdown to be ready
echo "[launcher] Waiting for Showdown to be ready..."
for i in $(seq 1 30); do
    if node -e "const net=require('net');const c=net.connect(${SHOWDOWN_PORT},'localhost',()=>{c.end();process.exit(0)});c.on('error',()=>process.exit(1))" 2>/dev/null; then
        echo "[launcher] Showdown is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[launcher] ERROR: Showdown did not start within 30 seconds."
        exit 1
    fi
    sleep 1
done

# Graceful shutdown
cleanup() {
    echo "[launcher] Shutting down..."
    kill "$SHOWDOWN_PID" 2>/dev/null || true
    wait "$SHOWDOWN_PID" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

# ----------------------------------------------------------
# Step 4: Launch training
# ----------------------------------------------------------
DASHBOARD_PORT="${DASHBOARD_PORT:-5555}"
echo "[launcher] Starting training... (logs: ${LOG_FILE})"
echo "[launcher] Dashboard: http://localhost:${DASHBOARD_PORT}"
echo "[launcher] Press Ctrl+C to stop."
python "${SCRIPT_DIR}/train.py" \
    --headless \
    --log-file "${LOG_FILE}" \
    --server-url localhost \
    --server-port "${SHOWDOWN_PORT}" \
    --checkpoint-dir "${SCRIPT_DIR}/checkpoints" \
    --dashboard \
    --dashboard-port "${DASHBOARD_PORT}" \
    --resume \
    "$@"
