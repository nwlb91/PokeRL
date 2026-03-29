#!/usr/bin/env bash
set -euo pipefail

SHOWDOWN_PORT="${SHOWDOWN_PORT:-8000}"
LOG_FILE="${LOG_FILE:-/app/logs/training.log}"

echo "[entrypoint] Starting Pokemon Showdown on port ${SHOWDOWN_PORT}..."
node /app/pokemon-showdown/pokemon-showdown start --no-security --skip-build --port="${SHOWDOWN_PORT}" &
SHOWDOWN_PID=$!

# Wait for Showdown to accept connections
echo "[entrypoint] Waiting for Showdown to be ready..."
for i in $(seq 1 30); do
    if node -e "const net=require('net');const c=net.connect(${SHOWDOWN_PORT},'localhost',()=>{c.end();process.exit(0)});c.on('error',()=>process.exit(1))" 2>/dev/null; then
        echo "[entrypoint] Showdown is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[entrypoint] ERROR: Showdown did not start within 30 seconds."
        exit 1
    fi
    sleep 1
done

# Graceful shutdown
cleanup() {
    echo "[entrypoint] Shutting down..."
    kill "$SHOWDOWN_PID" 2>/dev/null || true
    wait "$SHOWDOWN_PID" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

DASHBOARD_PORT="${DASHBOARD_PORT:-5555}"

# Launch training, forwarding any extra arguments
mkdir -p /app/logs
echo "[entrypoint] Starting training..."
echo "[entrypoint] Dashboard: http://localhost:${DASHBOARD_PORT}"
exec python train.py \
    --headless \
    --log-file "${LOG_FILE}" \
    --server-url localhost \
    --server-port "${SHOWDOWN_PORT}" \
    --checkpoint-dir /app/checkpoints \
    --dashboard \
    --dashboard-port "${DASHBOARD_PORT}" \
    --resume \
    "$@"
