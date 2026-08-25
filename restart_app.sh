#!/bin/bash
# ==============================================================================
# AI Hindi Cinema Studio - 1-Click App & GPU Engine Restart Script
# Safely restarts FastAPI Backend, ComfyUI GPU Server, and Cloudflare Tunnel
# ==============================================================================

set -e

echo "================================================================="
echo "🔄 Restarting AI Hindi Cinema Studio & GPU Engines..."
echo "================================================================="

# 1. Kill old processes on Ports 8000, 8188 and old tunnels
echo "🛑 Stopping existing processes..."
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8188/tcp 2>/dev/null || true
pkill -f cloudflared 2>/dev/null || true
pkill -f uvicorn 2>/dev/null || true

sleep 2

# 2. Re-launch Server and Tunnel
echo "🚀 Starting fresh instance..."
chmod +x start_server_tunnel.sh
./start_server_tunnel.sh
