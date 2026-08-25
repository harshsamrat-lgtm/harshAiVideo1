#!/bin/bash
# ==============================================================================
# AI Hindi Cinema Studio - Combined Server & Cloudflare Tunnel Launcher
# Solves 502 Bad Gateway by ensuring the Python server is running before tunneling
# ==============================================================================

set -e

echo "================================================================="
echo "🎬 Starting AI Hindi Cinema Studio + Public Web Tunnel"
echo "================================================================="

# 1. Install dependencies if needed
pip install -r backend/requirements.txt || pip install fastapi uvicorn pillow pydantic

# 2. Kill any old process on port 8000
fuser -k 8000/tcp || true

# 3. Start Studio FastAPI backend in background
echo "🚀 Starting Studio Backend on port 8000..."
cd backend
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > ../server.log 2>&1 &
cd ..

# 4. Wait for backend to be ready
echo "⏳ Waiting for backend to initialize..."
sleep 3

# 5. Check if backend is alive
if curl -s http://127.0.0.1:8000/api/system/status > /dev/null; then
    echo "✅ Backend is healthy and running on port 8000!"
else
    echo "⚠️ Backend starting up, checking log:"
    cat server.log || true
fi

# 6. Install cloudflared if not present
if ! command -v cloudflared &> /dev/null; then
    echo "📥 Installing Cloudflared..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi

# 7. Start Cloudflare Tunnel
echo "================================================================="
echo "🌐 Launching Public Web URL... (Click the https link below 👇)"
echo "================================================================="
cloudflared tunnel --url http://127.0.0.1:8000
