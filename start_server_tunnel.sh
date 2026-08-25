#!/bin/bash
# ==============================================================================
# AI Hindi Cinema Studio - Combined Server & Cloudflare Tunnel Launcher
# ==============================================================================

set -e

echo "================================================================="
echo "🎬 Starting AI Hindi Cinema Studio + Public Web Tunnel"
echo "================================================================="

# 1. Install dependencies if needed
pip install -r backend/requirements.txt || pip install fastapi uvicorn pillow pydantic requests

# 2. Check if ComfyUI is running on Port 8188
if curl -s http://127.0.0.1:8188/system_stats > /dev/null; then
    echo "🟢 GPU Engine: ComfyUI is running on port 8188!"
elif [ -d "ComfyUI" ]; then
    echo "🚀 Starting ComfyUI GPU Server in background on port 8188..."
    cd ComfyUI
    nohup python3 main.py --listen 127.0.0.1 --port 8188 --gpu-only > comfyui.log 2>&1 &
    cd ..
    sleep 3
else
    echo "⚠️ Note: MiniMax H3 ComfyUI is not yet installed."
    echo "👉 Run: './download_minimax_h3.sh' to download MiniMax H3 weights and start GPU Engine!"
fi

# 3. Kill any old process on port 8000
fuser -k 8000/tcp || true

# 4. Start Studio FastAPI backend in background
echo "🚀 Starting Studio Backend on port 8000..."
cd backend
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > ../server.log 2>&1 &
cd ..

# 5. Wait for backend to be ready
echo "⏳ Waiting for backend to initialize..."
sleep 3

# 6. Check if backend is alive
if curl -s http://127.0.0.1:8000/api/system/status > /dev/null; then
    echo "✅ Studio Backend is healthy and running on port 8000!"
else
    echo "⚠️ Backend starting up, checking log:"
    cat server.log || true
fi

# 7. Install cloudflared if not present
if ! command -v cloudflared &> /dev/null; then
    echo "📥 Installing Cloudflared..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi

# 8. Start Cloudflare Tunnel
echo "================================================================="
echo "🌐 Launching Public Web URL... (Click the https link below 👇)"
echo "================================================================="
cloudflared tunnel --url http://127.0.0.1:8000
