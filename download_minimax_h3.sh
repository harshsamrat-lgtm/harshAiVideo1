#!/bin/bash
# ==============================================================================
# AI Hindi Cinema Studio - MiniMax H3 Model Downloader & ComfyUI Starter
# ==============================================================================

set -e

echo "================================================================="
echo "📥 Setting up MiniMax H3 (Hailuo 3.0) on GPU Server"
echo "================================================================="

# 1. Install dependencies
apt-get update -y || true
apt-get install -y ffmpeg git curl wget python3-pip || true

pip install huggingface_hub requests tqdm || true

# 2. Setup ComfyUI if not already present
if [ ! -d "ComfyUI" ]; then
    echo "📦 Cloning ComfyUI repository..."
    git clone https://github.com/comfyanonymous/ComfyUI.git
fi

cd ComfyUI
pip install -r requirements.txt || true
cd ..

# 3. Run Python Downloader to fetch exact MiniMax H3 Safetensors
echo "📥 Running Python Hugging Face Downloader..."
python3 download_models.py

# 4. Start ComfyUI Server on Port 8188 in Background
echo "================================================================="
echo "🚀 Launching ComfyUI GPU Server on port 8188..."
echo "================================================================="

fuser -k 8188/tcp || true

cd ComfyUI
nohup python3 main.py --listen 127.0.0.1 --port 8188 --gpu-only > comfyui.log 2>&1 &
cd ..

echo "⏳ Waiting for ComfyUI to initialize..."
sleep 5

if curl -s http://127.0.0.1:8188/system_stats > /dev/null; then
    echo "🎉 SUCCESS: MiniMax H3 ComfyUI GPU Engine is ONLINE on port 8188!"
else
    echo "⚠️ ComfyUI is loading weights into VRAM. Check log in ComfyUI/comfyui.log"
fi

echo "================================================================="
echo "✅ Setup Complete! Now run: ./start_server_tunnel.sh"
echo "================================================================="
