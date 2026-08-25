#!/bin/bash
# ==============================================================================
# AI Hindi Cinema Studio - Fast Single 20GB MiniMax H3 Downloader & ComfyUI Starter
# ==============================================================================

set -e

echo "================================================================="
echo "📥 Setting up MiniMax H3 (Single ~20GB File Only)"
echo "================================================================="

# 1. Setup ComfyUI if not already present
if [ ! -d "ComfyUI" ]; then
    echo "📦 Cloning ComfyUI repository..."
    git clone https://github.com/comfyanonymous/ComfyUI.git
fi

cd ComfyUI
pip install -r requirements.txt || true
cd ..

pip install huggingface_hub requests tqdm || true

# 2. Run Targeted 20GB Downloader
python3 download_models.py

# 3. Start ComfyUI Server on Port 8188 in Background
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
