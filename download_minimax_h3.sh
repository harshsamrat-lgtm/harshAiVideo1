#!/bin/bash
# ==============================================================================
# AI Hindi Cinema Studio - MiniMax H3 High-Speed Model Downloader & GPU Setup
# Downloads H3-Base-Ref2VA weights & launches ComfyUI Engine on Port 8188
# ==============================================================================

set -e

echo "================================================================="
echo "📥 Setting up MiniMax H3 Open-Weights on GPU Server"
echo "================================================================="

# 1. Install high-speed download tools and dependencies
apt-get update -y
apt-get install -y aria2 ffmpeg git curl wget python3-pip

# 2. Setup ComfyUI if not already present
if [ ! -d "ComfyUI" ]; then
    echo "📦 Cloning ComfyUI repository..."
    git clone https://github.com/comfyanonymous/ComfyUI.git
fi

cd ComfyUI
echo "📦 Installing ComfyUI dependencies..."
pip install -r requirements.txt
pip install huggingface_hub torch torchvision torchaudio --upgrade

# 3. Create Model Directories
mkdir -p models/checkpoints
mkdir -p models/diffusion_models
mkdir -p models/text_encoders
mkdir -p models/vae

# 4. Fast Download MiniMax H3 Checkpoints (Hugging Face via aria2c - 16 parallel connections)
echo "================================================================="
echo "⚡ Downloading MiniMax H3 (H3-Base-Ref2VA) Checkpoint..."
echo "================================================================="

CHECKPOINT_PATH="models/checkpoints/H3-Base-Ref2VA.safetensors"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    # Download H3-Base-Ref2VA weights with 16 connections for blazing fast speed
    aria2c -x 16 -s 16 -k 1M \
      "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/H3-Base-Ref2VA.safetensors" \
      -d models/checkpoints \
      -o H3-Base-Ref2VA.safetensors || \
    wget -c "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/H3-Base-Ref2VA.safetensors" -O "$CHECKPOINT_PATH"
else
    echo "✅ MiniMax H3 weights already downloaded: $CHECKPOINT_PATH"
fi

# 5. Start ComfyUI Server on Port 8188 in Background
echo "================================================================="
echo "🚀 Launching ComfyUI GPU Server on port 8188..."
echo "================================================================="

# Kill any existing process on 8188
fuser -k 8188/tcp || true

nohup python3 main.py --listen 127.0.0.1 --port 8188 --gpu-only > comfyui.log 2>&1 &

cd ..

# 6. Wait and test connection
echo "⏳ Waiting for ComfyUI GPU Server to initialize..."
sleep 5

if curl -s http://127.0.0.1:8188/system_stats > /dev/null; then
    echo "🎉 SUCCESS: MiniMax H3 ComfyUI GPU Engine is ONLINE on port 8188!"
else
    echo "⚠️ ComfyUI is starting up, check log in ComfyUI/comfyui.log"
fi

echo "================================================================="
echo "✅ Setup Complete! You can now run ./start_server_tunnel.sh"
echo "================================================================="
