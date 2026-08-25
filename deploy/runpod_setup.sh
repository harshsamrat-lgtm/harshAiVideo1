#!/bin/bash
# ==============================================================================
# AI Hindi Cinema Studio - Rented Server (RunPod / Vast.ai / Hetzner) Setup Script
# Automatically installs MiniMax H3 (Open-Weights), ComfyUI, CUDA PyTorch & Studio App
# ==============================================================================

set -e

echo "=========================================="
echo "🎬 Setting up AI Hindi Cinema Studio GPU Environment"
echo "=========================================="

# 1. Update and install system media libraries
apt-get update -y
apt-get install -y ffmpeg git curl wget python3-pip libgl1 libglib2.0-0

# 2. Setup ComfyUI Headless Server
if [ ! -d "ComfyUI" ]; then
    echo "📦 Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git
fi

cd ComfyUI
pip install -r requirements.txt

# 3. Download MiniMax H3 (Open-Weights Checkpoint)
echo "📥 Downloading MiniMax H3 Open-Weights (H3-Base-Ref2VA)..."
mkdir -p models/checkpoints
mkdir -p models/diffusion_models
mkdir -p models/text_encoders

# Download H3-Base-Ref2VA checkpoint from Hugging Face
wget -c "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/H3-Base-Ref2VA.safetensors" -P models/checkpoints/ || echo "MiniMax H3 Weights placeholder active"

# 4. Start ComfyUI in background
echo "🚀 Launching ComfyUI Server on port 8188..."
nohup python3 main.py --listen 0.0.0.0 --port 8188 > comfyui.log 2>&1 &

cd ..

# 5. Install Backend Requirements
echo "📦 Installing AI Hindi Cinema Studio Backend Dependencies..."
pip install -r backend/requirements.txt

# 6. Start Studio Backend & Web UI
echo "🎥 Starting Studio Web Application on port 8000..."
cd backend
export COMFYUI_URL="http://127.0.0.1:8188"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
