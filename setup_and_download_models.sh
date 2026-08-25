#!/bin/bash
# ==============================================================================
# AI Hindi Cinema Studio - Automated Model Downloader & Environment Setup
# ==============================================================================

set -e

echo "================================================================="
echo "🎬 AI Hindi Cinema Studio: Smart Model Setup"
echo "================================================================="

# 1. System Packages
apt-get update -y || true
apt-get install -y ffmpeg git curl wget python3-pip || true

# 2. Python Packages for Multi-Model Suite
echo "📦 Installing AI Libraries (Edge-TTS, Diffusers, Huggingface Hub, FastAPI)..."
pip install -r backend/requirements.txt || true
pip install huggingface_hub torch torchvision torchaudio edge-tts requests tqdm || true

# 3. Setup ComfyUI (if needed for local diffusion)
if [ ! -d "ComfyUI" ]; then
    echo "📦 Setting up ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git || true
fi

# 4. Run Smart Downloader for exact weights (Wan2.1 + SDXL/Flux + MiniMax)
echo "📥 Running Smart Multi-Model Downloader..."
python3 download_models.py

echo "================================================================="
echo "✅ Setup Complete! Now run: ./start_server_tunnel.sh"
echo "================================================================="
