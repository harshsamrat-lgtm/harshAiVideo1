#!/bin/bash
# ==============================================================================
# AI Hindi Cinema Studio - 1-Click NVIDIA RTX 5070 Ti (CUDA 12.8) Setup Script
# Installs PyTorch, Diffusers, Transformers, Accelerate and tests Real GPU Video Generation
# ==============================================================================

set -e

echo "================================================================="
echo "🚀 Setting up AI Video Engine for NVIDIA RTX 5070 Ti (16GB VRAM)"
echo "================================================================="

# 1. System packages
apt-get update -y || true
apt-get install -y ffmpeg git curl wget python3-pip aria2 || true

# 2. Python AI & CUDA Libraries
echo "📦 Installing PyTorch, Diffusers, Transformers, Accelerate..."
pip install --upgrade pip || true
pip install fastapi uvicorn pydantic python-multipart pillow requests edge-tts huggingface_hub tqdm || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 || pip install torch torchvision torchaudio || true
pip install diffusers transformers accelerate imageio[ffmpeg] || true

# 3. Test Real GPU Diffusion immediately
echo "================================================================="
echo "🧪 Running Real GPU Video Diffusion Test on RTX 5070 Ti..."
python3 test_gpu_real_video.py

echo "================================================================="
echo "🎉 Setup Complete! Now run: ./start_server_tunnel.sh"
echo "================================================================="
