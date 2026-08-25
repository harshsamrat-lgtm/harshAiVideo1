#!/bin/bash
# ==============================================================================
# AI Hindi Cinema Studio - PyTorch CUDA 12.8 / Blackwell (RTX 5070 Ti) Setup
# ==============================================================================

set -e

echo "================================================================="
echo "🔧 Fixing PyTorch CUDA Driver Matching for RTX 5070 Ti (CUDA 12.8)"
echo "================================================================="

# 1. Uninstall mismatched torch
echo "🧹 Removing old PyTorch..."
pip uninstall -y torch torchvision torchaudio || true

# 2. Install PyTorch matching CUDA 12.8 / 12.6 driver
echo "📦 Installing PyTorch with CUDA 12.8 / 12.6 support..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 || \
pip install torch torchvision torchaudio

# 3. Ensure diffusers, transformers, accelerate are present
pip install diffusers transformers accelerate imageio[ffmpeg] || true

# 4. Verify CUDA initialization
echo "================================================================="
echo "🧪 Checking CUDA GPU Status..."
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')"

# 5. Run Real GPU Video Diffusion Test
echo "================================================================="
echo "🎬 Running Real 20-Step AI Video Diffusion Test on RTX 5070 Ti..."
python3 test_gpu_real_video.py

echo "================================================================="
echo "🎉 Setup Complete! Now run: ./start_server_tunnel.sh"
echo "================================================================="
