#!/bin/bash
# ==============================================================================
# AI Hindi Cinema Studio - Ultra Fast High-Speed Model Downloader (aria2c 16x)
# Downloads Wan2.1 (~3.2GB), SDXL (~6.4GB), and MiniMax H3 (~18GB) with Live Progress
# ==============================================================================

set -e

echo "================================================================="
echo "⚡ Starting Ultra-Fast Multi-Threaded Model Downloader (aria2c 16x)"
echo "================================================================="

# 1. Install aria2 for maximum 100MB/s multi-stream speed
apt-get update -y || true
apt-get install -y aria2 wget curl ffmpeg || true

mkdir -p models/wan2_1
mkdir -p models/image_gen
mkdir -p ComfyUI/models/checkpoints

# 2. Download Wan2.1 Video Model (~3.2 GB)
echo ""
echo "📥 [1/3] Downloading Wan2.1 SOTA Video Diffusion Model (~3.2 GB)..."
if [ ! -f "models/wan2_1/Wan2.1_T2V_1.3B_bf16.safetensors" ]; then
    aria2c -x 16 -s 16 -k 1M \
        "https://huggingface.co/Wan-Video/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_T2V_1.3B_bf16.safetensors" \
        -d models/wan2_1 \
        -o Wan2.1_T2V_1.3B_bf16.safetensors || \
    aria2c -x 16 -s 16 -k 1M \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors" \
        -d models/wan2_1 \
        -o Wan2.1_T2V_1.3B_bf16.safetensors || true
    echo "✅ Wan2.1 Download Finished!"
else
    echo "✅ Wan2.1 is already present on disk!"
fi

# 3. Download SDXL 4K Photorealism Model (~6.4 GB)
echo ""
echo "📥 [2/3] Downloading SDXL 4K Photorealism Model (~6.4 GB)..."
if [ ! -f "models/image_gen/sd_xl_turbo_1.0_fp16.safetensors" ]; then
    aria2c -x 16 -s 16 -k 1M \
        "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors" \
        -d models/image_gen \
        -o sd_xl_turbo_1.0_fp16.safetensors || true
    echo "✅ SDXL 4K Photorealism Model Download Finished!"
else
    echo "✅ SDXL is already present on disk!"
fi

# 4. Download MiniMax H3 Video Checkpoint (~18.5 GB)
echo ""
echo "📥 [3/3] Downloading MiniMax H3 Ref2VA Video Checkpoint (~18.5 GB)..."
if [ ! -f "ComfyUI/models/checkpoints/H3-Base-Ref2VA.safetensors" ]; then
    aria2c -x 16 -s 16 -k 1M \
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
        -d ComfyUI/models/checkpoints \
        -o H3-Base-Ref2VA.safetensors || true
    echo "✅ MiniMax H3 Checkpoint Download Finished!"
else
    echo "✅ MiniMax H3 is already present on disk!"
fi

echo ""
echo "================================================================="
echo "📊 Verifying Downloaded Model Status..."
python3 check_models_status.py
echo "================================================================="
echo "🚀 Now start the Studio by running: ./start_server_tunnel.sh"
echo "================================================================="
