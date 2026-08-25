#!/bin/bash
# ==============================================================================
# AI Hindi Cinema Studio - Multi-Model High-Speed Downloader (Wan2.1 14B + 1.3B + MiniMax)
# ==============================================================================

set -e

echo "================================================================="
echo "⚡ AI Hindi Cinema Studio: High-Speed Model Downloader"
echo "================================================================="

apt-get update -y || true
apt-get install -y aria2 wget curl ffmpeg || true

mkdir -p models/wan2_1
mkdir -p models/image_gen
mkdir -p ComfyUI/models/checkpoints

echo ""
echo "कृपया चुनें कि आप Wan2.1 का कौन सा वर्ज़न डाउनलोड करना चाहते हैं:"
echo "1) Wan2.1-14B फ्लैगशिप मास्टर मॉडल (~28 GB - सिनेमा-ग्रेड 1080p मास्टर क्वालिटी) [अनुशंसित]"
echo "2) Wan2.1-14B FP8 क्वांटाइज़्ड (~14 GB - तेज़ 24GB VRAM अनुकूलित)"
echo "3) Wan2.1-1.3B लाइटवेट वर्ज़न (~3.2 GB - सुपरफास्ट ड्राफ्ट)"
echo "4) सब कुछ डाउनलोड करें (Wan2.1 14B + MiniMax H3 + SDXL 4K)"
echo ""

# Default to Flagship 14B if unattended
CHOICE=${1:-4}

if [ "$CHOICE" == "1" ] || [ "$CHOICE" == "4" ]; then
    echo "📥 Downloading Wan2.1-14B Flagship Video Model (~28 GB)..."
    aria2c -x 16 -s 16 -k 1M \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors" \
        -d models/wan2_1 \
        -o wan2.1_i2v_720p_14B_bf16.safetensors || \
    aria2c -x 16 -s 16 -k 1M \
        "https://huggingface.co/Wan-Video/Wan2.1-I2V-14B-720P/resolve/main/diffusion_pytorch_model.safetensors" \
        -d models/wan2_1 \
        -o wan2.1_i2v_720p_14B_bf16.safetensors || true
    echo "✅ Wan2.1-14B Master Download Finished!"
fi

if [ "$CHOICE" == "2" ]; then
    echo "📥 Downloading Wan2.1-14B FP8 Optimized Video Model (~14.5 GB)..."
    aria2c -x 16 -s 16 -k 1M \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_720p_14B_fp8_e4m3fn.safetensors" \
        -d models/wan2_1 \
        -o wan2.1_i2v_720p_14B_fp8.safetensors || true
    echo "✅ Wan2.1-14B FP8 Download Finished!"
fi

if [ "$CHOICE" == "3" ] || [ "$CHOICE" == "4" ]; then
    echo "📥 Downloading Wan2.1-1.3B Fast Video Model (~3.2 GB)..."
    aria2c -x 16 -s 16 -k 1M \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors" \
        -d models/wan2_1 \
        -o Wan2.1_T2V_1.3B_bf16.safetensors || true
    echo "✅ Wan2.1-1.3B Download Finished!"
fi

if [ "$CHOICE" == "4" ]; then
    echo "📥 Downloading MiniMax H3 Ref2VA Video Checkpoint (~18.5 GB)..."
    aria2c -x 16 -s 16 -k 1M \
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
        -d ComfyUI/models/checkpoints \
        -o H3-Base-Ref2VA.safetensors || true
fi

echo ""
echo "================================================================="
echo "📊 Current Model Status on Disk:"
python3 check_models_status.py
echo "================================================================="
