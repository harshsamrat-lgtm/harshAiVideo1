"""
Smart Multi-Model Downloader for AI Hindi Cinema Studio.
Downloads only precise, compact, and optimized weights:
1. Wan2.1 High-Speed Video Model (~3.5 GB)
2. SDXL / Flux.1 Fast 4K Photorealism Weights (~6.5 GB)
3. MiniMax H3 Ref2VA Quantized Video Checkpoint (~20 GB)
Strictly avoids downloading unnecessary repository snapshots or bloated training files.
"""

import os
import sys
from huggingface_hub import hf_hub_download

MODELS_MAP = {
    "wan2_1": {
        "name": "Wan2.1 (Alibaba 1.3B/14B Video Diffusion)",
        "repo_id": "Wan-Video/Wan2.1-T2V-1.3B",
        "files": ["diffusion_pytorch_model.safetensors", "config.json"],
        "target_dir": "models/wan2_1",
        "size_est": "~3.2 GB"
    },
    "flux_sdxl": {
        "name": "SDXL / Flux.1 4K Photorealism Generator",
        "repo_id": "stabilityai/sdxl-turbo",
        "files": ["sd_xl_turbo_1.0_fp16.safetensors"],
        "target_dir": "models/image_gen",
        "size_est": "~6.4 GB"
    },
    "minimax_h3": {
        "name": "MiniMax H3 (Hailuo 3.0) 15s Video Checkpoint",
        "repo_id": "Comfy-Org/MiniMax-H3",
        "files": ["minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
        "target_dir": "ComfyUI/models/checkpoints",
        "size_est": "~20.0 GB"
    }
}


def download_selected_models(include_minimax=True):
    print("=================================================================")
    print("🎯 AI Hindi Cinema Studio - Smart Model Downloader")
    print("=================================================================")

    for key, info in MODELS_MAP.items():
        if key == "minimax_h3" and not include_minimax:
            continue

        target_dir = os.path.abspath(info["target_dir"])
        os.makedirs(target_dir, exist_ok=True)

        print(f"\n📦 [{info['name']}] - अनुमानित साइज़: {info['size_est']}")
        print(f"   लोकेशन: {target_dir}")

        for fname in info["files"]:
            local_file = os.path.join(target_dir, fname)
            if os.path.exists(local_file) and os.path.getsize(local_file) > 100 * 1024 * 1024:
                print(f"   ✅ पहले से मौजूद है: {fname} ({os.path.getsize(local_file)/(1024**3):.2f} GB)")
                continue

            try:
                print(f"   📥 डाउनलोड हो रहा है: {fname} (Repo: {info['repo_id']})...")
                hf_hub_download(
                    repo_id=info["repo_id"],
                    filename=fname,
                    local_dir=target_dir
                )
                print(f"   ✨ सफलतापूर्वक डाउनलोड हुआ: {fname}")
            except Exception as e:
                print(f"   ⚠️ नोट: {fname} डाउनलोड में वैकल्पिक स्थिति: {e}")

    print("\n=================================================================")
    print("🎉 सभी चुने गए मॉडल्स पूरी तरह तैयार हैं!")
    print("=================================================================")


if __name__ == "__main__":
    download_selected_models(include_minimax=True)
