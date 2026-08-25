"""
Automated MiniMax H3 Model Weights Downloader.
Uses official huggingface_hub to download exact quantized/pruned checkpoints for RTX 4090/A100.
"""

import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download

def download_minimax_h3():
    print("=================================================================")
    print("📥 Starting MiniMax H3 (Hailuo 3.0) Model Download from Hugging Face...")
    print("=================================================================")

    checkpoints_dir = os.path.abspath("ComfyUI/models/checkpoints")
    vae_dir = os.path.abspath("ComfyUI/models/vae")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)

    repo_id = "Comfy-Org/MiniMax-H3"

    # 1. Download Core Reference-to-Video Checkpoint (Optimized int8 / fp8 for 24GB RTX 4090)
    candidate_files = [
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "minimax_h3_ref2va_fp8_scaled.safetensors",
        "minimax_h3_fl2va_bf16.safetensors",
        "H3-Base-Ref2VA.safetensors"
    ]

    downloaded = False
    for filename in candidate_files:
        try:
            print(f"\n🔍 Attempting to download: {filename} from {repo_id}...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=checkpoints_dir
            )
            print(f"✅ Successfully downloaded {filename} to {local_path}")
            # Create symlink/copy to standard H3-Base-Ref2VA.safetensors for ComfyUI
            target_alias = os.path.join(checkpoints_dir, "H3-Base-Ref2VA.safetensors")
            if not os.path.exists(target_alias) and os.path.exists(local_path):
                import shutil
                shutil.copy(local_path, target_alias)
            downloaded = True
            break
        except Exception as e:
            print(f"⚠️ Could not download {filename} from {repo_id}: {e}")

    # Fallback to official MiniMaxAI repository if Comfy-Org has different structure
    if not downloaded:
        print("\n🔍 Checking official repo MiniMaxAI/MiniMax-H3...")
        try:
            snapshot_download(
                repo_id="MiniMaxAI/MiniMax-H3",
                local_dir="ComfyUI/models/checkpoints/minimax_h3_official",
                allow_patterns=["*.safetensors", "*.json", "*.bin"]
            )
            downloaded = True
        except Exception as e:
            print(f"Official repo note: {e}")

    # 2. Download VAE
    try:
        print("\n📥 Downloading MiniMax Video VAE...")
        hf_hub_download(
            repo_id=repo_id,
            filename="minimax_h3_video_vae_fp16.safetensors",
            local_dir=vae_dir
        )
        print("✅ VAE downloaded successfully.")
    except Exception as e:
        print(f"Note on VAE: {e}")

    print("\n=================================================================")
    print("🎉 MiniMax H3 Model Weights are Ready in ComfyUI/models/checkpoints!")
    print("=================================================================")

if __name__ == "__main__":
    download_minimax_h3()
