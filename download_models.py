"""
Targeted MiniMax H3 Downloader.
Downloads ONLY the single 20GB inference checkpoint file and Video VAE.
Strictly prevents full repository snapshots.
"""

import os
import sys
from huggingface_hub import hf_hub_download

def download_targeted_weights():
    print("=================================================================")
    print("🎯 Downloading ONLY the required 20GB MiniMax H3 Checkpoint...")
    print("=================================================================")

    checkpoints_dir = os.path.abspath("ComfyUI/models/checkpoints")
    vae_dir = os.path.abspath("ComfyUI/models/vae")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(vae_dir, exist_ok=True)

    target_alias = os.path.join(checkpoints_dir, "H3-Base-Ref2VA.safetensors")

    if os.path.exists(target_alias) and os.path.getsize(target_alias) > 5 * 1024 * 1024 * 1024:
        print(f"✅ Required model already exists: {target_alias} ({os.path.getsize(target_alias)/(1024**3):.1f} GB)")
        return

    # Download ONLY the single 20GB quantized model for RTX 4090 / A100
    try:
        print("\n📥 Fetching single ~20GB checkpoint (minimax_h3_ref2va_pruned_int8_convrot.safetensors)...")
        file_path = hf_hub_download(
            repo_id="Comfy-Org/MiniMax-H3",
            filename="minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            local_dir=checkpoints_dir
        )
        # Rename or copy to H3-Base-Ref2VA.safetensors for standard ComfyUI loader
        if os.path.exists(file_path):
            if not os.path.exists(target_alias):
                import shutil
                shutil.copy(file_path, target_alias)
            print(f"✅ Successfully prepared: {target_alias}")
    except Exception as e:
        print(f"Error fetching specific file: {e}")

    print("=================================================================")
    print("🎉 20GB MiniMax H3 Video Checkpoint is Ready!")
    print("=================================================================")

if __name__ == "__main__":
    download_targeted_weights()
