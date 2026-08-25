"""
Standalone Real GPU Video Diffusion Test for NVIDIA RTX 5070 Ti / RTX 4090 / A100.
Directly loads Stable Video Diffusion / Wan2.1 onto CUDA, runs 20 denoising steps,
and outputs a genuine neural AI video with real moving pixels and physics.
"""

import os
import sys
import time
from PIL import Image, ImageDraw

def test_real_gpu_video():
    print("=================================================================")
    print("🎬 Testing Real GPU Neural AI Video Diffusion on CUDA...")
    print("=================================================================")

    # 1. Check CUDA & GPU Info
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ Error: PyTorch CUDA is not available. Please install PyTorch with CUDA support.")
            return False

        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🟢 GPU Detected: {gpu_name}")
        print(f"📊 Total VRAM: {total_vram:.2f} GB")
    except ImportError:
        print("❌ Error: PyTorch is not installed. Run: pip install torch")
        return False

    # 2. Prepare Sample Test Keyframe
    os.makedirs("media_store/test", exist_ok=True)
    test_img_path = "media_store/test/test_keyframe.jpg"
    
    img = Image.new("RGB", (1024, 576), color=(25, 30, 45))
    draw = ImageDraw.Draw(img)
    # Draw cinematic landscape with sun and character silhouette
    draw.ellipse([400, 150, 624, 374], fill=(255, 180, 50)) # Golden Sun
    draw.rectangle([0, 380, 1024, 576], fill=(15, 20, 25)) # Ground
    draw.ellipse([480, 320, 544, 420], fill=(220, 150, 100)) # Character Head
    draw.polygon([(512, 420), (450, 540), (574, 540)], fill=(40, 50, 70)) # Character Body
    img.save(test_img_path)
    print(f"🖼️ Sample Keyframe Image Created: {test_img_path}")

    # 3. Load Real Diffusion Pipeline
    try:
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        print("\n📥 Loading Stable Video Diffusion XT Model into GPU VRAM (FP16)...")
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        pipeline.enable_model_cpu_offload()

        vram_used = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"⚡ Model Loaded in VRAM! Current Usage: {vram_used:.2f} GB / {total_vram:.2f} GB")

        # 4. Run Real 20-Step Diffusion on GPU
        print("\n⚡ Running Real 20-Step Neural AI Video Diffusion on GPU...")
        start_time = time.time()
        
        image = load_image(test_img_path).resize((1024, 576))
        generator = torch.manual_seed(42)
        
        frames = pipeline(
            image,
            decode_chunk_size=4,
            generator=generator,
            num_inference_steps=20,
            motion_bucket_id=127,
            noise_aug_strength=0.02
        ).frames[0]

        elapsed = time.time() - start_time
        print(f"✅ GPU Diffusion Finished in {elapsed:.1f} seconds!")

        # 5. Export Real Video File
        out_mp4 = "media_store/test/real_gpu_video_output.mp4"
        export_to_video(frames, out_mp4, fps=8)
        print(f"🎉 REAL AI NEURAL VIDEO SAVED TO: {out_mp4}")
        print(f"   फ़ाइल साइज़: {os.path.getsize(out_mp4)/(1024**2):.2f} MB")
        print("=================================================================")
        print("✅ SUCCESS: Your GPU is 100% READY for Real AI Video Diffusion!")
        print("=================================================================")
        return True

    except Exception as e:
        print(f"❌ Error during diffusion execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_real_gpu_video()
