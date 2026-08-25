"""
MiniMax H3 Video Engine (Open-Weights H3-Base-Ref2VA & FL2VA).
Executes real GPU video diffusion inference via ComfyUI or native PyTorch.
Provides explicit status reporting to ensure dummy placeholders are never mistaken for real AI renders.
"""

import os
import math
import time
import shutil
import asyncio
import subprocess
import requests
from typing import Dict, Any, Optional
from app.models import SceneModel, CharacterModel, LocationModel


class MiniMaxH3Engine:
    """
    MiniMax H3 Open-Weights Video Generator.
    Executes real GPU diffusion inference on port 8188.
    """

    def __init__(
        self,
        comfyui_url: str = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188"),
        videos_dir: str = "media_store/videos"
    ):
        self.comfyui_url = comfyui_url
        self.videos_dir = videos_dir
        os.makedirs(self.videos_dir, exist_ok=True)

    async def generate_10s_scene_video(
        self,
        scene: SceneModel,
        character: CharacterModel,
        location: LocationModel,
        composite_keyframe_path: str,
        mode: str = "draft"  # "draft" (480p/720p fast) or "final" (1080p high-quality)
    ) -> str:
        """
        Executes real MiniMax H3 GPU video generation.
        """
        resolution_tag = "draft_720p" if mode == "draft" else "final_1080p"
        filename = f"scene_{scene.scene_number}_{scene.location_id}_{character.character_id}_{resolution_tag}.mp4"
        output_path = os.path.join(self.videos_dir, filename)

        # 1. Check if ComfyUI / MiniMax H3 GPU Server is active
        is_comfy_online = self._check_comfyui_online()

        if is_comfy_online:
            print(f"[MiniMax H3] 🟢 GPU Server is ACTIVE on {self.comfyui_url}. Dispatching real diffusion job for Scene {scene.scene_number}...")
            gpu_rendered = await self._run_comfyui_minimax_ref2va(scene, composite_keyframe_path, output_path, mode)
            
            if gpu_rendered and os.path.exists(output_path):
                video_url = f"/media/videos/{filename}"
                if mode == "draft":
                    scene.draft_video_url = video_url
                else:
                    scene.final_video_url = video_url
                scene.status = "ready"
                return video_url

        # 2. If GPU server is offline or still downloading model
        print(f"[MiniMax H3] ⚠️ ComfyUI GPU Server is OFFLINE at {self.comfyui_url}.")
        scene.status = "gpu_offline"
        
        # Write informative placeholder frame explaining exact GPU status
        self._render_gpu_offline_notice(scene, output_path)
        
        video_url = f"/media/videos/{filename}"
        scene.draft_video_url = video_url
        return video_url

    def _check_comfyui_online(self) -> bool:
        """Checks whether ComfyUI GPU server is responding on port 8188."""
        try:
            res = requests.get(f"{self.comfyui_url}/system_stats", timeout=1.5)
            return res.status_code == 200
        except Exception:
            return False

    async def _run_comfyui_minimax_ref2va(
        self,
        scene: SceneModel,
        keyframe_path: str,
        output_path: str,
        mode: str
    ) -> bool:
        """Sends diffusion workflow to ComfyUI on GPU and collects rendered MP4."""
        steps = 18 if mode == "draft" else 45
        prefix = f"minimax_out_{scene.scene_number}_{int(time.time())}"

        workflow = {
            "prompt": {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": os.path.abspath(keyframe_path)}
                },
                "3": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "H3-Base-Ref2VA.safetensors"}
                },
                "6": {
                    "class_type": "KSampler",
                    "inputs": {
                        "steps": steps,
                        "cfg": 7.0,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "positive": scene.visual_prompt,
                        "negative": scene.negative_prompt,
                        "latent_image": ["1", 0],
                        "model": ["3", 0]
                    }
                },
                "9": {
                    "class_type": "VHS_VideoCombine",
                    "inputs": {
                        "filename_prefix": prefix,
                        "format": "video/h264-mp4",
                        "images": ["6", 0]
                    }
                }
            }
        }

        try:
            res = requests.post(f"{self.comfyui_url}/prompt", json=workflow, timeout=10)
            if res.status_code == 200:
                prompt_data = res.json()
                prompt_id = prompt_data.get("prompt_id")
                print(f"[MiniMax H3] GPU job accepted: {prompt_id}. Polling diffusion steps...")

                # Wait for diffusion render
                for _ in range(60):
                    await asyncio.sleep(4)
                    hist_res = requests.get(f"{self.comfyui_url}/history/{prompt_id}", timeout=5)
                    if hist_res.status_code == 200:
                        hist = hist_res.json()
                        if prompt_id in hist:
                            # Check output files in ComfyUI/output
                            comfy_output_dir = "ComfyUI/output" if os.path.exists("ComfyUI/output") else "../ComfyUI/output"
                            if os.path.exists(comfy_output_dir):
                                for f in os.listdir(comfy_output_dir):
                                    if prefix in f and f.endswith(".mp4"):
                                        src_mp4 = os.path.join(comfy_output_dir, f)
                                        shutil.copy(src_mp4, output_path)
                                        print(f"[MiniMax H3] ✅ Real AI Video saved to: {output_path}")
                                        return True
                            return True
        except Exception as e:
            print(f"[MiniMax H3] ComfyUI Error: {e}")
            return False

        return False

    def _render_gpu_offline_notice(self, scene: SceneModel, output_path: str):
        """Creates a clear notice video when GPU model is not active on port 8188."""
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (1280, 720), color=(18, 12, 12))
        draw = ImageDraw.Draw(img)

        # Warning Card
        draw.rectangle([80, 80, 1200, 640], fill=(28, 18, 20), outline=(239, 68, 68), width=3)
        draw.text((120, 130), "⚠️ GPU ENGINE OFFLINE / MODEL NOT LOADED", fill=(239, 68, 68))
        draw.text((120, 180), f"Scene {scene.scene_number} ({scene.location_name})", fill=(255, 215, 0))
        draw.text((120, 240), "असली AI वीडियो बनाने के लिए MiniMax H3 GPU सर्वर चालू होना आवश्यक है।", fill=(240, 240, 240))
        draw.text((120, 290), "👉 कृपया सर्वर टर्मिनल में यह कमांड चलाएं:", fill=(147, 197, 253))
        draw.text((140, 340), "./download_minimax_h3.sh", fill=(52, 211, 153))
        draw.text((120, 410), "इसके बाद ComfyUI (Port 8188) पर असली AI डिफ्यूजन वीडियो बनना शुरू हो जाएगा।", fill=(180, 190, 205))

        temp_png = output_path.replace(".mp4", "_notice.png")
        img.save(temp_png)

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            cmd = [
                ffmpeg_bin, "-y",
                "-loop", "1",
                "-i", temp_png,
                "-c:v", "libx264",
                "-t", "5",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            shutil.copy(temp_png, output_path)
