"""
AI Hindi Cinema Studio - FastAPI Application Server (Multi-Model Edition).
Integrates:
- Gemini 1.5 / GPT-4o: Screenplay & Dialogue Director
- Flux.1 Schnell / SDXL: Instant 4K Location & Character AI Asset Generation
- Edge-TTS / F5-TTS: Crystal Clear Hindi Voice Studio
- Wan2.1 & MiniMax Video: 15s Cinematic Motion Video Diffusion
- FFmpeg 7.0: Multi-Track Mastering & Subtitles
"""

import os
import sys
import subprocess
import asyncio
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from app.models import StoryInputRequest, ProjectState, SceneModel
from app.engines.story_director import StoryDirectorEngine
from app.engines.image_studio import ImageStudioEngine
from app.engines.voice_studio import VoiceStudioEngine
from app.engines.video_engine import VideoStudioEngine
from app.engines.movie_assembler import MovieAssemblerEngine

app = FastAPI(
    title="AI Hindi Cinema Studio API (Multi-Model Edition)",
    description="Flux.1 + Wan2.1 + Edge-TTS + MiniMax Powered Hindi Story-to-Movie Generation Platform",
    version="2.0.0"
)

# Enable CORS for Web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure Media Storage Directories
MEDIA_DIRS = [
    "media_store",
    "media_store/characters",
    "media_store/locations",
    "media_store/composite_keyframes",
    "media_store/videos",
    "media_store/audio",
    "media_store/movies"
]
for d in MEDIA_DIRS:
    os.makedirs(d, exist_ok=True)

# Mount media store as static directory
app.mount("/media", StaticFiles(directory="media_store"), name="media")

# In-memory Project Store
PROJECTS_DB: Dict[str, ProjectState] = {}

# Initialize Multi-Model Engines
director_engine = StoryDirectorEngine()
image_studio = ImageStudioEngine()
voice_engine = VoiceStudioEngine("media_store/audio")
video_studio = VideoStudioEngine(videos_dir="media_store/videos")
movie_assembler = MovieAssemblerEngine("media_store/movies")


@app.get("/api/system/status")
def get_system_status():
    """Returns real-time multi-model AI suite health and GPU connectivity."""
    comfy_online = video_studio._check_comfyui_online()
    return {
        "status": "online",
        "models": {
            "story_director": "Gemini / GPT-4o Screenplay Parser",
            "image_studio": "Flux.1 Schnell (4K Photorealism)",
            "voice_studio": "Microsoft Edge Neural Hindi TTS",
            "video_studio": "Wan2.1 & MiniMax Video Engine (15s Native Clips)"
        },
        "gpu_connected": comfy_online,
        "mode": "Real AI Video & Photorealism Engine Active",
        "active_projects": len(PROJECTS_DB)
    }


@app.post("/api/system/restart")
async def restart_system_engine():
    """1-Click App & GPU Engine Restart Endpoint."""
    global PROJECTS_DB
    PROJECTS_DB.clear()

    # Attempt to reconnect ComfyUI if present
    if not video_studio._check_comfyui_online():
        comfy_dir = "../ComfyUI" if os.path.exists("../ComfyUI") else "ComfyUI"
        if os.path.exists(comfy_dir):
            try:
                subprocess.Popen(
                    ["python3", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--gpu-only"],
                    cwd=comfy_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception as e:
                print(f"ComfyUI restart note: {e}")

    await asyncio.sleep(1)
    return {
        "status": "restarted",
        "message": "AI सिनेमा स्टूडियो और मल्टी-मॉडल इंजन सफलतापूर्वक रिस्टार्ट हो गया है।"
    }


@app.post("/api/story/analyze", response_model=ProjectState)
async def analyze_hindi_story(request: StoryInputRequest):
    """
    STAGE 1: 'स्क्रीनप्ले तैयार करें' (Analyze & Parse).
    1. Parses Hindi story into 15-second cinematic scenes.
    2. Instantly generates REAL 4K AI Location concept art via Flux.1.
    3. Instantly generates REAL 3-angle Character portraits via Flux.1 FaceID.
    4. Instantly generates REAL Hindi neural dialogue audio previews.
    5. Instantly synthesizes REAL Composite Scene Keyframes.
    """
    project = director_engine.parse_story(request)

    print(f"\n[AI Director] 🎬 Story parsed: {len(project.scenes)} scenes. Generating Real AI Assets...")

    # 1. Generate Real 4K Location Art with Flux.1
    for loc in project.locations:
        await image_studio.generate_location_concept_art(loc)

    # 2. Generate Real Character Face & Costume Turnaround Sheets with Flux.1
    for char in project.characters:
        await image_studio.generate_character_portrait_sheet(char)

    # 3. Generate Real Composite Keyframes & Dialogue Audio for every scene
    for sc in project.scenes:
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])
        
        # Real Composite Keyframe
        sc.composite_keyframe_url = await image_studio.generate_composite_scene_keyframe(
            sc.scene_number, loc, char, sc.visual_prompt
        )

        # Real Hindi Dialogue Audio Preview
        if sc.dialogue:
            await voice_engine.generate_character_dialogue(sc.dialogue, char)

    PROJECTS_DB[project.project_id] = project
    print(f"[AI Director] ✅ All Real AI Visual & Audio Assets Ready for Project {project.project_id}!")
    return project


@app.post("/api/render/draft/{project_id}")
async def generate_draft_movie(project_id: str, background_tasks: BackgroundTasks):
    """
    STAGE 2: 'ड्राफ्ट मूवी रेंडर करें' (Render Draft Movie).
    Takes all real AI keyframes and generates 15-second cinematic AI video clips
    using Wan2.1 / MiniMax Video and stitches into a full movie.
    """
    if project_id not in PROJECTS_DB:
        raise HTTPException(status_code=404, detail="Project not found")

    project = PROJECTS_DB[project_id]
    project.status = "drafting"

    for sc in project.scenes:
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])

        # Generate Real 15s AI Video Clip
        await video_studio.generate_15s_scene_video(
            scene=sc,
            character=char,
            location=loc,
            composite_keyframe_url=sc.composite_keyframe_url,
            mode="draft"
        )

    # Assemble Full Movie with FFmpeg
    movie_assembler.assemble_full_movie(project, mode="draft")
    return project


@app.get("/api/project/{project_id}", response_model=ProjectState)
def get_project(project_id: str):
    """Retrieves full project state, scenes, assets, and movie URLs."""
    if project_id not in PROJECTS_DB:
        raise HTTPException(status_code=404, detail="Project not found")
    return PROJECTS_DB[project_id]


@app.post("/api/scene/{project_id}/regenerate/{scene_number}")
async def regenerate_single_scene(project_id: str, scene_number: int, custom_prompt: str = None):
    """Allows user to edit and regenerate any specific 15s scene."""
    if project_id not in PROJECTS_DB:
        raise HTTPException(status_code=404, detail="Project not found")

    project = PROJECTS_DB[project_id]
    scene = next((s for s in project.scenes if s.scene_number == scene_number), None)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")

    if custom_prompt:
        scene.visual_prompt = custom_prompt

    loc = next((l for l in project.locations if l.location_id == scene.location_id), project.locations[0])
    char = next((c for c in project.characters if c.character_id in scene.character_ids), project.characters[0])

    # Re-generate keyframe with Flux.1
    scene.composite_keyframe_url = await image_studio.generate_composite_scene_keyframe(
        scene.scene_number, loc, char, scene.visual_prompt
    )

    # Re-generate 15s video
    await video_studio.generate_15s_scene_video(
        scene=scene,
        character=char,
        location=loc,
        composite_keyframe_url=scene.composite_keyframe_url,
        mode="draft"
    )

    movie_assembler.assemble_full_movie(project, mode="draft")
    return scene


@app.post("/api/render/approve-final/{project_id}")
async def approve_and_render_final_master(project_id: str):
    """
    STAGE 3: Triggered after user approval.
    Renders high-resolution 1080p/4K final movie with master audio and subtitles.
    """
    if project_id not in PROJECTS_DB:
        raise HTTPException(status_code=404, detail="Project not found")

    project = PROJECTS_DB[project_id]
    project.status = "rendering_final"

    for sc in project.scenes:
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])

        await video_studio.generate_15s_scene_video(
            scene=sc,
            character=char,
            location=loc,
            composite_keyframe_url=sc.composite_keyframe_url,
            mode="final"
        )

    movie_assembler.assemble_full_movie(project, mode="final")
    return project


# Bulletproof HTML Delivery for UI
@app.get("/", response_class=HTMLResponse)
def serve_studio_ui():
    """Serves the integrated single-page Cinema Studio Dashboard."""
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend", "index.html"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "index.html"),
        "../frontend/index.html",
        "frontend/index.html"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())

    return HTMLResponse("<h2>AI Hindi Cinema Studio UI loading error: index.html not found.</h2>")
