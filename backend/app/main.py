"""
AI Hindi Cinema Studio - FastAPI Application Server.
Orchestrates MiniMax H3 Video Gen, Character Consistency, Location DNA, Voice Studio, and Movie Assembly.
Includes 1-Click App & GPU Engine Restart Capability.
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
from app.engines.location_manager import LocationManager
from app.engines.character_manager import CharacterManager
from app.engines.voice_studio import VoiceStudioEngine
from app.engines.minimax_h3_engine import MiniMaxH3Engine
from app.engines.movie_assembler import MovieAssemblerEngine

app = FastAPI(
    title="AI Hindi Cinema Studio API",
    description="MiniMax H3 Powered Automated Hindi Story-to-Movie Generation Platform",
    version="1.0.0"
)

# Enable CORS for Next.js / React / Browser UI
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

# Initialize Core Engines
director_engine = StoryDirectorEngine()
location_mgr = LocationManager("media_store/locations")
character_mgr = CharacterManager("media_store/characters")
voice_engine = VoiceStudioEngine("media_store/audio")
video_engine = MiniMaxH3Engine(videos_dir="media_store/videos")
movie_assembler = MovieAssemblerEngine("media_store/movies")


@app.get("/api/system/status")
def get_system_status():
    """Returns GPU status, ComfyUI MiniMax H3 connectivity, and engine health."""
    comfy_online = video_engine._check_comfyui_online()
    return {
        "status": "online",
        "minimax_h3_model": "H3-Base-Ref2VA (Open-Weights)",
        "comfyui_gpu_connected": comfy_online,
        "mode": "GPU Accelerated" if comfy_online else "High-Fidelity Studio Simulation",
        "active_projects": len(PROJECTS_DB)
    }


@app.post("/api/system/restart")
async def restart_system_engine():
    """
    1-Click App & GPU Engine Restart Endpoint.
    Resets project memory and attempts to reconnect or re-launch ComfyUI on port 8188.
    """
    global PROJECTS_DB
    PROJECTS_DB.clear()

    # Re-check and attempt to launch ComfyUI if offline
    if not video_engine._check_comfyui_online():
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
                print(f"ComfyUI auto-restart error: {e}")

    await asyncio.sleep(1)
    comfy_online = video_engine._check_comfyui_online()

    return {
        "status": "restarted",
        "message": "ऐप और इंजन सफलतापूर्वक रिस्टार्ट हो गया है।",
        "comfyui_gpu_connected": comfy_online
    }


@app.post("/api/story/analyze", response_model=ProjectState)
async def analyze_hindi_story(request: StoryInputRequest):
    """
    Step 1: Analyzes raw Hindi story text.
    Builds persistent Character profiles, Location DNAs, and 15s max scene breakdowns.
    """
    project = director_engine.parse_story(request)

    # 1. Generate master character turnaround sheets
    for char in project.characters:
        character_mgr.generate_master_character_sheet(char)

    # 2. Generate master location establishing shots
    for loc in project.locations:
        location_mgr.generate_master_establishing_shot(loc)

    # 3. Generate composite keyframes for all scenes
    for sc in project.scenes:
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])
        sc.composite_keyframe_url = location_mgr.create_composite_keyframe(
            sc.scene_number, loc, char, sc.lighting_mood
        )

    PROJECTS_DB[project.project_id] = project
    return project


@app.post("/api/render/draft/{project_id}")
async def generate_draft_movie(project_id: str, background_tasks: BackgroundTasks):
    """
    Step 2: Generates fast 15-second draft clips (480p/720p) + Hindi dialogues + stitches draft movie.
    """
    if project_id not in PROJECTS_DB:
        raise HTTPException(status_code=404, detail="Project not found")

    project = PROJECTS_DB[project_id]
    project.status = "drafting"

    async def _process_draft_pipeline():
        for sc in project.scenes:
            loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
            char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])

            # 1. Synthesize Hindi Voice Dialogue
            if sc.dialogue:
                await voice_engine.generate_character_dialogue(sc.dialogue, char)

            # 2. Generate 15s Draft Video
            keyframe_path = sc.composite_keyframe_url.replace("/media/", "media_store/")
            await video_engine.generate_10s_scene_video(
                scene=sc,
                character=char,
                location=loc,
                composite_keyframe_path=keyframe_path,
                mode="draft"
            )

        # 3. Assemble Draft Full Movie
        movie_assembler.assemble_full_movie(project, mode="draft")

    await _process_draft_pipeline()
    return project


@app.get("/api/project/{project_id}", response_model=ProjectState)
def get_project(project_id: str):
    """Retrieves full project state, scenes, assets, and movie URLs."""
    if project_id not in PROJECTS_DB:
        raise HTTPException(status_code=404, detail="Project not found")
    return PROJECTS_DB[project_id]


@app.post("/api/scene/{project_id}/regenerate/{scene_number}")
async def regenerate_single_scene(project_id: str, scene_number: int, custom_prompt: str = None):
    """
    Allows user to edit and regenerate any specific 15s scene in the timeline.
    """
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

    keyframe_path = scene.composite_keyframe_url.replace("/media/", "media_store/")
    await video_engine.generate_10s_scene_video(
        scene=scene,
        character=char,
        location=loc,
        composite_keyframe_path=keyframe_path,
        mode="draft"
    )

    movie_assembler.assemble_full_movie(project, mode="draft")
    return scene


@app.post("/api/render/approve-final/{project_id}")
async def approve_and_render_final_master(project_id: str):
    """
    Step 3: Triggered after user approval.
    Renders high-resolution 1080p/4K final movie with master audio and subtitles.
    """
    if project_id not in PROJECTS_DB:
        raise HTTPException(status_code=404, detail="Project not found")

    project = PROJECTS_DB[project_id]
    project.status = "rendering_final"

    for sc in project.scenes:
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])

        keyframe_path = sc.composite_keyframe_url.replace("/media/", "media_store/")
        await video_engine.generate_10s_scene_video(
            scene=sc,
            character=char,
            location=loc,
            composite_keyframe_path=keyframe_path,
            mode="final"
        )

    movie_assembler.assemble_full_movie(project, mode="final")
    return project


# Bulletproof Studio UI Static Delivery
@app.get("/", response_class=HTMLResponse)
def serve_studio_ui():
    """Serves the integrated single-page Cinema Studio Dashboard across any working directory."""
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend", "index.html"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "index.html"),
        "../frontend/index.html",
        "frontend/index.html",
        "/app/frontend/index.html"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())

    return HTMLResponse("<h2>AI Hindi Cinema Studio UI loading error: index.html not found.</h2>")
