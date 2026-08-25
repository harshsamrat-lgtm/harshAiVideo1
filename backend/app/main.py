"""
AI Hindi Cinema Studio - FastAPI Application Server (User-Selectable Multi-Model Edition).
"""

import os
import sys
import subprocess
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
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
    title="AI Hindi Cinema Studio API",
    description="Multi-Model Hindi Story-to-Movie Generation Platform with User Model Selection",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

app.mount("/media", StaticFiles(directory="media_store"), name="media")

PROJECTS_DB: Dict[str, ProjectState] = {}

director_engine = StoryDirectorEngine()
image_studio = ImageStudioEngine()
voice_engine = VoiceStudioEngine("media_store/audio")
video_studio = VideoStudioEngine(videos_dir="media_store/videos")
movie_assembler = MovieAssemblerEngine("media_store/movies")


@app.get("/api/system/status")
def get_system_status():
    comfy_online = video_studio._check_comfyui_online()
    return {
        "status": "online",
        "available_models": [
            {"id": "wan2_1_14b", "name": "🌟 Wan2.1-14B (Alibaba Flagship Master 1080p Video)", "size": "28 GB"},
            {"id": "minimax_h3", "name": "🎬 MiniMax H3 (Hailuo 3.0 15s Native Ref2VA)", "size": "18.5 GB"},
            {"id": "wan2_1_1_3b", "name": "⚡ Wan2.1-1.3B (Ultra-Fast 720p Diffusion)", "size": "3.2 GB"},
            {"id": "svd_xt", "name": "🚀 Stable Video Diffusion XT (Native GPU)", "size": "4.5 GB"},
            {"id": "cloud_diffusion", "name": "🌐 Cloud AI Video Diffusion (Zero-Wait)", "size": "Cloud"}
        ],
        "comfyui_gpu_connected": comfy_online,
        "active_projects": len(PROJECTS_DB)
    }


@app.post("/api/system/restart")
async def restart_system_engine():
    global PROJECTS_DB
    PROJECTS_DB.clear()
    return {"status": "restarted", "message": "ऐप और GPU इंजन सफलतापूर्वक रीस्टार्ट हो गया है।"}


@app.post("/api/story/analyze", response_model=ProjectState)
async def analyze_hindi_story(request: StoryInputRequest):
    project = director_engine.parse_story(request)

    # 1. Location 4K Concept Art
    for loc in project.locations:
        await image_studio.generate_location_concept_art(loc)

    # 2. Character 360 Portraits
    for char in project.characters:
        await image_studio.generate_character_portrait_sheet(char)

    # 3. Keyframes & Voice
    for sc in project.scenes:
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])
        sc.composite_keyframe_url = await image_studio.generate_composite_scene_keyframe(
            sc.scene_number, loc, char, sc.visual_prompt
        )
        if sc.dialogue:
            await voice_engine.generate_character_dialogue(sc.dialogue, char)

    PROJECTS_DB[project.project_id] = project
    return project


@app.post("/api/render/draft/{project_id}")
async def generate_draft_movie(
    project_id: str,
    video_model: str = Query(default="wan2_1_14b", description="Chosen AI Video Model")
):
    if project_id not in PROJECTS_DB:
        raise HTTPException(status_code=404, detail="Project not found")

    project = PROJECTS_DB[project_id]
    project.status = "drafting"

    print(f"\n[API] 🎬 Generating Draft Movie with Selected Model: {video_model}...")

    for sc in project.scenes:
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])

        await video_studio.generate_15s_scene_video(
            scene=sc,
            character=char,
            location=loc,
            composite_keyframe_url=sc.composite_keyframe_url,
            mode="draft",
            selected_model=video_model
        )

    movie_assembler.assemble_full_movie(project, mode="draft")
    return project


@app.post("/api/scene/{project_id}/regenerate/{scene_number}")
async def regenerate_single_scene(
    project_id: str,
    scene_number: int,
    custom_prompt: Optional[str] = None,
    video_model: str = Query(default="wan2_1_14b")
):
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

    scene.composite_keyframe_url = await image_studio.generate_composite_scene_keyframe(
        scene.scene_number, loc, char, scene.visual_prompt
    )

    await video_studio.generate_15s_scene_video(
        scene=scene,
        character=char,
        location=loc,
        composite_keyframe_url=scene.composite_keyframe_url,
        mode="draft",
        selected_model=video_model
    )

    movie_assembler.assemble_full_movie(project, mode="draft")
    return scene


@app.post("/api/render/approve-final/{project_id}")
async def approve_and_render_final_master(
    project_id: str,
    video_model: str = Query(default="wan2_1_14b")
):
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
            mode="final",
            selected_model=video_model
        )

    movie_assembler.assemble_full_movie(project, mode="final")
    return project


@app.get("/api/project/{project_id}", response_model=ProjectState)
def get_project(project_id: str):
    if project_id not in PROJECTS_DB:
        raise HTTPException(status_code=404, detail="Project not found")
    return PROJECTS_DB[project_id]


@app.get("/", response_class=HTMLResponse)
def serve_studio_ui():
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
    return HTMLResponse("<h2>AI Hindi Cinema Studio UI loading error</h2>")
