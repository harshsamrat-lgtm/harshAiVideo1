"""
AI Hindi Cinema Studio - FastAPI Application Server.
Provides high-speed parallel Draft & Final Movie rendering with 100% audio-visual synchronization.
Features: Progress tracking, proper error handling, model routing, scene management.
"""

import os
import sys
import random
import asyncio
import traceback
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.models import (
    StoryInputRequest, ProjectState, SceneModel, CharacterModel,
    LocationModel, ErrorResponse, VideoModelChoice
)
from app.engines.story_director import StoryDirectorEngine
from app.engines.image_studio import ImageStudioEngine
from app.engines.voice_studio import VoiceStudioEngine
from app.engines.video_engine import VideoStudioEngine
from app.engines.movie_assembler import MovieAssemblerEngine

app = FastAPI(
    title="AI Hindi Cinema Studio API",
    description="Multi-Model Hindi Story-to-Movie Generation Platform — दुनिया का सबसे उत्तम AI वीडियो जनरेशन ऐप",
    version="3.0.0"
)

# CORS — configurable origins
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure all media directories exist
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

# In-memory project store
PROJECTS_DB: Dict[str, ProjectState] = {}

# Engine instances
director_engine = StoryDirectorEngine()
image_studio = ImageStudioEngine()
voice_engine = VoiceStudioEngine("media_store/audio")
video_studio = VideoStudioEngine(videos_dir="media_store/videos")
movie_assembler = MovieAssemblerEngine("media_store/movies")


# ─── Utility ──────────────────────────────────────────────────────────────────

def _log_exceptions(results: list, context: str = ""):
    """Logs any exceptions from asyncio.gather results."""
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[{context}] ❌ Task {i} failed: {result}")
            traceback.print_exception(type(result), result, result.__traceback__)


def _get_project(project_id: str) -> ProjectState:
    """Retrieves project or raises 404."""
    if project_id not in PROJECTS_DB:
        raise HTTPException(status_code=404, detail=f"प्रोजेक्ट '{project_id}' नहीं मिला। कृपया पहले कहानी विश्लेषित करें।")
    return PROJECTS_DB[project_id]


# ─── System APIs ──────────────────────────────────────────────────────────────

@app.get("/api/system/status")
def get_system_status():
    comfy_online = video_studio._check_comfyui_online()
    return {
        "status": "online",
        "version": "3.0.0",
        "available_models": [
            {"id": "wan2_1_14b", "name": "🌟 Wan2.1-14B (Alibaba Flagship Master 1080p Video)", "size": "28 GB", "requires_gpu": True},
            {"id": "minimax_h3", "name": "🎬 MiniMax H3 (Hailuo 3.0 15s Native Ref2VA)", "size": "18.5 GB", "requires_gpu": True},
            {"id": "wan2_1_1_3b", "name": "⚡ Wan2.1-1.3B (Ultra-Fast 720p Diffusion)", "size": "3.2 GB", "requires_gpu": True},
            {"id": "svd_xt", "name": "🚀 Stable Video Diffusion XT (Native GPU)", "size": "4.5 GB", "requires_gpu": True},
            {"id": "cloud_diffusion", "name": "🌐 Cloud Neural Video Diffusion (Instant 0-Wait)", "size": "Cloud", "requires_gpu": False}
        ],
        "comfyui_gpu_connected": comfy_online,
        "active_projects": len(PROJECTS_DB),
        "media_dirs_ready": all(os.path.isdir(d) for d in MEDIA_DIRS)
    }


@app.post("/api/system/restart")
async def restart_system_engine():
    global PROJECTS_DB
    PROJECTS_DB.clear()
    return {"status": "restarted", "message": "ऐप और GPU इंजन सफलतापूर्वक रीस्टार्ट हो गया है। सभी प्रोजेक्ट्स साफ हो गए हैं।"}


# ─── Story Analysis ──────────────────────────────────────────────────────────

@app.post("/api/story/analyze", response_model=ProjectState)
async def analyze_hindi_story(request: StoryInputRequest):
    """
    Parses Hindi story into context-matched screenplay and generates fresh 4K AI Assets in PARALLEL.
    """
    try:
        project = director_engine.parse_story(request)
        project.status = "analyzing"
        project.progress_message = "कहानी का विश्लेषण हो रहा है..."
        project.progress_percent = 10
        PROJECTS_DB[project.project_id] = project

        # 1. Locations and Characters in Parallel
        asset_tasks = []
        for loc in project.locations:
            asset_tasks.append(image_studio.generate_location_concept_art(loc, force_refresh=True))
        for char in project.characters:
            asset_tasks.append(image_studio.generate_character_portrait_sheet(char, force_refresh=True))

        project.progress_message = "लोकेशन और कलाकारों की 4K तस्वीरें बनाई जा रही हैं..."
        project.progress_percent = 30

        results = await asyncio.gather(*asset_tasks, return_exceptions=True)
        _log_exceptions(results, "Asset Generation")

        # 2. Scene Keyframes and Audio Dialogues in Parallel
        project.progress_message = "सीन कीफ्रेम और डायलॉग ऑडियो तैयार हो रहे हैं..."
        project.progress_percent = 60

        scene_tasks = []
        for sc in project.scenes:
            loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
            char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])

            async def process_scene(s=sc, l=loc, c=char):
                s.composite_keyframe_url = await image_studio.generate_composite_scene_keyframe(
                    s.scene_number, l, c, s.visual_prompt, force_refresh=True
                )
                if s.dialogue:
                    await voice_engine.generate_character_dialogue(s.dialogue, c)

            scene_tasks.append(process_scene())

        results = await asyncio.gather(*scene_tasks, return_exceptions=True)
        _log_exceptions(results, "Scene Processing")

        project.status = "analyzed"
        project.progress_percent = 100
        project.progress_message = "सभी AI तस्वीरें, चेहरे और आवाज़ तैयार हैं!"

        print(f"[Analyze API] ✅ Screenplay & All Assets ready: {project.project_id} ({len(project.scenes)} scenes)")
        return project

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Analyze API] ❌ Error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"कहानी विश्लेषण में त्रुटि: {str(e)}। कृपया पुनः प्रयास करें।"
        )


# ─── Character Management ────────────────────────────────────────────────────

@app.post("/api/character/{project_id}/regenerate/{character_id}")
async def regenerate_character_face(
    project_id: str, character_id: str, appearance: Optional[str] = None
):
    project = _get_project(project_id)
    char = next((c for c in project.characters if c.character_id == character_id), None)
    if not char:
        raise HTTPException(status_code=404, detail="कलाकार नहीं मिला")

    if appearance:
        char.appearance = appearance
    char.seed = random.randint(10000, 99999)

    await image_studio.generate_character_portrait_sheet(char, force_refresh=True)

    # Update all scenes featuring this character
    update_tasks = []
    for sc in project.scenes:
        if char.character_id in sc.character_ids:
            loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])

            async def update_scene(s=sc, l=loc, c=char):
                s.composite_keyframe_url = await image_studio.generate_composite_scene_keyframe(
                    s.scene_number, l, c, s.visual_prompt, force_refresh=True
                )

            update_tasks.append(update_scene())

    if update_tasks:
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        _log_exceptions(results, "Character Re-roll Scene Update")

    return char


# ─── Location Management ─────────────────────────────────────────────────────

@app.post("/api/location/{project_id}/regenerate/{location_id}")
async def regenerate_location_art(
    project_id: str, location_id: str, custom_style: Optional[str] = None
):
    project = _get_project(project_id)
    loc = next((l for l in project.locations if l.location_id == location_id), None)
    if not loc:
        raise HTTPException(status_code=404, detail="लोकेशन नहीं मिला")

    if custom_style:
        loc.architecture_style = custom_style
    loc.seed = random.randint(10000, 99999)

    await image_studio.generate_location_concept_art(loc, force_refresh=True)

    # Update all scenes using this location
    update_tasks = []
    for sc in project.scenes:
        if sc.location_id == loc.location_id:
            char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])

            async def update_scene(s=sc, l=loc, c=char):
                s.composite_keyframe_url = await image_studio.generate_composite_scene_keyframe(
                    s.scene_number, l, c, s.visual_prompt, force_refresh=True
                )

            update_tasks.append(update_scene())

    if update_tasks:
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        _log_exceptions(results, "Location Re-roll Scene Update")

    return loc


# ─── Draft Movie Rendering ───────────────────────────────────────────────────

@app.post("/api/render/draft/{project_id}")
async def generate_draft_movie(
    project_id: str,
    video_model: str = Query(default="wan2_1_14b", description="Chosen AI Video Model")
):
    """
    Renders all 15s scenes in PARALLEL and stitches into unified movie with Audio.
    """
    project = _get_project(project_id)
    project.status = "drafting"
    project.progress_percent = 0
    project.progress_message = f"[{video_model.upper()}] वीडियो सीन्स रेंडर हो रहे हैं..."

    # Parallel scene rendering
    video_tasks = []
    for idx, sc in enumerate(project.scenes):
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])

        video_tasks.append(video_studio.generate_15s_scene_video(
            scene=sc,
            character=char,
            location=loc,
            composite_keyframe_url=sc.composite_keyframe_url or "",
            mode="draft",
            selected_model=video_model
        ))

    results = await asyncio.gather(*video_tasks, return_exceptions=True)
    _log_exceptions(results, "Draft Video Rendering")

    project.progress_percent = 80
    project.progress_message = "सभी सीन रेंडर हो गए, मूवी असेंबल हो रही है..."

    movie_assembler.assemble_full_movie(project, mode="draft")

    project.progress_percent = 100
    project.progress_message = "ड्राफ्ट मूवी तैयार है!"

    print(f"[Draft API] ✅ Full draft movie rendered: {project.full_draft_movie_url}")
    return project


# ─── Scene Management ─────────────────────────────────────────────────────────

@app.post("/api/scene/{project_id}/regenerate/{scene_number}")
async def regenerate_single_scene(
    project_id: str,
    scene_number: int,
    custom_prompt: Optional[str] = None,
    video_model: str = Query(default="wan2_1_14b")
):
    project = _get_project(project_id)
    scene = next((s for s in project.scenes if s.scene_number == scene_number), None)
    if not scene:
        raise HTTPException(status_code=404, detail=f"सीन {scene_number} नहीं मिला")

    if custom_prompt:
        scene.visual_prompt = custom_prompt

    loc = next((l for l in project.locations if l.location_id == scene.location_id), project.locations[0])
    char = next((c for c in project.characters if c.character_id in scene.character_ids), project.characters[0])

    scene.composite_keyframe_url = await image_studio.generate_composite_scene_keyframe(
        scene.scene_number, loc, char, scene.visual_prompt, force_refresh=True
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


@app.post("/api/scene/{project_id}/reorder")
async def reorder_scenes(project_id: str, new_order: list[int]):
    """Reorders scenes based on provided scene number order."""
    project = _get_project(project_id)

    # Validate all scene numbers exist
    existing_nums = {s.scene_number for s in project.scenes}
    if set(new_order) != existing_nums:
        raise HTTPException(status_code=400, detail="अमान्य सीन क्रम — सभी मौजूदा सीन नंबर शामिल होने चाहिए")

    # Reorder
    scene_map = {s.scene_number: s for s in project.scenes}
    project.scenes = [scene_map[num] for num in new_order]

    # Renumber
    for idx, scene in enumerate(project.scenes):
        scene.scene_number = idx + 1

    return {"message": "सीन क्रम सफलतापूर्वक बदल दिया गया", "scene_count": len(project.scenes)}


# ─── Final Master Rendering ──────────────────────────────────────────────────

@app.post("/api/render/approve-final/{project_id}")
async def approve_and_render_final_master(
    project_id: str,
    video_model: str = Query(default="wan2_1_14b")
):
    project = _get_project(project_id)
    project.status = "rendering_final"
    project.progress_percent = 0
    project.progress_message = f"[{video_model.upper()}] 1080p फाइनल मास्टर रेंडर हो रहा है..."

    video_tasks = []
    for sc in project.scenes:
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])

        video_tasks.append(video_studio.generate_15s_scene_video(
            scene=sc,
            character=char,
            location=loc,
            composite_keyframe_url=sc.composite_keyframe_url or "",
            mode="final",
            selected_model=video_model
        ))

    results = await asyncio.gather(*video_tasks, return_exceptions=True)
    _log_exceptions(results, "Final Video Rendering")

    project.progress_percent = 85
    project.progress_message = "फाइनल मूवी असेंबल हो रही है..."

    movie_assembler.assemble_full_movie(project, mode="final")

    project.progress_percent = 100
    project.progress_message = "1080p फाइनल मास्टर मूवी तैयार है!"

    return project


# ─── Project Retrieval ────────────────────────────────────────────────────────

@app.get("/api/project/{project_id}", response_model=ProjectState)
def get_project(project_id: str):
    return _get_project(project_id)


@app.get("/api/project/{project_id}/progress")
def get_project_progress(project_id: str):
    project = _get_project(project_id)
    return {
        "project_id": project.project_id,
        "status": project.status,
        "progress_percent": project.progress_percent,
        "progress_message": project.progress_message,
        "scenes_ready": sum(1 for s in project.scenes if s.status == "ready"),
        "total_scenes": len(project.scenes)
    }


# ─── Frontend UI Serving ─────────────────────────────────────────────────────

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
    return HTMLResponse(
        "<h2 style='color:#ffc400;font-family:sans-serif;text-align:center;margin-top:100px;'>"
        "🎬 AI Hindi Cinema Studio — UI Loading Error<br>"
        "<small style='color:#94a3b8;'>frontend/index.html not found. Please check your file structure.</small>"
        "</h2>"
    )
