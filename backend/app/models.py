"""
Data models and schemas for AI Hindi Cinema Studio.
Manages Story Analysis, Characters, Locations, 15s Max Native Scenes, and Movie Projects.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class SceneStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"
    GPU_OFFLINE = "gpu_offline"


class ProjectStatus(str, Enum):
    CREATED = "created"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    DRAFTING = "drafting"
    DRAFT_READY = "draft_ready"
    APPROVED = "approved"
    RENDERING_FINAL = "rendering_final"
    COMPLETED = "completed"
    ERROR = "error"


class VideoModelChoice(str, Enum):
    WAN2_1_14B = "wan2_1_14b"
    MINIMAX_H3 = "minimax_h3"
    WAN2_1_1_3B = "wan2_1_1_3b"
    SVD_XT = "svd_xt"
    CLOUD_DIFFUSION = "cloud_diffusion"


# ─── Request / Response Models ────────────────────────────────────────────────

class StoryInputRequest(BaseModel):
    title: Optional[str] = Field(default="मेरी हिंदी कहानी", description="कहानी का शीर्षक")
    story_text: str = Field(..., description="उपयोगकर्ता द्वारा इनपुट की गई हिंदी कहानी का कच्चा पाठ")
    genre: str = Field(default="Cinematic Drama", description="कहानी का जॉनर (उदा. Thriller, Drama, Horror, Historical, Romance, Sci-Fi)")
    visual_style: str = Field(default="Bollywood Cinematic Realism, 8K, Volumetric Lighting, Masterful Composition", description="दृश्य शैली")
    scene_duration_seconds: int = Field(default=15, description="प्रत्येक दृश्य की अवधि - MiniMax H3 की अधिकतम नेटिव अवधि 15 सेकंड")


class ErrorResponse(BaseModel):
    error: str
    detail: str
    suggestion: Optional[str] = None


class ProgressUpdate(BaseModel):
    project_id: str
    step: str
    progress_percent: int = 0
    message: str = ""
    is_complete: bool = False


class ExportSettings(BaseModel):
    resolution: str = "1080p"
    fps: int = 24
    codec: str = "libx264"
    quality_crf: int = 18
    format: str = "mp4"
    include_subtitles: bool = True
    include_bgm: bool = True


# ─── Core Data Models ────────────────────────────────────────────────────────

class CharacterModel(BaseModel):
    character_id: str
    name: str
    gender: str = "Male"
    age: int = 28
    appearance: str
    costume: str
    voice_profile: str = "hi-IN-MadhurNeural"  # Default Hindi neural voice
    voice_pitch: str = "+0Hz"
    voice_rate: str = "+0%"
    voice_emotion: str = "neutral"  # neutral, happy, sad, angry, fearful, excited
    master_portrait_url: Optional[str] = None
    seed: int = 42


class LocationModel(BaseModel):
    location_id: str
    name: str
    description: str
    architecture_style: str
    anchor_props: List[str] = []
    color_palette: str = "Warm Earthy Tones, Amber Shadows"
    lighting_scheme: str = "Atmospheric Cinematic"
    master_establishing_url: Optional[str] = None
    seed: int = 100


class DialogueModel(BaseModel):
    character_id: str
    character_name: str
    text: str
    emotion: str = "neutral"
    audio_url: Optional[str] = None
    duration_seconds: float = 0.0


class SceneModel(BaseModel):
    scene_number: int
    duration_seconds: int = 15  # MiniMax H3 Max Native Limit (15s)
    location_id: str
    location_name: str
    character_ids: List[str] = []
    camera_movement: str
    lighting_mood: str
    visual_prompt: str
    negative_prompt: str = "blurry, low quality, distorted face, extra limbs, cartoonish, low resolution"
    transition_effect: str = "crossfade"  # crossfade, fade_black, cut, dissolve
    composite_keyframe_url: Optional[str] = None
    dialogue: Optional[DialogueModel] = None
    sfx: List[str] = []
    bgm_mood: str = "Dramatic Suspense"
    draft_video_url: Optional[str] = None
    final_video_url: Optional[str] = None
    status: SceneStatus = SceneStatus.PENDING


class ProjectState(BaseModel):
    project_id: str
    title: str
    story_text: str
    genre: str
    visual_style: str
    scene_duration_seconds: int = 15
    status: ProjectStatus = ProjectStatus.CREATED
    progress_percent: int = 0
    progress_message: str = ""
    characters: List[CharacterModel] = []
    locations: List[LocationModel] = []
    scenes: List[SceneModel] = []
    export_settings: ExportSettings = Field(default_factory=ExportSettings)
    full_draft_movie_url: Optional[str] = None
    full_final_movie_url: Optional[str] = None
    subtitle_srt_url: Optional[str] = None
    total_duration_seconds: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
