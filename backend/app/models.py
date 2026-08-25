"""
Data models and schemas for AI Hindi Cinema Studio.
Manages Story Analysis, Characters, Locations, 15s Max Native Scenes, and Movie Projects.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class StoryInputRequest(BaseModel):
    title: Optional[str] = Field(default="मेरी हिंदी कहानी", description="कहानी का शीर्षक")
    story_text: str = Field(..., description="उपयोगकर्ता द्वारा इनपुट की गई हिंदी कहानी का कच्चा पाठ")
    genre: str = Field(default="Cinematic Drama", description="कहानी का जॉनर (उदा. Thriller, Drama, Horror, Historical, Romance, Sci-Fi)")
    visual_style: str = Field(default="Bollywood Cinematic Realism, 8K, Volumetric Lighting, Masterful Composition", description="दृश्य शैली")
    scene_duration_seconds: int = Field(default=15, description="प्रत्येक दृश्य की अवधि - MiniMax H3 की अधिकतम नेटिव अवधि 15 सेकंड")


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
    composite_keyframe_url: Optional[str] = None
    dialogue: Optional[DialogueModel] = None
    sfx: List[str] = []
    bgm_mood: str = "Dramatic Suspense"
    draft_video_url: Optional[str] = None
    final_video_url: Optional[str] = None
    status: str = "pending"  # pending, generating, ready, failed


class ProjectState(BaseModel):
    project_id: str
    title: str
    story_text: str
    genre: str
    visual_style: str
    scene_duration_seconds: int = 15
    status: str = "created"  # created, analyzing, drafted, draft_ready, approved, rendering_final, completed
    characters: List[CharacterModel] = []
    locations: List[LocationModel] = []
    scenes: List[SceneModel] = []
    full_draft_movie_url: Optional[str] = None
    full_final_movie_url: Optional[str] = None
    subtitle_srt_url: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
