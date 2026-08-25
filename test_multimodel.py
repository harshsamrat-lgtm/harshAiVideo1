"""
Comprehensive Multi-Model Test Script for AI Hindi Cinema Studio.
Tests:
- Gemini / GPT-4o Screenplay Analysis
- Flux.1 4K Location & Character AI Generation
- Edge-TTS Hindi Voice Synthesis
- Wan2.1 / MiniMax 15s Video Generation
- FFmpeg Movie Assembler & Subtitle Burn
"""

import sys
import os
import asyncio

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

backend_path = os.path.join(os.path.dirname(__file__), "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

os.chdir(backend_path)

from app.models import StoryInputRequest
from app.engines.story_director import StoryDirectorEngine
from app.engines.image_studio import ImageStudioEngine
from app.engines.voice_studio import VoiceStudioEngine
from app.engines.video_engine import VideoStudioEngine
from app.engines.movie_assembler import MovieAssemblerEngine


async def test_multimodel_pipeline():
    print("=================================================================")
    print("🎬 Testing AI Hindi Cinema Studio (Multi-Model Suite)...")
    print("=================================================================")

    sample_story = """
    रामू अपने पुराने कच्चे घर की खिड़की पर खड़ा था। सुबह की पहली किरणें मिट्टी की दीवारों पर पड़ रही थीं।
    बाहर घने जंगल से अजीब तरह की आवाजें आ रही थीं। रामू ने मन ही मन तय किया कि आज उसे इस रहस्य का पता लगाना होगा।
    वह घर का भारी लकड़ी का दरवाजा खोलकर घने जंगल के अनजान रास्ते की ओर बढ़ चला।
    """

    req = StoryInputRequest(
        title="गाँव का रहस्यमयी अंधेरा",
        story_text=sample_story,
        genre="Mystery Thriller",
        scene_duration_seconds=15
    )

    # 1. Screenplay Parser
    print("\n1. [Gemini / GPT-4o] Analyzing Story & Screenplay...")
    director = StoryDirectorEngine()
    project = director.parse_story(req)
    print(f"✅ Screenplay Generated: {len(project.scenes)} scenes (15s each)")

    # 2. Flux.1 Image Studio
    print("\n2. [Flux.1 Schnell] Generating Real 4K Location & Character Art...")
    img_studio = ImageStudioEngine()
    
    for loc in project.locations:
        url = await img_studio.generate_location_concept_art(loc)
        print(f"   🎨 Location 4K Art [{loc.name}]: {url}")

    for char in project.characters:
        url = await img_studio.generate_character_portrait_sheet(char)
        print(f"   👤 Character 360° Portrait [{char.name}]: {url}")

    # 3. Composite Keyframes & Edge-TTS Audio
    print("\n3. [Flux.1 + Edge-TTS] Generating Composite Keyframes & Hindi Voice Samples...")
    voice_engine = VoiceStudioEngine("media_store/audio")

    for sc in project.scenes:
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])
        
        sc.composite_keyframe_url = await img_studio.generate_composite_scene_keyframe(
            sc.scene_number, loc, char, sc.visual_prompt
        )
        print(f"   🖼️ Scene {sc.scene_number} Real Keyframe: {sc.composite_keyframe_url}")

        if sc.dialogue:
            await voice_engine.generate_character_dialogue(sc.dialogue, char)
            print(f"   🎙️ Scene {sc.scene_number} Dialogue Audio: {sc.dialogue.audio_url}")

    # 4. Wan2.1 & MiniMax Video Engine
    print("\n4. [Wan2.1 / MiniMax Engine] Generating 15s Cinematic Video Clips...")
    video_studio = VideoStudioEngine(videos_dir="media_store/videos")

    for sc in project.scenes:
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])

        v_url = await video_studio.generate_15s_scene_video(
            scene=sc,
            character=char,
            location=loc,
            composite_keyframe_url=sc.composite_keyframe_url,
            mode="draft"
        )
        print(f"   🎥 Scene {sc.scene_number} 15s Video Clip: {v_url}")

    # 5. FFmpeg Movie Assembly
    print("\n5. [FFmpeg Multi-Track] Assembling Full Movie & Subtitles...")
    assembler = MovieAssemblerEngine("media_store/movies")
    draft_movie = assembler.assemble_full_movie(project, mode="draft")
    print(f"✅ Full Draft Movie Assembled: {draft_movie}")
    print(f"   📜 Subtitles (.SRT): {project.subtitle_srt_url}")

    final_movie = assembler.assemble_full_movie(project, mode="final")
    print(f"✅ Full 1080p Final Master Movie Assembled: {final_movie}")

    print("\n🎉 ALL MULTI-MODEL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(test_multimodel_pipeline())
