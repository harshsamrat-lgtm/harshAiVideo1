"""
Comprehensive Test Script for AI Hindi Cinema Studio Pipeline.
Tests Story Analysis, Location DNA, Character Consistency, 10s Scene Generation, and Movie Stitching.
"""

import sys
import os
import asyncio

# Configure UTF-8 output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add backend directory to sys.path
backend_path = os.path.join(os.path.dirname(__file__), "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

os.chdir(backend_path)

from app.models import StoryInputRequest
from app.engines.story_director import StoryDirectorEngine
from app.engines.location_manager import LocationManager
from app.engines.character_manager import CharacterManager
from app.engines.voice_studio import VoiceStudioEngine
from app.engines.minimax_h3_engine import MiniMaxH3Engine
from app.engines.movie_assembler import MovieAssemblerEngine


async def test_full_movie_generation_pipeline():
    print("[TEST] Starting AI Hindi Cinema Studio Pipeline Test...")

    sample_story = """
    रामू अपने पुराने कच्चे घर की खिड़की पर खड़ा था। सुबह की पहली किरणें मिट्टी की दीवारों पर पड़ रही थीं।
    बाहर घने जंगल से अजीब तरह की आवाजें आ रही थीं। रामू ने मन ही मन तय किया कि आज उसे इस रहस्य का पता लगाना होगा।
    वह घर का भारी लकड़ी का दरवाजा खोलकर घने जंगल के अनजान रास्ते की ओर बढ़ चला।
    """

    req = StoryInputRequest(
        title="गाँव का रहस्यमयी अंधेरा",
        story_text=sample_story,
        genre="Mystery Thriller",
        scene_duration_seconds=10
    )

    # 1. Story Director Test
    print("\n1. Testing AI Story Director (Screenplay Analysis)...")
    director = StoryDirectorEngine()
    project = director.parse_story(req)
    print(f"[OK] Screenplay Generated: {len(project.scenes)} scenes (10s each)")
    print(f"   - Characters detected: {[c.name for c in project.characters]}")
    print(f"   - Locations detected: {[l.name for l in project.locations]}")

    # 2. Location DNA & Character Consistency Test
    print("\n2. Testing Location DNA & Character Master Sheets...")
    loc_mgr = LocationManager("media_store/locations")
    char_mgr = CharacterManager("media_store/characters")

    for loc in project.locations:
        url = loc_mgr.generate_master_establishing_shot(loc)
        print(f"   - Location Asset [{loc.name}]: {url}")

    for char in project.characters:
        url = char_mgr.generate_master_character_sheet(char)
        print(f"   - Character Turnaround [{char.name}]: {url}")

    # 3. Composite Keyframes Test
    print("\n3. Testing Composite Keyframe Synthesis...")
    for sc in project.scenes:
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])
        sc.composite_keyframe_url = loc_mgr.create_composite_keyframe(
            sc.scene_number, loc, char, sc.lighting_mood
        )
        print(f"   - Scene {sc.scene_number} Keyframe: {sc.composite_keyframe_url}")

    # 4. Voice Studio & 10s Scene Generation Test
    print("\n4. Testing Voice Studio & 10s MiniMax H3 Video Gen...")
    voice_engine = VoiceStudioEngine("media_store/audio")
    video_engine = MiniMaxH3Engine(videos_dir="media_store/videos")

    for sc in project.scenes:
        loc = next((l for l in project.locations if l.location_id == sc.location_id), project.locations[0])
        char = next((c for c in project.characters if c.character_id in sc.character_ids), project.characters[0])

        if sc.dialogue:
            await voice_engine.generate_character_dialogue(sc.dialogue, char)
            print(f"   - Scene {sc.scene_number} Dialogue Audio: {sc.dialogue.audio_url}")

        keyframe_path = sc.composite_keyframe_url.replace("/media/", "media_store/")
        v_url = await video_engine.generate_10s_scene_video(
            scene=sc,
            character=char,
            location=loc,
            composite_keyframe_path=keyframe_path,
            mode="draft"
        )
        print(f"   - Scene {sc.scene_number} 10s Video Clip: {v_url}")

    # 5. Movie Assembly Test (Draft & Final)
    print("\n5. Testing FFmpeg Movie Assembly & Subtitle Burn...")
    assembler = MovieAssemblerEngine("media_store/movies")
    draft_movie = assembler.assemble_full_movie(project, mode="draft")
    print(f"[OK] Full Draft Movie Assembled: {draft_movie}")
    print(f"   - Subtitles (.SRT): {project.subtitle_srt_url}")

    final_movie = assembler.assemble_full_movie(project, mode="final")
    print(f"[OK] Full 1080p Final Master Assembled: {final_movie}")

    print("\n[SUCCESS] ALL PIPELINE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(test_full_movie_generation_pipeline())
