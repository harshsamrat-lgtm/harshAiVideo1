"""
AI Story Director Engine.
Parses raw Hindi stories into structured Screenplays with Characters, Locations, and 15-second Max Native Scenes.
"""

import os
import json
import uuid
import re
from typing import Dict, Any, List
from app.models import StoryInputRequest, ProjectState, CharacterModel, LocationModel, SceneModel, DialogueModel


class StoryDirectorEngine:
    """
    Intelligent Story Director.
    Extracts screenplay metadata, builds consistent Character & Location profiles,
    and splits the narrative into structured 15-second cinematic scenes (MiniMax H3 Max Native Limit).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    def parse_story(self, request: StoryInputRequest) -> ProjectState:
        project_id = str(uuid.uuid4())[:8]
        story_text = request.story_text.strip()
        genre = request.genre
        visual_style = request.visual_style
        scene_duration = request.scene_duration_seconds or 15

        # Heuristic analysis or LLM-driven parsing
        characters, locations, scenes = self._extract_screenplay(story_text, genre, visual_style, scene_duration)

        return ProjectState(
            project_id=project_id,
            title=request.title or "हिंदी सिनेमैटिक कहानी",
            story_text=story_text,
            genre=genre,
            visual_style=visual_style,
            scene_duration_seconds=scene_duration,
            status="analyzed",
            characters=characters,
            locations=locations,
            scenes=scenes
        )

    def _extract_screenplay(self, text: str, genre: str, visual_style: str, duration: int = 15):
        """
        Parses the story sentences and context to build realistic characters,
        persistent location anchors, and 15-second cinematic scenes.
        """
        # Split into logical narrative chunks (paragraphs / sentences)
        raw_sentences = [s.strip() for s in re.split(r'[।\.\n]+', text) if len(s.strip()) > 5]
        if not raw_sentences:
            raw_sentences = ["एक रोमांचक और रहस्यमयी यात्रा की शुरुआत।", "नायक अपने लक्ष्य की ओर बढ़ता है।"]

        # 1. Identify Characters
        characters: List[CharacterModel] = []
        character_names = self._detect_character_names(text)
        if not character_names:
            character_names = ["नायक (Lead Actor)", "सह-कलाकार (Supporting Lead)"]

        voice_presets = [
            ("hi-IN-MadhurNeural", "Male", 28, "तीखे नैन-नक्श, मध्यम कद, गहरी आँखें, आत्मविश्वास भरा चेहरा", "सफेद सूती कुर्ता और पायजामा"),
            ("hi-IN-SwaraNeural", "Female", 24, "सुंदर भावपूर्ण आँखें, सादगी भरा चेहरा, लंबे काले बाल", "पारंपरिक भारतीय परिधान"),
            ("hi-IN-GulNeural", "Female", 32, "गंभीर एवं परिपक्व व्यक्तित्व, राजसी रूप", "आधुनिक सिल्क पोशाक"),
            ("hi-IN-NilNeural", "Male", 45, "गहरा रौबदार चेहरा, घनी मूंछें, अनुभवी नज़रें", "गहरा विंटेज कोट")
        ]

        for idx, name in enumerate(character_names[:4]):
            preset = voice_presets[idx % len(voice_presets)]
            char_id = f"CHAR_{idx+1}_{name.replace(' ', '_')}"
            characters.append(CharacterModel(
                character_id=char_id,
                name=name,
                gender=preset[1],
                age=preset[2],
                appearance=preset[3],
                costume=preset[4],
                voice_profile=preset[0],
                voice_pitch="+0Hz",
                voice_rate="+0%",
                seed=1000 + (idx * 37)
            ))

        # 2. Identify Locations (Environment Anchors)
        locations: List[LocationModel] = []
        location_names = self._detect_locations(text, genre)
        
        loc_palette_map = {
            "ग्रामीण / कच्चा घर": ("Rustic Indian Clay Hut", ["मिट्टी की दीवारें", "लकड़ी की नक्काशीदार खिड़की", "पीतल का जलता लालटेन", "पुरानी खाट"], "Warm Amber & Terracotta", "Dramatic Interior Sunbeams"),
            "घना जंगल": ("Mysterious Dense Forest", ["विशाल बरगद के पेड़", "जमीन पर छाया कोहरा", "पुराने पत्थरों का रास्ता"], "Cold Indigo & Moonlight Blue", "Volumetric Foggy Night Light"),
            "शाही महल / दरबार": ("Grand Royal Indian Palace", ["सफेद संगमरमर के खंभे", "सोने के नक्काशीदार दीप", "लाल कालीन"], "Rich Crimson & Royal Gold", "Cinematic Golden Hour Chandelier Glow"),
            "आधुनिक शहर": ("Modern Urban Metropolis", ["ऊंची कांच की इमारतें", "सड़क पर जलती नियॉन लाइट्स", "भीड़"], "Neon Cyan & Wet Asphalt Reflection", "Rainy Cyberpunk Night Lights")
        }

        for idx, loc_name in enumerate(location_names):
            loc_id = f"LOC_{idx+1}"
            match_key = "ग्रामीण / कच्चा घर"
            for k in loc_palette_map:
                if k in loc_name:
                    match_key = k
                    break
            
            style, props, palette, lighting = loc_palette_map.get(match_key, loc_palette_map["ग्रामीण / कच्चा घर"])

            locations.append(LocationModel(
                location_id=loc_id,
                name=loc_name,
                description=f"{loc_name} का प्रामाणिक एवं सिनेमाई परिवेश",
                architecture_style=style,
                anchor_props=props,
                color_palette=palette,
                lighting_scheme=lighting,
                seed=2000 + (idx * 59)
            ))

        # 3. Create 15-Second Scenes (MiniMax H3 Max Native Duration)
        scenes: List[SceneModel] = []
        camera_moves = [
            "Cinematic 15s slow tracking shot moving gracefully forward through the environment",
            "Atmospheric 15s low-angle crane shot sweeping across the dramatic architecture",
            "Emotional 15s tight close-up shot capturing character subtle expressions and dialogue",
            "Wide 15s establishing drone shot revealing the breathtaking depth of the location",
            "Slow 15s panning Dutch angle shot building intense cinematic suspense and atmosphere"
        ]

        lighting_moods = [
            "Golden hour warm volumetric sunbeams cutting through atmospheric haze",
            "Deep dramatic chiaroscuro lighting with high contrast shadows",
            "Cool atmospheric midnight moonlight casting long blue shadows",
            "Ethereal soft diffused cinematic light with rich color grading"
        ]

        bgm_moods = [
            "Intense Indian cinematic strings and deep orchestral suspense",
            "Slow emotional flute melody with ethereal ambient drones",
            "Fast-paced heart-thumping cinematic percussion and brass",
            "Dark mysterious sitar and cello harmonics building tension"
        ]

        sfx_pools = [
            ["Distant wind howling", "Creaking wooden frame", "Soft footsteps on earth"],
            ["Chirping forest night crickets", "Leaves rustling in gentle breeze", "Subtle water drop echo"],
            ["Grand palace hall resonance", "Silk rustling", "Heavy brass gate opening sound"],
            ["Distant city rain on asphalt", "Soft breath inhale", "Thunder rumble far away"]
        ]

        for idx, chunk in enumerate(raw_sentences):
            scene_num = idx + 1
            cur_loc = locations[idx % len(locations)]
            cur_char = characters[idx % len(characters)]
            cam_move = camera_moves[idx % len(camera_moves)]
            light_mood = lighting_moods[idx % len(lighting_moods)]
            bgm = bgm_moods[idx % len(bgm_moods)]
            sfx = sfx_pools[idx % len(sfx_pools)]

            # Formulate detailed MiniMax H3 prompt for 15s continuous shot
            visual_prompt = (
                f"Masterpiece 8K 15-second cinematic shot. {cam_move}. "
                f"Subject: {cur_char.name} ({cur_char.appearance}, wearing {cur_char.costume}). "
                f"Environment: {cur_loc.name} with {cur_loc.architecture_style}, featuring {', '.join(cur_loc.anchor_props[:2])}. "
                f"Lighting: {light_mood}. Palette: {cur_loc.color_palette}. "
                f"Context: {chunk}. Ultra-photorealistic, 35mm anamorphic lens, IMAX quality, smooth continuous motion."
            )

            dialogue = DialogueModel(
                character_id=cur_char.character_id,
                character_name=cur_char.name,
                text=chunk if len(chunk) < 110 else chunk[:105] + "...",
                emotion="intense_cinematic",
                duration_seconds=float(min(duration, max(4, len(chunk) // 8)))
            )

            scenes.append(SceneModel(
                scene_number=scene_num,
                duration_seconds=duration,
                location_id=cur_loc.location_id,
                location_name=cur_loc.name,
                character_ids=[cur_char.character_id],
                camera_movement=cam_move,
                lighting_mood=light_mood,
                visual_prompt=visual_prompt,
                negative_prompt="blurry, distorted face, low quality, morphing artifacts, extra fingers, cartoon, 3d render look",
                dialogue=dialogue,
                sfx=sfx,
                bgm_mood=bgm,
                status="pending"
            ))

        return characters, locations, scenes

    def _detect_character_names(self, text: str) -> List[str]:
        known = ["रामू", "रोहन", "विक्रम", "राघव", "सीमा", "प्रिया", "अनन्या", "अर्जुन", "माया", "ठाकुर", "राजा", "साधु"]
        found = []
        for name in known:
            if name in text and name not in found:
                found.append(name)
        if not found:
            found = ["रामू (नायक)", "सीमा (नायिका)"]
        return found

    def _detect_locations(self, text: str, genre: str) -> List[str]:
        locs = []
        if "घर" in text or "कमरा" in text or "गांव" in text:
            locs.append("गाँव का पुराना कच्चा घर")
        if "जंगल" in text or "पेड़" in text or "रास्ता" in text or "रात" in text:
            locs.append("घना रहस्यमयी जंगल")
        if "महल" in text or "किला" in text or "दरबार" in text:
            locs.append("शाही महल का भव्य दरबार")
        if "शहर" in text or "सड़क" in text or "दुकान" in text:
            locs.append("शहर का पुराना बाजार")

        if not locs:
            locs = ["गाँव का पुराना कच्चा घर", "घना रहस्यमयी जंगल"]
        return locs
