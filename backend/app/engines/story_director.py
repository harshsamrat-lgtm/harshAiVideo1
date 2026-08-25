"""
AI Story Director Engine (Vivid Action & Context-Aware Prompt Generator).
"""

import os
import uuid
import re
import random
from typing import Dict, Any, List, Tuple
from app.models import StoryInputRequest, ProjectState, CharacterModel, LocationModel, SceneModel, DialogueModel


class StoryDirectorEngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    def parse_story(self, request: StoryInputRequest) -> ProjectState:
        project_id = str(uuid.uuid4())[:8]
        story_text = request.story_text.strip()
        genre = request.genre
        visual_style = request.visual_style
        scene_duration = request.scene_duration_seconds or 15

        characters, locations, scenes = self._extract_screenplay(
            project_id, story_text, genre, visual_style, scene_duration
        )

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

    def _extract_screenplay(
        self, project_id: str, text: str, genre: str, visual_style: str, duration: int = 15
    ) -> Tuple[List[CharacterModel], List[LocationModel], List[SceneModel]]:
        raw_chunks = [s.strip() for s in re.split(r'[।\.\n\?!]+', text) if len(s.strip()) > 6]
        if not raw_chunks:
            raw_chunks = [text.strip() or "कहानी की शुरुआत एक रोमांचक मोड़ से होती है।"]

        # 1. Detect Characters
        detected_names = self._detect_character_names(text)
        characters: List[CharacterModel] = []
        voice_profiles = [
            ("hi-IN-MadhurNeural", "Male", 28, "तीखे नैन-नक्श, मध्यम कद, गहरी आँखें, गंभीर एवं दृढ़ निश्चयी चेहरा", "पारंपरिक सूती सफेद धोती और कुर्ता"),
            ("hi-IN-SwaraNeural", "Female", 24, "सुंदर भावपूर्ण आँखें, सादगी भरा सौम्य चेहरा, लंबे काले बाल", "पारंपरिक भारतीय लाल और सुनहरी साड़ी"),
            ("hi-IN-NilNeural", "Male", 48, "गहरा रौबदार चेहरा, घनी मूंछें, अनुभवी नज़रें, बुलंद व्यक्तित्व", "गहरा विंटेज बंदगला कुर्ता"),
            ("hi-IN-GulNeural", "Female", 32, "आत्मविश्वास से परिपूर्ण, राजसी सौंदर्य, तेज नैन-नक्श", "शाही सिल्क परिधान")
        ]

        for idx, name in enumerate(detected_names):
            preset = voice_profiles[idx % len(voice_profiles)]
            gender = "Female" if any(name.endswith(e) for e in ["ा", "ी", "िया", "ाली", "ाता"]) and name not in ["राजा", "राणा", "रामा", "रामू", "साधु", "बाबा"] else preset[1]
            char_id = f"CHAR_{project_id}_{idx+1}_{name.replace(' ', '_')}"
            characters.append(CharacterModel(
                character_id=char_id,
                name=name,
                gender=gender,
                age=preset[2] + (idx * 4),
                appearance=preset[3],
                costume=preset[4],
                voice_profile="hi-IN-SwaraNeural" if gender == "Female" else "hi-IN-MadhurNeural",
                voice_pitch="+0Hz",
                voice_rate="+0%",
                seed=random.randint(10000, 99999)
            ))

        # 2. Detect Locations
        locations: List[LocationModel] = []
        loc_specs = self._detect_locations_with_metadata(text, genre, project_id)
        for spec in loc_specs:
            locations.append(LocationModel(
                location_id=spec["id"],
                name=spec["name"],
                description=spec["desc"],
                architecture_style=spec["style"],
                anchor_props=spec["props"],
                color_palette=spec["palette"],
                lighting_scheme=spec["lighting"],
                seed=random.randint(10000, 99999)
            ))

        # 3. Create Context-Matched 15s Scenes with Vivid Visual Prompts
        scenes: List[SceneModel] = []
        last_loc = locations[0]
        last_char = characters[0]

        camera_templates = [
            ("Slow cinematic tracking shot focusing on character emotional reaction", "चरित्र के चेहरे और हावभाव पर केंद्रित स्लो ट्रैकिंग शॉट"),
            ("Atmospheric low-angle camera sweep showcasing environment scale", "लोकेशन की वास्तुकला को दर्शाता लो-एंगल कैमरा पैन"),
            ("Dynamic forward dolly shot following character movement", "चरित्र के कदमों और गतिविधि का पीछा करता फॉरवर्ड डॉली शॉट"),
            ("Wide establishing IMAX angle revealing dramatic landscape depth", "गहराई और माहौल को प्रकट करता विस्तृत वाइड शॉट")
        ]

        for idx, chunk in enumerate(raw_chunks):
            scene_num = idx + 1
            
            # Context-matched character
            cur_char = last_char
            for c in characters:
                if c.name in chunk:
                    cur_char = c
                    last_char = c
                    break

            # Context-matched location
            cur_loc = last_loc
            for loc in locations:
                keywords = loc.anchor_props + [loc.name]
                if any(kw in chunk for kw in keywords) or (loc.name in chunk):
                    cur_loc = loc
                    last_loc = loc
                    break
                elif "जंगल" in chunk or "पेड़" in chunk or "रास्ता" in chunk or "अंधेरा" in chunk:
                    forest_loc = next((l for l in locations if "जंगल" in l.name or "Forest" in l.architecture_style), None)
                    if forest_loc:
                        cur_loc = forest_loc
                        last_loc = forest_loc
                        break
                elif "घर" in chunk or "कमरा" in chunk or "दीवार" in chunk or "खिड़की" in chunk or "दरवाजा" in chunk:
                    house_loc = next((l for l in locations if "घर" in l.name or "Hut" in l.architecture_style), None)
                    if house_loc:
                        cur_loc = house_loc
                        last_loc = house_loc
                        break

            cam_en, cam_hi = camera_templates[idx % len(camera_templates)]

            # Convert action to vivid diffusion prompt
            action_desc = self._translate_action_to_english_prompt(chunk, cur_char.name, cur_loc.name)

            visual_prompt = (
                f"{action_desc}. {cam_en}. "
                f"Protagonist: {cur_char.name} ({cur_char.appearance}, wearing {cur_char.costume}). "
                f"Setting: {cur_loc.name} ({cur_loc.architecture_style}). "
                f"Lighting: {cur_loc.lighting_scheme}. Palette: {cur_loc.color_palette}. "
                f"8K resolution, 35mm anamorphic cinema lens, hyper-realistic, award winning photography."
            )

            dialogue = DialogueModel(
                character_id=cur_char.character_id,
                character_name=cur_char.name,
                text=chunk if len(chunk) < 120 else chunk[:115] + "...",
                emotion="intense_cinematic",
                duration_seconds=float(min(duration, max(4, len(chunk) // 9)))
            )

            scenes.append(SceneModel(
                scene_number=scene_num,
                duration_seconds=duration,
                location_id=cur_loc.location_id,
                location_name=cur_loc.name,
                character_ids=[cur_char.character_id],
                camera_movement=f"{cam_hi} ({cam_en})",
                lighting_mood=cur_loc.lighting_scheme,
                visual_prompt=visual_prompt,
                negative_prompt="blurry, distorted face, low quality, morphing artifacts, extra fingers, cartoon, 3d render look",
                dialogue=dialogue,
                sfx=["Cinematic environmental ambiance", "Subtle footsteps", "Wind breeze"],
                bgm_mood="Intense dramatic Indian orchestral strings with ethereal emotional flute",
                status="pending"
            ))

        return characters, locations, scenes

    def _translate_action_to_english_prompt(self, hindi_chunk: str, char_name: str, loc_name: str) -> str:
        """Translates specific Hindi action beats into vivid diffusion prompt descriptors."""
        desc = f"Cinematic scene of {char_name} in {loc_name}"
        if "खिड़की" in hindi_chunk or "खड़ा" in hindi_chunk:
            desc = f"Dramatic view of {char_name} standing by an old carved wooden window, looking outside intently as morning golden sunlight hits rustic clay walls"
        elif "दरवाजा" in hindi_chunk or "रास्ता" in hindi_chunk:
            desc = f"Atmospheric shot of {char_name} opening a heavy rustic wooden door and stepping out onto an ominous misty dirt path leading into dense wilderness"
        elif "जंगल" in hindi_chunk or "आवाज" in hindi_chunk:
            desc = f"Suspenseful shot of {char_name} cautiously walking through a mysterious dense forest with ancient banyan trees, ground fog, and eerie moonlight"
        elif "तय किया" in hindi_chunk or "रहस्य" in hindi_chunk:
            desc = f"Close-up intense determination on {char_name}'s face, cinematic Bollywood rim lighting, clutching his traditional clothes ready for a perilous mystery"
        else:
            desc = f"Cinematic shot of {char_name} actively experiencing the story moment: '{hindi_chunk[:50]}' with rich realistic environment interaction"

        return desc

    def _detect_character_names(self, text: str) -> List[str]:
        known = [
            "रामू", "रोहन", "विक्रम", "राघव", "सीमा", "प्रिया", "अनन्या", "अर्जुन", 
            "माया", "ठाकुर", "राजा", "साधु", "कमल", "अमित", "संजय", "दीपक", "नेहा", "पूजा"
        ]
        found = []
        for name in known:
            if name in text and name not in found:
                found.append(name)
        if not found:
            found = ["रामू (मुख्य पात्र)"]
        return found

    def _detect_locations_with_metadata(self, text: str, genre: str, project_id: str) -> List[Dict[str, Any]]:
        locs = []
        idx = 1
        
        if "घर" in text or "कमरा" in text or "गांव" in text or "खिड़की" in text or "दीवार" in text:
            locs.append({
                "id": f"LOC_{project_id}_{idx}",
                "name": "गाँव का पुराना कच्चा घर",
                "desc": "मिट्टी की पारंपरिक दीवारों और नक्काशीदार लकड़ी की खिड़की वाला ग्रामीण घर",
                "style": "Rustic Indian Rural Clay House with Handcrafted Wooden Beams",
                "props": ["मिट्टी की दीवारें", "लकड़ी की खिड़की", "पीतल का लालटेन", "पुरानी खाट"],
                "palette": "Warm Terracotta, Earthy Ochre & Sunlit Amber",
                "lighting": "Dramatic golden sunbeams piercing through window shutters"
            })
            idx += 1

        if "जंगल" in text or "पेड़" in text or "रास्ता" in text or "रात" in text or "अंधेरा" in text:
            locs.append({
                "id": f"LOC_{project_id}_{idx}",
                "name": "घना रहस्यमयी जंगल",
                "desc": "प्राचीन विशाल पेड़ों और रहस्यमयी कोहरे से घिरा हुआ घना वन",
                "style": "Dense Ancient Indian Deep Forest with Twisted Banyan Trees",
                "props": ["विशाल बरगद के पेड़", "जमीन पर बिछा कोहरा", "पथरीला कच्चा रास्ता"],
                "palette": "Deep Midnight Blue, Indigo & Ethereal Emerald Green",
                "lighting": "Atmospheric moonlight filtering through thick canopy with ground fog"
            })
            idx += 1

        if not locs:
            locs.append({
                "id": f"LOC_{project_id}_{idx}",
                "name": "गाँव का पुराना कच्चा घर",
                "desc": "पारंपरिक ग्रामीण भारतीय परिवेश",
                "style": "Rustic Indian Rural House",
                "props": ["मिट्टी की दीवारें", "लकड़ी की खिड़की"],
                "palette": "Warm Amber & Terracotta",
                "lighting": "Volumetric sunlight"
            })

        return locs
