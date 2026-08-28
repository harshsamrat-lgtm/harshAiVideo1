"""
AI Story Director Engine (Vivid Action & Context-Aware Prompt Generator).
Parses Hindi stories into contextually rich cinematic screenplays with
intelligent character/location detection and vivid diffusion prompts.
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

        total_duration = sum(s.duration_seconds for s in scenes)

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
            scenes=scenes,
            total_duration_seconds=total_duration
        )

    def _extract_screenplay(
        self, project_id: str, text: str, genre: str, visual_style: str, duration: int = 15
    ) -> Tuple[List[CharacterModel], List[LocationModel], List[SceneModel]]:
        raw_chunks = [s.strip() for s in re.split(r'[।\.\n\?\!]+', text) if len(s.strip()) > 6]
        if not raw_chunks:
            raw_chunks = [text.strip() or "कहानी की शुरुआत एक रोमांचक मोड़ से होती है।"]

        # 1. Detect Characters
        detected_names = self._detect_character_names(text)
        characters: List[CharacterModel] = []
        voice_profiles = [
            ("hi-IN-MadhurNeural", "Male", 28, "तीखे नैन-नक्श, मध्यम कद, गहरी आँखें, गंभीर एवं दृढ़ निश्चयी चेहरा", "पारंपरिक सूती सफेद धोती और कुर्ता"),
            ("hi-IN-SwaraNeural", "Female", 24, "सुंदर भावपूर्ण आँखें, सादगी भरा सौम्य चेहरा, लंबे काले बाल", "पारंपरिक भारतीय लाल और सुनहरी साड़ी"),
            ("hi-IN-MadhurNeural", "Male", 48, "गहरा रौबदार चेहरा, घनी मूंछें, अनुभवी नज़रें, बुलंद व्यक्तित्व", "गहरा विंटेज बंदगला कुर्ता"),
            ("hi-IN-SwaraNeural", "Female", 32, "आत्मविश्वास से परिपूर्ण, राजसी सौंदर्य, तेज नैन-नक्श", "शाही सिल्क परिधान"),
            ("hi-IN-MadhurNeural", "Male", 60, "बुज़ुर्ग सफ़ेद दाढ़ी, शांत ज्ञानी चेहरा, गहरी झुर्रियाँ", "भगवा वस्त्र और रुद्राक्ष माला"),
            ("hi-IN-SwaraNeural", "Female", 18, "किशोरी, मासूम गोल चेहरा, चमकती आँखें, हल्की मुस्कान", "सादा सूती कुर्ता और चूड़ीदार"),
        ]

        # Male-ending exceptions (names ending with feminine-sounding suffix but are male)
        male_exceptions = {"राजा", "राणा", "रामा", "रामू", "साधु", "बाबा", "राजू", "बंटू", "गोलू", "पप्पू", "बब्बू", "बिट्टू"}

        for idx, name in enumerate(detected_names):
            preset = voice_profiles[idx % len(voice_profiles)]
            # Improved gender detection
            female_suffixes = ("ा ", "ी", "िया", "ाली")
            is_female = any(name.endswith(e) for e in female_suffixes) and name not in male_exceptions
            gender = "Female" if is_female else "Male"

            char_id = f"CHAR_{project_id}_{idx+1}_{name.replace(' ', '_')}"
            characters.append(CharacterModel(
                character_id=char_id,
                name=name,
                gender=gender,
                age=preset[2] + (idx * 3),
                appearance=preset[3] if gender == preset[1] else voice_profiles[(idx + 1) % len(voice_profiles)][3],
                costume=preset[4] if gender == preset[1] else voice_profiles[(idx + 1) % len(voice_profiles)][4],
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
            ("Wide establishing IMAX angle revealing dramatic landscape depth", "गहराई और माहौल को प्रकट करता विस्तृत वाइड शॉट"),
            ("Close-up portrait shot with shallow depth of field on character face", "चरित्र के चेहरे पर शैलो डेप्थ-ऑफ-फ़ील्ड क्लोज़-अप"),
            ("Overhead bird-eye crane shot revealing full scene geography", "ऊपर से पूरे दृश्य को प्रकट करता बर्ड-आई क्रेन शॉट"),
            ("Smooth orbit shot rotating around character in 180 degrees", "चरित्र के चारों ओर 180° घूमता स्मूथ ऑर्बिट शॉट"),
            ("Dramatic push-in zoom intensifying emotional tension", "भावनात्मक तनाव को तीव्र करता ड्रामैटिक पुश-इन ज़ूम"),
        ]

        transition_effects = ["crossfade", "fade_black", "crossfade", "dissolve", "cut", "crossfade"]

        for idx, chunk in enumerate(raw_chunks):
            scene_num = idx + 1

            # Context-matched character
            cur_char = last_char
            for c in characters:
                if c.name in chunk:
                    cur_char = c
                    last_char = c
                    break

            # Context-matched location (improved matching)
            cur_loc = self._match_location_for_chunk(chunk, locations, last_loc)
            last_loc = cur_loc

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

            # Detect dialogue emotion from content
            emotion = self._detect_emotion_from_text(chunk)

            dialogue = DialogueModel(
                character_id=cur_char.character_id,
                character_name=cur_char.name,
                text=chunk if len(chunk) < 120 else chunk[:115] + "...",
                emotion=emotion,
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
                transition_effect=transition_effects[idx % len(transition_effects)],
                dialogue=dialogue,
                sfx=["Cinematic environmental ambiance", "Subtle footsteps", "Wind breeze"],
                bgm_mood="Intense dramatic Indian orchestral strings with ethereal emotional flute",
                status="pending"
            ))

        return characters, locations, scenes

    def _match_location_for_chunk(self, chunk: str, locations: List[LocationModel], last_loc: LocationModel) -> LocationModel:
        """Intelligently matches story chunk to the most relevant location."""
        # Direct name/prop match
        for loc in locations:
            keywords = loc.anchor_props + [loc.name]
            if any(kw in chunk for kw in keywords):
                return loc

        # Keyword-based matching
        keyword_map = {
            "जंगल": ["जंगल", "Forest"],
            "पेड़": ["जंगल", "Forest", "बगीचा"],
            "रास्ता": ["जंगल", "रास्ता", "Path"],
            "अंधेरा": ["जंगल", "Forest", "रात"],
            "घर": ["घर", "Hut", "House"],
            "कमरा": ["घर", "Room", "House"],
            "दीवार": ["घर", "House"],
            "खिड़की": ["घर", "House", "Window"],
            "दरवाजा": ["घर", "House", "Door"],
            "मंदिर": ["मंदिर", "Temple"],
            "पूजा": ["मंदिर", "Temple"],
            "नदी": ["नदी", "River"],
            "पानी": ["नदी", "River", "तालाब"],
            "बाज़ार": ["बाज़ार", "Market"],
            "दुकान": ["बाज़ार", "Market"],
            "शहर": ["शहर", "City"],
            "सड़क": ["शहर", "City", "Road"],
            "पहाड़": ["पहाड़", "Mountain"],
            "गुफा": ["गुफा", "Cave"],
            "खेत": ["खेत", "Field"],
            "तालाब": ["तालाब", "Pond"],
        }

        for hindi_key, loc_keywords in keyword_map.items():
            if hindi_key in chunk:
                for loc in locations:
                    if any(kw in loc.name or kw in loc.architecture_style for kw in loc_keywords):
                        return loc

        return last_loc

    def _detect_emotion_from_text(self, text: str) -> str:
        """Detects emotional tone from Hindi text content."""
        if any(w in text for w in ["डर", "भय", "कांप", "डरा", "घबरा"]):
            return "fearful"
        if any(w in text for w in ["खुश", "हँस", "मुस्कुरा", "आनंद", "प्रसन्न"]):
            return "happy"
        if any(w in text for w in ["दुख", "रो", "आँसू", "विलाप", "उदास"]):
            return "sad"
        if any(w in text for w in ["क्रोध", "गुस्सा", "चिल्ला", "भड़क"]):
            return "angry"
        if any(w in text for w in ["रहस्य", "अजीब", "अनजान", "चौंक"]):
            return "intense_cinematic"
        if any(w in text for w in ["तय किया", "दृढ़", "निश्चय", "साहस"]):
            return "excited"
        return "neutral"

    def _translate_action_to_english_prompt(self, hindi_chunk: str, char_name: str, loc_name: str) -> str:
        """Translates specific Hindi action beats into vivid diffusion prompt descriptors."""

        # Pattern matching with priority order
        action_patterns = [
            (["खिड़की", "खड़ा", "देख रहा"],
             f"Dramatic view of {char_name} standing by an old carved wooden window, looking outside intently as morning golden sunlight hits rustic clay walls"),
            (["दरवाजा", "खोल"],
             f"Atmospheric shot of {char_name} opening a heavy rustic wooden door, dramatic light spilling in from outside with dust particles visible in light beams"),
            (["रास्ता", "चल", "बढ़"],
             f"Cinematic tracking shot of {char_name} walking determinedly along an ominous misty dirt path through dense wilderness"),
            (["जंगल", "घना", "पेड़"],
             f"Suspenseful shot of {char_name} cautiously navigating through a mysterious dense forest with ancient banyan trees, ground fog, and eerie moonlight"),
            (["आवाज", "सुन"],
             f"Close-up of {char_name} freezing mid-step, head tilted, intensely listening to mysterious distant sounds in the darkness"),
            (["तय किया", "रहस्य", "निश्चय"],
             f"Close-up intense determination on {char_name}'s face, cinematic Bollywood rim lighting, eyes burning with resolve"),
            (["डर", "भय", "कांप"],
             f"Dramatic low-angle shot of {char_name} trembling with fear, shadows deepening around, breath visible in cold air"),
            (["दौड़", "भाग"],
             f"Dynamic chase shot of {char_name} running urgently through the environment, camera tracking in parallel motion"),
            (["नदी", "पानी", "तालाब"],
             f"Cinematic wide shot of {char_name} standing at the edge of a glistening river, moonlight reflecting on water surface"),
            (["मंदिर", "पूजा", "प्रार्थना"],
             f"Sacred establishing shot of {char_name} standing before an ancient ornate Hindu temple, oil lamps flickering in foreground"),
            (["रात", "अंधेरा", "चाँद"],
             f"Atmospheric night shot of {char_name} silhouetted against a dramatic moonlit sky, deep shadows and silver-blue lighting"),
            (["सुबह", "किरण", "उजाला"],
             f"Golden hour cinematic shot of {char_name} bathed in warm sunrise light, volumetric god rays filtering through"),
            (["बारिश", "बरसात"],
             f"Dramatic rain-soaked shot of {char_name}, water streaming down face, lightning illuminating the background"),
            (["लड़ाई", "युद्ध", "संघर्ष"],
             f"Epic action shot of {char_name} in intense combat stance, dynamic motion blur, dust and debris in air"),
        ]

        for keywords, prompt in action_patterns:
            if any(kw in hindi_chunk for kw in keywords):
                return prompt

        # Default: contextual scene description
        return f"Cinematic shot of {char_name} actively experiencing the story moment in {loc_name}, rich environmental interaction, dramatic lighting, emotionally charged composition"

    def _detect_character_names(self, text: str) -> List[str]:
        """Detects character names from Hindi text — expanded list."""
        known = [
            # Male names
            "रामू", "रोहन", "विक्रम", "राघव", "अर्जुन", "कमल", "अमित", "संजय",
            "दीपक", "सूरज", "मोहन", "राजू", "गोपाल", "हरि", "श्याम", "कृष्ण",
            "राम", "भीम", "लक्ष्मण", "विष्णु", "शिव", "गणेश", "ओम", "आदित्य",
            "विवेक", "मनोज", "सचिन", "राजेश", "महेश", "सुरेश", "नरेश", "दिनेश",
            # Female names
            "सीमा", "प्रिया", "अनन्या", "माया", "नेहा", "पूजा", "रीना", "गीता",
            "सीता", "राधा", "मीरा", "लक्ष्मी", "पार्वती", "दुर्गा", "अनीता",
            "सविता", "कविता", "रेखा", "सुनीता", "ममता", "शांति",
            # Title-based
            "ठाकुर", "राजा", "साधु", "बाबा", "पंडित", "महाराज",
        ]
        found = []
        for name in known:
            if name in text and name not in found:
                found.append(name)
        if not found:
            found = ["नायक"]
        return found

    def _detect_locations_with_metadata(self, text: str, genre: str, project_id: str) -> List[Dict[str, Any]]:
        """Detects locations from story text with rich metadata — significantly expanded."""
        locs = []
        idx = 1

        # Each location definition: (keywords_to_detect, location_spec)
        location_definitions = [
            {
                "keywords": ["घर", "कमरा", "गांव", "गाँव", "खिड़की", "दीवार", "दरवाजा", "छत", "आँगन"],
                "name": "गाँव का पुराना कच्चा घर",
                "desc": "मिट्टी की पारंपरिक दीवारों और नक्काशीदार लकड़ी की खिड़की वाला ग्रामीण घर",
                "style": "Rustic Indian Rural Clay House with Handcrafted Wooden Beams",
                "props": ["मिट्टी की दीवारें", "लकड़ी की खिड़की", "पीतल का लालटेन", "पुरानी खाट"],
                "palette": "Warm Terracotta, Earthy Ochre & Sunlit Amber",
                "lighting": "Dramatic golden sunbeams piercing through window shutters"
            },
            {
                "keywords": ["जंगल", "पेड़", "रास्ता", "रात", "अंधेरा", "वन", "बरगद"],
                "name": "घना रहस्यमयी जंगल",
                "desc": "प्राचीन विशाल पेड़ों और रहस्यमयी कोहरे से घिरा हुआ घना वन",
                "style": "Dense Ancient Indian Deep Forest with Twisted Banyan Trees",
                "props": ["विशाल बरगद के पेड़", "जमीन पर बिछा कोहरा", "पथरीला कच्चा रास्ता"],
                "palette": "Deep Midnight Blue, Indigo & Ethereal Emerald Green",
                "lighting": "Atmospheric moonlight filtering through thick canopy with ground fog"
            },
            {
                "keywords": ["मंदिर", "पूजा", "प्रार्थना", "देवता", "मूर्ति", "घंटी"],
                "name": "प्राचीन शिव मंदिर",
                "desc": "पत्थर की नक्काशी और दीपों की रोशनी वाला प्राचीन मंदिर",
                "style": "Ancient Indian Stone Temple with Intricate Carvings and Oil Lamps",
                "props": ["पत्थर की मूर्तियाँ", "तेल के दीपक", "घंटी", "लाल फूल"],
                "palette": "Sacred Saffron, Deep Crimson & Golden Oil Lamp Glow",
                "lighting": "Warm flickering oil lamp light with deep shadows in stone corridors"
            },
            {
                "keywords": ["नदी", "पानी", "किनारा", "तालाब", "सरोवर", "झील"],
                "name": "पवित्र नदी का किनारा",
                "desc": "शांत बहती नदी और घाट की सीढ़ियों वाला पवित्र तटीय दृश्य",
                "style": "Sacred Indian Riverbank Ghat with Stone Steps and Flowing Water",
                "props": ["पत्थर की सीढ़ियाँ", "बहती नदी", "दीपक तैरते हुए", "नावें"],
                "palette": "Tranquil River Blue, Silver Moonlight & Sandy Gold",
                "lighting": "Serene twilight reflecting on gently flowing river water"
            },
            {
                "keywords": ["बाज़ार", "दुकान", "सड़क", "शहर", "भीड़", "गली"],
                "name": "पुराना शहरी बाज़ार",
                "desc": "संकरी गलियों और रंगीन दुकानों वाला व्यस्त पारंपरिक बाज़ार",
                "style": "Vibrant Old Indian Market Bazaar with Narrow Alleys and Colorful Shops",
                "props": ["मसालों की दुकानें", "रंगीन कपड़े", "पीतल के बर्तन", "फल-सब्जी"],
                "palette": "Vibrant Saffron, Deep Red, Royal Blue & Market Gold",
                "lighting": "Dappled sunlight through cloth canopies with busy market atmosphere"
            },
            {
                "keywords": ["पहाड़", "चोटी", "पर्वत", "ऊंचाई", "गुफा"],
                "name": "विशाल पर्वत शिखर",
                "desc": "बर्फ़ से ढकी चोटियों और गहरी खाइयों वाला हिमालयी पर्वत",
                "style": "Majestic Himalayan Mountain Peak with Snow and Deep Valleys",
                "props": ["बर्फ़ीली चोटी", "गहरी खाई", "बादल", "पथरीला मार्ग"],
                "palette": "Icy Blue, Snow White, Deep Slate Grey & Sunrise Orange",
                "lighting": "Epic dramatic sunlight breaking through mountain clouds"
            },
            {
                "keywords": ["खेत", "फसल", "किसान", "हल", "बैल"],
                "name": "हरे-भरे खेत",
                "desc": "फसलों से लहलहाते विशाल खेत और गाँव का कृषि दृश्य",
                "style": "Lush Indian Agricultural Fields with Golden Wheat and Green Crops",
                "props": ["गेहूँ के खेत", "बैलगाड़ी", "कुआँ", "पगडंडी"],
                "palette": "Lush Green, Golden Wheat, Earth Brown & Sky Blue",
                "lighting": "Warm afternoon sunlight casting long shadows across vast fields"
            },
            {
                "keywords": ["महल", "राजा", "सिंहासन", "दरबार", "राजघराना"],
                "name": "शाही राजमहल का दरबार",
                "desc": "सोने और संगमरमर से सजा भव्य राजसी दरबार",
                "style": "Opulent Indian Royal Palace Durbar with Marble and Gold",
                "props": ["सिंहासन", "संगमरमर के स्तंभ", "शाही पर्दे", "सोने के झूमर"],
                "palette": "Royal Gold, Ivory White, Deep Ruby Red & Emerald",
                "lighting": "Regal warm golden chandeliers with sunlight through stained glass"
            },
        ]

        for loc_def in location_definitions:
            if any(kw in text for kw in loc_def["keywords"]):
                locs.append({
                    "id": f"LOC_{project_id}_{idx}",
                    "name": loc_def["name"],
                    "desc": loc_def["desc"],
                    "style": loc_def["style"],
                    "props": loc_def["props"],
                    "palette": loc_def["palette"],
                    "lighting": loc_def["lighting"]
                })
                idx += 1

        # Default fallback location
        if not locs:
            locs.append({
                "id": f"LOC_{project_id}_{idx}",
                "name": "गाँव का पुराना कच्चा घर",
                "desc": "पारंपरिक ग्रामीण भारतीय परिवेश",
                "style": "Rustic Indian Rural House with Traditional Architecture",
                "props": ["मिट्टी की दीवारें", "लकड़ी की खिड़की", "पीतल का लालटेन"],
                "palette": "Warm Amber & Terracotta",
                "lighting": "Volumetric golden sunlight through wooden shutters"
            })

        return locs
