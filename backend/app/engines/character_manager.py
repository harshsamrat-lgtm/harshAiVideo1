"""
Character DNA & Face Consistency Manager.
Ensures identical facial structure, costume, hair, and age across all scenes.
"""

import os
from PIL import Image, ImageDraw
from app.models import CharacterModel


class CharacterManager:
    """
    Manages persistent 360° character profiles, portrait turnaround sheets,
    and face metadata for MiniMax H3 Reference-to-Video.
    """

    def __init__(self, assets_dir: str = "media_store/characters"):
        self.assets_dir = assets_dir
        os.makedirs(self.assets_dir, exist_ok=True)

    def generate_master_character_sheet(self, character: CharacterModel) -> str:
        """
        Creates or retrieves the master 3-angle character turnaround portrait sheet.
        """
        filename = f"master_{character.character_id}_{character.seed}.png"
        filepath = os.path.join(self.assets_dir, filename)

        if not os.path.exists(filepath):
            self._render_master_portrait_sheet(filepath, character)

        character.master_portrait_url = f"/media/characters/{filename}"
        return character.master_portrait_url

    def _render_master_portrait_sheet(self, filepath: str, char: CharacterModel):
        """Renders an aesthetic 3-angle turnaround character card."""
        img = Image.new("RGB", (1200, 600), color=(14, 16, 20))
        draw = ImageDraw.Draw(img)

        # Header info
        draw.rectangle([0, 0, 1200, 70], fill=(22, 26, 34))
        draw.text((40, 25), f"👤 MASTER ACTOR PROFILE: {char.name} ({char.age} yrs, {char.gender})", fill=(255, 215, 0))
        draw.text((700, 25), f"🎙 Voice: {char.voice_profile}", fill=(180, 200, 230))

        # 3 View Angles (Front, 45-degree, Side)
        angles = ["FRONT PROFILE (0°)", "CINEMATIC 3/4 (45°)", "SIDE PROFILE (90°)"]
        for i, angle in enumerate(angles):
            x_start = 50 + i * 380
            draw.rectangle([x_start, 100, x_start + 340, 520], fill=(20, 25, 32), outline=(70, 85, 110), width=2)
            draw.text((x_start + 80, 120), angle, fill=(240, 180, 50))

            # Character portrait graphics placeholder
            draw.ellipse([x_start + 95, 180, x_start + 245, 330], fill=(215, 170, 130), outline=(255, 240, 200), width=2)
            draw.polygon([(x_start + 170, 330), (x_start + 50, 480), (x_start + 290, 480)], fill=(45, 60, 85))

            # Feature descriptions
            draw.text((x_start + 20, 490), f"Costume: {char.costume[:22]}", fill=(170, 180, 195))

        # Bottom banner
        draw.rectangle([0, 550, 1200, 600], fill=(18, 22, 28))
        draw.text((40, 565), f"✨ DNA Description: {char.appearance}", fill=(190, 200, 215))

        img.save(filepath, "PNG")
