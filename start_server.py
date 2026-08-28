"""
Launcher script for AI Hindi Cinema Studio v3.0.
Starts the FastAPI Backend and Web UI with environment validation.
"""

import os
import sys
import uvicorn


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(current_dir, "backend")

    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    os.chdir(backend_dir)

    # Configuration
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "true").lower() == "true"

    print("=" * 65)
    print("🎬 AI HINDI CINEMA STUDIO v3.0")
    print("   दुनिया का सबसे उत्तम AI वीडियो जनरेशन सॉफ्टवेयर")
    print("=" * 65)
    print(f"👉 Web Studio UI:      http://{host}:{port}")
    print(f"👉 API Documentation:  http://{host}:{port}/docs")
    print(f"👉 ComfyUI GPU:        {os.getenv('COMFYUI_URL', 'http://127.0.0.1:8188')}")
    print("=" * 65)

    # Environment check
    if os.getenv("GEMINI_API_KEY"):
        print("✅ Gemini API Key: Configured")
    else:
        print("⚠️  Gemini API Key: Not set (optional — for AI story analysis)")

    import shutil
    if shutil.which("ffmpeg"):
        print("✅ FFmpeg: Found")
    else:
        print("⚠️  FFmpeg: Not found — video assembly will use fallbacks")

    print("=" * 65)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level="info"
    )


if __name__ == "__main__":
    main()
