"""
Launcher script for AI Hindi Cinema Studio.
Starts the FastAPI Backend and Web UI on http://127.0.0.1:8000
"""

import os
import sys
import uvicorn

if __name__ == "__main__":
    # Ensure current working directory is backend
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(current_dir, "backend")
    
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    os.chdir(backend_dir)

    print("===============================================================")
    print("🎬 AI HINDI CINEMA STUDIO (MiniMax H3 Open-Weights Engine)")
    print("===============================================================")
    print("👉 Web Studio UI: http://127.0.0.1:8000")
    print("👉 API Documentation: http://127.0.0.1:8000/docs")
    print("===============================================================")

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
