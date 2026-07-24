#!/usr/bin/env python3
"""
UniBuddy - API-only startup (production-ready)
Initializes core backend systems (models, RAG) and starts the FastAPI server.
No CLI or interactive prompts are ever launched.
"""

import sys
from pathlib import Path

# Ensure stdout uses UTF-8 to allow emoji/status characters on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    import os
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# Make src/ importable
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Load .env BEFORE importing any src modules
from dotenv import load_dotenv
load_dotenv(dotenv_path=current_dir / ".env")

import uvicorn

if __name__ == "__main__":
    print("Initializing core systems...\n")

    # Initialize core systems — preserve all ✅ status prints
    try:
        from src.threaded_models import ThreadedModelManager
        from src.threaded_rag import ThreadedRAGSystem

        model_manager = ThreadedModelManager()
        rag_system = ThreadedRAGSystem()

        print("\n✅ Core systems initialized. Starting API server...\n")
    except Exception as e:
        # Avoid non-encodable characters in error message
        print(f"FATAL: error during initialization: {e}")
        sys.exit(1)

    # Start Uvicorn directly
    uvicorn.run(
        "api:app",
        host=os.environ.get("BACKEND_HOST", "127.0.0.1"),
        port=int(os.environ.get("BACKEND_PORT", "9000")),
        log_level="info",
        reload=False
    )
