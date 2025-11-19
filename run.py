#!/usr/bin/env python3
"""
University Assistant AI - Main Entry Point

This is the main entry point for the University Assistant AI chatbot.
It provides a simple interface to launch the CLI application.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def main():
    """Main entry point."""
    try:
        # Import and run the CLI
        from src.main import main_menu
        
        print("🎓 Starting University Assistant AI...")
        print("=" * 50)
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/vectordb", exist_ok=True)
        os.makedirs("secure_storage", exist_ok=True)
        
        # Run the main menu
        main_menu()
        
    except KeyboardInterrupt:
        print("\\n👋 Application interrupted. Goodbye!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
