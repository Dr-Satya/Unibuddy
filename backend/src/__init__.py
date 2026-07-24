"""Source package for UniBuddy backend services."""

import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Version metadata
__version__ = "1.0.0"
__author__ = "UniBuddy Team"

# Import core components
from src.config import settings
from src.models import ModelManager
from src.services import ChatService
from src.main import RAGSystem

# Package initialization
