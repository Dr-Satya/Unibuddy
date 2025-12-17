"""
University Assistant AI Chatbot

A production-grade AI chatbot with RAG, multi-model support, and university information management.
"""

__version__ = "1.0.0"
__author__ = "University Assistant Team"
__email__ = "admin@university.edu"
__license__ = "Apache-2.0"

# Core imports for easier access
from src.config import settings
from src.models import ModelManager
from src.services import ChatService
from src.main import RAGSystem

__all__ = [
    "settings",
    "ModelManager", 
    "ChatService",
    "RAGSystem",
]
