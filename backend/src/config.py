# backend/src/config.py
import os
from dataclasses import dataclass

# Basic RAG settings
TOP_K_RESULTS = int(os.environ.get('TOP_K_RESULTS', 15))
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 300))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 100))
DEBUG_RAG = os.environ.get('DEBUG_RAG', 'false').lower() in ('1','true','yes')
VECTOR_DB_PATH = os.environ.get('VECTOR_DB_PATH', './data/vectordb/')

# Common app settings (backward-compatible names expected by older code)
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./data/unibuddy.db')
SECRET_KEY = os.environ.get('SECRET_KEY', 'unibuddy_dev_secret')
DEBUG = os.environ.get('DEBUG', 'false').lower() in ('1','true','yes')
BACKEND_HOST = os.environ.get('BACKEND_HOST', '127.0.0.1')
BACKEND_PORT = int(os.environ.get('BACKEND_PORT', 9000))
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

@dataclass
class Settings:
    TOP_K_RESULTS: int = TOP_K_RESULTS
    CHUNK_SIZE: int = CHUNK_SIZE
    CHUNK_OVERLAP: int = CHUNK_OVERLAP
    DEBUG_RAG: bool = DEBUG_RAG
    VECTOR_DB_PATH: str = VECTOR_DB_PATH
    DATABASE_URL: str = DATABASE_URL
    SECRET_KEY: str = SECRET_KEY
    DEBUG: bool = DEBUG
    BACKEND_HOST: str = BACKEND_HOST
    BACKEND_PORT: int = BACKEND_PORT
    LOG_LEVEL: str = LOG_LEVEL

    # fallback to environment for any unknown attribute access
    def __getattr__(self, name):
        if name in os.environ:
            val = os.environ[name]
            # try to coerce ints/bools
            if val.isdigit():
                return int(val)
            if val.lower() in ('1','true','yes','false','0','no'):
                return val.lower() in ('1','true','yes')
            return val
        raise AttributeError(name)

# backward-compatible object expected by older code
settings = Settings()
