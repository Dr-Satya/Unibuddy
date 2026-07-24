# backend/src/config.py
import os
from dataclasses import dataclass, field


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def _required_int_env(name: str) -> int:
    value = _required_env(name)
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer: {value!r}") from exc


def _required_csv_env(name: str) -> list[str]:
    value = _required_env(name)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise RuntimeError(f"Environment variable {name} must contain at least one comma-separated value")
    return items

# Basic RAG settings
TOP_K_RESULTS = int(os.environ.get('TOP_K_RESULTS', 15))
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 300))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 100))
DEBUG_RAG = os.environ.get('DEBUG_RAG', 'true').lower() in ('1','true','yes')
VECTOR_DB_PATH = os.environ.get('VECTOR_DB_PATH', './data/vectordb/')

# Common app settings (backward-compatible names expected by older code)
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./data/unibuddy.db')
SECRET_KEY = os.environ.get('SECRET_KEY', 'unibuddy_dev_secret')
DEBUG = os.environ.get('DEBUG', 'false').lower() in ('1','true','yes')
BACKEND_HOST = os.environ.get('BACKEND_HOST', '0.0.0.0').strip() or '0.0.0.0'
BACKEND_PORT = int(os.environ.get('BACKEND_PORT', 9000))
BACKEND_URL = _required_env('BACKEND_URL')
CLIENT_URL = _required_env('CLIENT_URL')
AUTH_URL = _required_env('AUTH_URL')
GOOGLE_REDIRECT_URI = _required_env('GOOGLE_REDIRECT_URI')
CORS_ALLOWED_ORIGINS = _required_csv_env('CORS_ALLOWED_ORIGINS')
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
    BACKEND_URL: str = BACKEND_URL
    CLIENT_URL: str = CLIENT_URL
    AUTH_URL: str = AUTH_URL
    GOOGLE_REDIRECT_URI: str = GOOGLE_REDIRECT_URI
    CORS_ALLOWED_ORIGINS: list[str] = field(default_factory=lambda: list(CORS_ALLOWED_ORIGINS))
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
