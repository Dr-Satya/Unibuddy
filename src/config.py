import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    # API Keys
    HUGGINGFACE_API_TOKEN: str = os.getenv("HUGGINGFACE_API_TOKEN", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Database & Vector DB
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/uni_assistant.db")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vectordb/")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change_me")
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "change_me")

    # HIPAA / Monitoring
    ENABLE_MFA: bool = os.getenv("ENABLE_MFA", "true").lower() == "true"
    ENABLE_MONITORING: bool = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # University Data / Scraping
    UNIVERSITY_URL: str = os.getenv("UNIVERSITY_URL", "https://www.gdgoenkauniversity.com/admissions/fee-structure")
    SCRAPING_INTERVAL_HOURS: int = int(os.getenv("SCRAPING_INTERVAL_HOURS", "24"))

    # Model Configuration
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt2")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))

    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # CLI
    CLI_HISTORY_FILE: str = os.getenv("CLI_HISTORY_FILE", ".cli_history")
    MAX_CONVERSATION_LENGTH: int = int(os.getenv("MAX_CONVERSATION_LENGTH", "50"))

settings = Settings()

