from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
import os

# Load .env from backend directory
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from src.api_adapter import get_reply

app = FastAPI(title="UniBuddy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_email: Optional[str] = None
    user_profile: Optional[dict] = None  # {name, email, degree, branch, year, section}

@app.get('/')
async def root():
    return {"service": "UniBuddy Chatbot API", "status": "running", "version": "1.0.0"}

@app.get('/health')
async def health():
    return {"status": "healthy"}

@app.get('/sections')
async def get_sections():
    """Returns all available section options for signup dropdown."""
    try:
        from src.timetable_lookup import get_section_options
        return {"sections": get_section_options()}
    except Exception as e:
        return {"sections": [], "error": str(e)}

@app.post('/chat')
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail='Empty message')
    result = get_reply(
        req.message,
        session_id=req.session_id,
        user_profile=req.user_profile
    )
    if req.user_email:
        result['user_email'] = req.user_email
    return result
