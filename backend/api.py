from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from backend directory
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from src.api_adapter import get_reply

app = FastAPI(title="UniBuddy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

@app.post('/chat')
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail='Empty message')
    result = get_reply(req.message, session_id=req.session_id)
    return result
