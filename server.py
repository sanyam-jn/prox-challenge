"""FastAPI server — serves the frontend and proxies chat requests to the agent."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import run_agent

app = FastAPI(title="Vulcan OmniPro 220 AI Assistant")
_pool = ThreadPoolExecutor(max_workers=4)


class ChatRequest(BaseModel):
    message: str
    images: Optional[list[str]] = None  # base64-encoded


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _pool, run_agent, req.message, req.images or []
        )
        return ChatResponse(response=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok"}


# Static frontend — must be mounted last
app.mount("/", StaticFiles(directory="static", html=True), name="static")
