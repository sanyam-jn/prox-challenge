"""Vercel serverless entry point — wraps the agent with chat + health routes."""
import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Make project root importable (Vercel runs from repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import run_agent_stream

app = FastAPI()
_pool = ThreadPoolExecutor(max_workers=4)


class ChatRequest(BaseModel):
    message: str
    images: Optional[list[str]] = None
    api_key: Optional[str] = None   # supplied by user in Settings panel
    model: Optional[str] = None     # optional model override


@app.post("/chat")
async def chat(req: ChatRequest):
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run_sync():
        for event in run_agent_stream(
            req.message,
            req.images or [],
            api_key=req.api_key,
            model=req.model,
        ):
            loop.call_soon_threadsafe(queue.put_nowait, event)
        loop.call_soon_threadsafe(queue.put_nowait, None)

    _pool.submit(run_sync)

    async def generate():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
