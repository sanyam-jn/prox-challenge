"""FastAPI server — SSE streaming chat + static pages."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import run_agent_stream

app = FastAPI(title="Vulcan OmniPro 220 AI Assistant")
_pool = ThreadPoolExecutor(max_workers=4)


class ChatRequest(BaseModel):
    message: str
    images: Optional[list[str]] = None


@app.post("/chat")
async def chat(req: ChatRequest):
    """SSE stream of agent events: status updates then final response."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run_sync():
        for event in run_agent_stream(req.message, req.images or []):
            loop.call_soon_threadsafe(queue.put_nowait, event)
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# Serve rendered page images so the agent can embed them in responses
app.mount("/pages", StaticFiles(directory="data/pages"), name="pages")

# Frontend — must be last
app.mount("/", StaticFiles(directory="static", html=True), name="static")
