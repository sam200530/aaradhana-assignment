"""
AstroAgent API — FastAPI server with SSE streaming
"""
import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.graph import AstroGraph
from app.agents.state import BirthDetails, ConversationMessage

app = FastAPI(title="AstroAgent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

astro_graph = AstroGraph()


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str
    birth_details: dict | None = None


class BirthDetailsRequest(BaseModel):
    name: str
    date_of_birth: str  # YYYY-MM-DD
    time_of_birth: str  # HH:MM
    place_of_birth: str
    timezone: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream chat responses via SSE."""

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            birth_details = None
            if req.birth_details:
                birth_details = BirthDetails(**req.birth_details)

            async for event in astro_graph.stream(
                session_id=req.session_id,
                message=req.message,
                birth_details=birth_details,
            ):
                data = json.dumps(event)
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)

            yield "data: [DONE]\n\n"

        except Exception as e:
            error_event = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_event}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    """Non-streaming chat for evaluation."""
    birth_details = None
    if req.birth_details:
        birth_details = BirthDetails(**req.birth_details)

    result = await astro_graph.invoke(
        session_id=req.session_id,
        message=req.message,
        birth_details=birth_details,
    )
    return result


@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    """Get conversation history for a session."""
    history = await astro_graph.get_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a session."""
    await astro_graph.clear_session(session_id)
    return {"cleared": True}
