"""
Shared state schema for the AstroAgent graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class BirthDetails(BaseModel):
    name: str = ""
    date_of_birth: str = ""      # YYYY-MM-DD
    time_of_birth: str = ""      # HH:MM (24h)
    place_of_birth: str = ""
    latitude: float | None = None
    longitude: float | None = None
    timezone: str | None = None


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_output: Any | None = None


class AgentState(BaseModel):
    """The full state passed through the LangGraph nodes."""

    # LangGraph message history (annotated for append-only)
    messages: Annotated[list[BaseMessage], add_messages] = []

    # User's birth data (set once, reused)
    birth_details: BirthDetails | None = None

    # Cached birth chart computation
    birth_chart: dict | None = None

    # Conversation intent classification
    intent: Literal[
        "chart_request",
        "daily_horoscope",
        "free_question",
        "off_topic",
        "unknown",
    ] = "unknown"

    # Tool call tracking for eval
    tool_calls_made: list[str] = []
    step_count: int = 0

    # Final assistant response (populated by last node)
    final_response: str = ""

    # Error state
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True
