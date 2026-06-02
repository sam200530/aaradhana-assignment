"""
AstroAgent LangGraph Graph
──────────────────────────
Nodes:
  classify_intent   → detect what the user wants
  router            → conditional edge based on intent
  compute_chart     → runs ephemeris tools if needed
  reason            → main LLM reasoning node
  tool_executor     → runs any tools the LLM requests
  respond           → formats final reply

State flows:
  START → classify_intent → router →
    [chart_request]  → compute_chart → reason → tool_executor? → respond
    [daily_horoscope] → reason → tool_executor → respond
    [free_question]   → reason → tool_executor? → respond
    [off_topic]       → respond (direct refusal)
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agents.state import AgentState, BirthDetails
from app.tools.astro_tools import (
    compute_birth_chart,
    geocode_place,
    get_daily_transits,
    knowledge_lookup,
)

MAX_STEPS = 8  # step budget guard

SYSTEM_PROMPT = """You are Aradhana, a warm and wise AI astrologer companion.
You help people understand their birth charts, daily planetary energies, and the symbolic language of astrology.

Core principles:
- Be warm, empathetic, and spiritually grounded in tone
- Always use real tool outputs for planetary data — never invent positions
- Offer insight and reflection, not certainty or fate
- ALWAYS include a gentle disclaimer when the topic touches health, finance, or legal matters
- If asked for medical, legal, or financial advice, redirect warmly but clearly
- You may decline harmful, manipulative, or deceptive requests

You have access to these tools:
- geocode_place: convert a city/country to coordinates
- compute_birth_chart: calculate natal chart from birth data
- get_daily_transits: get today's planetary positions and aspects
- knowledge_lookup: retrieve curated astrology knowledge

When answering, reason step by step, call the tools you need, then synthesize a grounded, caring response.
"""


def _make_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=streaming,
        max_tokens=1024,
    )


# ── LangChain tool wrappers ────────────────────────────────────────────────────

@lc_tool
def tool_geocode_place(place: str) -> str:
    """Resolve a place name to latitude, longitude, and timezone."""
    result = geocode_place(place)
    return json.dumps(result)


@lc_tool
def tool_compute_birth_chart(
    date_of_birth: str,
    time_of_birth: str,
    latitude: float,
    longitude: float,
    timezone: str,
) -> str:
    """Compute a natal birth chart using Swiss Ephemeris. date_of_birth: YYYY-MM-DD, time_of_birth: HH:MM"""
    result = compute_birth_chart(date_of_birth, time_of_birth, latitude, longitude, timezone)
    return json.dumps(result)


@lc_tool
def tool_get_daily_transits(date: str | None = None, natal_chart_json: str | None = None) -> str:
    """Get current planetary transits and aspects to natal chart. date: YYYY-MM-DD optional."""
    natal_chart = None
    if natal_chart_json:
        try:
            natal_chart = json.loads(natal_chart_json)
        except Exception:
            pass
    result = get_daily_transits(date=date, natal_chart=natal_chart)
    return json.dumps(result)


@lc_tool
def tool_knowledge_lookup(query: str) -> str:
    """Look up curated astrology knowledge for interpretations and concepts."""
    result = knowledge_lookup(query)
    return json.dumps(result)


ALL_TOOLS = [
    tool_geocode_place,
    tool_compute_birth_chart,
    tool_get_daily_transits,
    tool_knowledge_lookup,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# ── Node implementations ───────────────────────────────────────────────────────

async def classify_intent_node(state: AgentState) -> dict:
    """Classify the user's intent from the latest message."""
    llm = _make_llm()
    last_msg = state.messages[-1].content if state.messages else ""

    prompt = f"""Classify this astrology chatbot message into exactly one category:
- chart_request: user wants their birth chart computed or interpreted
- daily_horoscope: user asks about today's energy, transits, or daily forecast  
- free_question: general astrology question, sign meanings, aspects, etc.
- off_topic: completely unrelated to astrology (medical advice, finance, etc.)

Message: "{last_msg}"

Reply with ONLY one word from the list above."""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    intent_raw = response.content.strip().lower()

    valid = {"chart_request", "daily_horoscope", "free_question", "off_topic"}
    intent = intent_raw if intent_raw in valid else "free_question"

    return {"intent": intent, "step_count": state.step_count + 1}


def route_by_intent(state: AgentState) -> str:
    """Conditional edge: route based on classified intent."""
    if state.intent == "off_topic":
        return "respond_direct"
    return "reason"


async def reason_node(state: AgentState) -> dict:
    """Main LLM reasoning node with tool binding."""
    if state.step_count >= MAX_STEPS:
        return {
            "final_response": "I've been thinking deeply, but let me give you a direct answer with what I know.",
            "error": "step_budget_exceeded",
        }

    llm = _make_llm().bind_tools(ALL_TOOLS)

    # Build context injection
    context_parts = [SYSTEM_PROMPT]

    if state.birth_details:
        bd = state.birth_details
        context_parts.append(
            f"\nUser birth details on file: {bd.name}, born {bd.date_of_birth} "
            f"at {bd.time_of_birth} in {bd.place_of_birth}."
        )

    if state.birth_chart:
        chart_summary = _summarize_chart(state.birth_chart)
        context_parts.append(f"\nCached birth chart summary:\n{chart_summary}")

    messages = [SystemMessage(content="\n".join(context_parts))] + list(state.messages)

    response = await llm.ainvoke(messages)

    updates: dict = {
        "messages": [response],
        "step_count": state.step_count + 1,
    }

    # Track tool calls made
    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        updates["tool_calls_made"] = state.tool_calls_made + tool_names

    return updates


def should_use_tools(state: AgentState) -> str:
    """Edge: does the last AI message contain tool calls?"""
    last = state.messages[-1] if state.messages else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_executor"
    return "respond"


async def tool_executor_node(state: AgentState) -> dict:
    """Execute any tool calls from the last AI message."""
    last = state.messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}

    tool_messages = []
    birth_chart_update = state.birth_chart

    for tc in last.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        tool_fn = TOOL_MAP.get(tool_name)
        if tool_fn is None:
            result = json.dumps({"error": f"Unknown tool: {tool_name}"})
        else:
            try:
                result = tool_fn.invoke(tool_args)
            except Exception as e:
                result = json.dumps({"error": str(e)})

        # Cache birth chart if computed
        if tool_name == "tool_compute_birth_chart":
            try:
                parsed = json.loads(result)
                if "planets" in parsed:
                    birth_chart_update = parsed
            except Exception:
                pass

        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tc["id"],
                name=tool_name,
            )
        )

    updates: dict = {
        "messages": tool_messages,
        "step_count": state.step_count + 1,
    }
    if birth_chart_update != state.birth_chart:
        updates["birth_chart"] = birth_chart_update

    return updates


async def respond_node(state: AgentState) -> dict:
    """Synthesize the final assistant response from conversation history."""
    # If LLM already gave a text response without tool calls, extract it
    last = state.messages[-1] if state.messages else None

    if isinstance(last, AIMessage) and last.content and not last.tool_calls:
        return {"final_response": last.content}

    # Otherwise do a final summarizing pass
    llm = _make_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state.messages)
    response = await llm.ainvoke(messages)
    return {
        "messages": [response],
        "final_response": response.content,
    }


async def respond_direct_node(state: AgentState) -> dict:
    """Handle off-topic or guardrail responses."""
    last_msg = state.messages[-1].content if state.messages else ""

    # Check for safety concerns
    safety_keywords = ["suicide", "self-harm", "hurt myself", "kill", "overdose"]
    if any(kw in last_msg.lower() for kw in safety_keywords):
        response = (
            "I hear that you're going through something difficult. "
            "As your astrology companion, I'm not equipped to support you in this moment — "
            "please reach out to a crisis helpline or mental health professional who can truly help. "
            "You matter, and support is available."
        )
    else:
        response = (
            "I'm Aradhana, your astrology companion. I'm here to explore your birth chart, "
            "daily planetary energies, and the cosmic patterns in your life. "
            "I'm not able to help with that topic, but I'd love to look at what the stars say for you. "
            "Would you like to share your birth details or ask an astrology question?"
        )

    ai_msg = AIMessage(content=response)
    return {"messages": [ai_msg], "final_response": response}


def _summarize_chart(chart: dict) -> str:
    """Create a short text summary of key chart points."""
    if not chart or "planets" not in chart:
        return "No chart data."
    planets = chart["planets"]
    asc = chart.get("ascendant", {})
    lines = []
    for name in ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"]:
        if name in planets:
            p = planets[name]
            retro = " (R)" if p.get("retrograde") else ""
            lines.append(f"{name}: {p['sign']} {p['degree']:.1f}°{retro}")
    if asc:
        lines.append(f"ASC: {asc.get('sign')} {asc.get('degree', 0):.1f}°")
    return "\n".join(lines)


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph() -> Any:
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("reason", reason_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("respond_direct", respond_direct_node)

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"reason": "reason", "respond_direct": "respond_direct"},
    )

    workflow.add_conditional_edges(
        "reason",
        should_use_tools,
        {"tool_executor": "tool_executor", "respond": "respond"},
    )

    workflow.add_edge("tool_executor", "reason")  # loop back for multi-turn tool use
    workflow.add_edge("respond", END)
    workflow.add_edge("respond_direct", END)

    return workflow.compile()


# ── AstroGraph wrapper ─────────────────────────────────────────────────────────

class AstroGraph:
    """Wrapper around the compiled LangGraph with session memory."""

    def __init__(self):
        self.graph = build_graph()
        self._sessions: dict[str, AgentState] = {}

    def _get_state(self, session_id: str, birth_details: BirthDetails | None) -> AgentState:
        state = self._sessions.get(session_id, AgentState())
        if birth_details:
            state = state.model_copy(update={"birth_details": birth_details})
        return state

    async def stream(
        self,
        session_id: str,
        message: str,
        birth_details: BirthDetails | None = None,
    ) -> AsyncGenerator[dict, None]:
        state = self._get_state(session_id, birth_details)
        human_msg = HumanMessage(content=message)

        new_messages = list(state.messages) + [human_msg]
        input_state = state.model_copy(update={"messages": new_messages})

        final_state = None
        async for chunk in self.graph.astream(
            input_state.model_dump(),
            stream_mode="updates",
        ):
            for node_name, node_output in chunk.items():
                event: dict = {"type": "node", "node": node_name}

                # Emit tool activity
                if node_name == "tool_executor" and "messages" in node_output:
                    for msg in node_output["messages"]:
                        if isinstance(msg, ToolMessage):
                            event["type"] = "tool_result"
                            event["tool_name"] = msg.name
                            try:
                                event["tool_output"] = json.loads(msg.content)
                            except Exception:
                                event["tool_output"] = msg.content
                            yield event

                # Emit AI text tokens
                elif "messages" in node_output:
                    for msg in node_output["messages"]:
                        if isinstance(msg, AIMessage) and msg.content:
                            yield {"type": "text", "content": msg.content}
                        elif isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield {
                                    "type": "tool_call",
                                    "tool_name": tc["name"],
                                    "tool_args": tc["args"],
                                }

                if node_output:
                    final_state = node_output

        # Persist updated session
        if final_state and "final_response" in final_state:
            updated = state.model_copy(
                update={
                    "messages": new_messages + [AIMessage(content=final_state["final_response"])],
                    "birth_details": birth_details or state.birth_details,
                    "birth_chart": final_state.get("birth_chart", state.birth_chart),
                }
            )
            self._sessions[session_id] = updated

    async def invoke(
        self,
        session_id: str,
        message: str,
        birth_details: BirthDetails | None = None,
    ) -> dict:
        """Non-streaming invocation for eval."""
        state = self._get_state(session_id, birth_details)
        human_msg = HumanMessage(content=message)
        new_messages = list(state.messages) + [human_msg]
        input_state = state.model_copy(update={"messages": new_messages})

        start = time.time()
        result = await self.graph.ainvoke(input_state.model_dump())
        latency = time.time() - start

        # Persist
        if result.get("final_response"):
            updated = input_state.model_copy(
                update={
                    "messages": new_messages + [AIMessage(content=result["final_response"])],
                    "birth_chart": result.get("birth_chart", state.birth_chart),
                }
            )
            self._sessions[session_id] = updated

        return {
            "response": result.get("final_response", ""),
            "intent": result.get("intent", "unknown"),
            "tool_calls_made": result.get("tool_calls_made", []),
            "step_count": result.get("step_count", 0),
            "latency_s": round(latency, 3),
            "error": result.get("error"),
        }

    async def get_history(self, session_id: str) -> list[dict]:
        state = self._sessions.get(session_id)
        if not state:
            return []
        messages = []
        for msg in state.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return messages

    async def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
