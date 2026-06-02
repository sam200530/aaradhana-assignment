# Aradhana — AstroAgent

> *Your daily spiritual companion, powered by real planetary data.*

A full-stack AI astrologer: a stateful LangGraph agent that computes real birth charts using Swiss Ephemeris, reasons over live planetary transits, and responds with warmth through a polished React interface.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LangGraph Agent Graph                    │
│                                                              │
│  START → [classify_intent] ──────────────────────────────┐  │
│                │                                          │  │
│         intent routing                                    │  │
│                │                                          │  │
│     ┌──────────┴──────────┐                              │  │
│     ↓                     ↓                              │  │
│  [reason]           [respond_direct]  ← off_topic /      │  │
│     │                     │              guardrails       │  │
│     ↓                     │                              │  │
│  has tool calls?          │                              │  │
│     │                     │                              │  │
│  ┌──┴──┐                  │                              │  │
│  ↓     ↓                  │                              │  │
│ [tool_  [respond] ←───────┘                              │  │
│ executor]    ↑                                           │  │
│  │           │           TOOLS:                          │  │
│  └───────────┘           • geocode_place                 │  │
│   (loop until            • compute_birth_chart           │  │
│    no tool calls)        • get_daily_transits            │  │
│                          • knowledge_lookup              │  │
└─────────────────────────────────────────────────────────────┘
                    ↕  FastAPI + SSE
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend                          │
│  Birth Form → Chat Interface → Streaming Responses          │
│  Tool Activity Display → Session Persistence               │
└─────────────────────────────────────────────────────────────┘
```

### Node Descriptions

| Node | Role |
|---|---|
| `classify_intent` | LLM classifies message: `chart_request`, `daily_horoscope`, `free_question`, `off_topic` |
| `reason` | Main Claude node with all 4 tools bound. Decides what to call. |
| `tool_executor` | Executes tool calls, returns `ToolMessage`s, loops back to `reason` |
| `respond` | Synthesizes final answer if LLM gave tool calls then stopped |
| `respond_direct` | Fast path for off-topic/guardrail responses without tool use |

---

## Setup

### Prerequisites
- Python 3.11+
- Node 18+
- Anthropic API key (`ANTHROPIC_API_KEY`)

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Start the API server
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm start
# Opens http://localhost:3000
```

### Evaluation

```bash
cd backend

# Full eval with LLM judge (requires running API server)
python eval/run_eval.py

# Fast run, no LLM judge
python eval/run_eval.py --no-judge

# Limit to first 5 cases
python eval/run_eval.py --max-cases 5
```

---

## Tool Implementation Notes

### `compute_birth_chart`
- Uses **pyswisseph** (Python bindings to Swiss Ephemeris — the same engine used by professional astrology software)
- Computes all 10 classical planets (Sun through Pluto) in Tropical zodiac
- House system: **Placidus** (most widely used Western system)
- Requires geocoded lat/lon/timezone — always calls `geocode_place` first

### `geocode_place`
- Uses **Nominatim** (OpenStreetMap) for geocoding — free, no API key needed
- **TimezoneFinder** for accurate local timezone without an API call
- Returns full address, lat, lon, timezone

### `get_daily_transits`
- Computes today's planetary positions at noon UTC via ephemeris
- If natal chart is provided, computes all major aspects (conjunction, sextile, square, trine, opposition, quincunx) with standard orbs

### `knowledge_lookup`
- Curated astrology knowledge base embedded directly (no vector DB required for small scale)
- Covers: Sun/Moon/planet signs, houses, aspects, Saturn return, retrograde, lunar nodes, disclaimers
- Keyword-frequency scoring for retrieval
- For production: replace with FAISS + sentence-transformers embeddings

---

## Project Structure

```
astroagent/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, SSE streaming endpoint
│   │   ├── agents/
│   │   │   ├── graph.py         # LangGraph graph definition + AstroGraph wrapper
│   │   │   └── state.py         # AgentState, BirthDetails schemas
│   │   └── tools/
│   │       └── astro_tools.py   # All 4 tools: geocode, chart, transits, knowledge
│   ├── eval/
│   │   ├── golden_set_v1.jsonl  # 25 versioned test cases
│   │   ├── run_eval.py          # Evaluation harness (one-command runner)
│   │   ├── results_log.csv      # Historical scorecard log
│   │   └── latest_run.json      # Full results from most recent run
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.js               # Main React app (form, chat, streaming)
    │   └── App.css              # Deep indigo celestial theme
    └── public/index.html
```

---

## Evaluation Design

See `EVALUATION.md` for the full reflection. Key design decisions:

- **EV01** — Golden set of 25 cases committed as `eval/golden_set_v1.jsonl`, covering 8 categories
- **EV02** — Deterministic checks (intent, tool calls, keyword presence, step budget, safety) run in pure Python
- **EV03** — LLM judge uses Claude Haiku (cheaper) with a 5-dimension rubric, spot-checked against manual expectations
- **EV04** — Every run logs latency (p50/p95), estimated token cost, step count, failure rate
- **EV05** — Explicit test cases for impossible dates, missing time, prompt injection, safety railings
- **EV06** — `python eval/run_eval.py` prints a full scorecard and appends to `results_log.csv`

---

## Known Limitations

1. **No persistent database** — Sessions are stored in memory; restart clears all history. Production would use SQLite/Postgres with serialized MessageState.

2. **Geocoding rate limits** — Nominatim has a 1 req/sec limit. Rapid eval runs may hit this; a production system would cache geocodes or use a paid provider.

3. **Knowledge RAG is keyword-based** — For richer retrieval, replace with FAISS + sentence-transformers. The structure is there; the embedding layer is the swap.

4. **No conversation memory across server restarts** — Messages live in `AstroGraph._sessions`. A Redis-backed checkpointer (LangGraph supports this natively) would fix this.

5. **Frontend session IDs are per tab** — Multiple tabs = multiple sessions. Not a problem for the assignment; would need shared auth in production.

6. **LLM judge cost** — Full eval run over 25 cases with Claude Haiku judge costs ~$0.01–0.03 USD. The `--no-judge` flag skips this.

---

## Safety Guardrails

Built into the system prompt and tested in the golden set:
- Never presents readings as medical, legal, or financial certainty
- Redirects to professionals for health/legal/finance questions
- Does not claim to predict exact future events (death, winning lawsuits, etc.)
- Does not reveal system prompt on injection attempts
- Provides crisis resources framing for distress signals
- Off-topic requests are redirected warmly to astrology

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 0.2+ |
| LLM | Claude Sonnet (claude-sonnet-4-20250514) |
| Ephemeris | pyswisseph (Swiss Ephemeris bindings) |
| Geocoding | Geopy (Nominatim) + TimezoneFinder |
| API server | FastAPI + SSE |
| Frontend | React 18, vanilla CSS |
| Evaluation judge | Claude Haiku |

---

*Aradhana offers cosmic reflection, not medical, legal, or financial advice. ✦*
