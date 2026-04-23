# Vulcan OmniPro 220 — AI Technical Assistant

A multimodal reasoning agent that answers deep technical questions about the Vulcan OmniPro 220 welding system using the Claude Agent SDK. Visual-first: duty cycle queries render as HTML tables, polarity setups render as SVG wiring diagrams, troubleshooting renders as flowcharts — never described in prose when they can be shown.

## Setup (under 2 minutes)

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
python preprocess.py          # extract text + render page images (run once)
uvicorn server:app --reload   # open http://localhost:8000
```

## Architecture

```
preprocess.py    — Renders each PDF page as PNG, extracts text, builds TF-IDF index
agent.py         — Claude claude-sonnet-4-6 agent with two tools:
                     search_manual(query)      → TF-IDF search over all docs
                     get_page_image(doc, page) → returns rendered page PNG (base64)
server.py        — FastAPI: POST /chat → agent, GET / → frontend
static/index.html — Vanilla JS chat UI; renders agent HTML/SVG responses inline
```

## Design Decisions

**Visual-first system prompt.** The agent is instructed to never describe a diagram in prose — SVG for wiring/polarity, HTML tables for duty cycles and settings, numbered HTML for troubleshooting. CSS class names and SVG templates are embedded directly in the prompt so every response is consistently styled.

**Page images as ground truth.** Critical manual content lives in images (process selection charts, duty cycle matrices, weld diagnosis photos, wiring schematics). Rather than imperfect OCR, the agent fetches the rendered page PNG via `get_page_image` and lets Claude's vision read it directly — no custom parsers needed.

**TF-IDF search without an embedding model.** Built at preprocessing time, this handles welding domain terminology (duty cycle, DCEN, DCEP, synergic, polarity) precisely without requiring a vector database or second model.

**Prompt caching.** The long system prompt (with SVG/table templates) is cached with `cache_control: ephemeral`, reducing latency and cost on every request.

**Multimodal input.** Users upload weld photos via button, drag-and-drop, or clipboard paste. The agent diagnoses visible defects (porosity, spatter, underfill, cold lap, burn-through) and returns a causes-and-fixes table.

## Example Queries

- *"What's the duty cycle for MIG welding at 200A on 240V?"* → color-coded duty cycle table
- *"How do I set up polarity for flux-core welding?"* → SVG cable routing diagram
- *"Show me TIG torch connection"* → SVG wiring diagram
- *"What wire size and settings for 3/8" steel?"* → settings matrix table
- *[weld photo] "What's wrong with this weld?"* → defect diagnosis with causes/fixes table

## Stack
- **Agent**: Anthropic Python SDK — Claude claude-sonnet-4-6
- **PDF processing**: PyMuPDF
- **Server**: FastAPI + Uvicorn
- **Frontend**: Vanilla HTML/CSS/JS (no build step)

---
1