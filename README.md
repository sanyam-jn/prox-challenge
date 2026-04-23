# Vulcan OmniPro 220 — AI Technical Assistant

> A multimodal reasoning agent that answers deep technical questions about the Vulcan OmniPro 220 welding system. Built on the Claude Agent SDK, it reads the 48-page owner's manual, quick-start guide, and selection chart — then responds with interactive calculators, SVG wiring diagrams, and duty cycle tables rather than plain text.

**Live demo:** https://prox-lyart-nine.vercel.app

---

## What it does

Most people who buy a multiprocess welder won't read 48 pages of dense technical documentation. But a machine with four welding processes, dual voltage input, synergic controls, and process-specific polarity requirements genuinely needs expert-level support.

This agent closes that gap. Ask it anything about the OmniPro 220 — setup, duty cycles, polarity, wire selection, troubleshooting — and it responds with the right level of detail and the right format for the question.

**It never describes a diagram when it can draw one.**

- Ask about duty cycle → interactive calculator with amperage slider
- Ask about polarity → SVG wiring diagram with labeled cable routing
- Ask about settings → parameter table with recommended wire size, voltage, feed speed
- Upload a weld photo → defect diagnosis with causes-and-fixes table
- Ask about troubleshooting → click-through decision tree
- Ask follow-up questions → full multi-turn memory across the conversation

---

## Setup

```bash
git clone https://github.com/sanyam-jn/prox-challenge
cd prox-challenge

python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # paste your Anthropic API key
uvicorn server:app --reload  # open http://localhost:8000
```

The preprocessed manual index and page images are already committed — no need to run `preprocess.py` unless you change the source PDFs.

---

## How it works

### Retrieval

The three source documents (owner's manual, quick-start guide, selection chart) are preprocessed into two artifacts:

- **`data/index.json`** — full page text + extracted headings for all 51 pages
- **`data/idf.json`** — TF-IDF inverse-document-frequency table built over the corpus

At query time, `search_manual(query)` scores each page by TF-IDF and returns the top-k text excerpts. This handles domain-specific terminology (DCEN, DCEP, synergic, duty cycle, flux-cored) without needing a vector database or embedding model.

### Vision

The critical information in this manual isn't in the text — it's in images. The duty cycle matrix, wiring schematics, polarity diagrams, process selection chart, and weld diagnosis photos are all embedded as images inside the PDF.

The agent has a `get_page_image(doc, page)` tool that returns a rendered 1.5x PNG of any manual page. Claude's vision reads these directly. When a user asks about polarity setup, the agent fetches the relevant schematic page and sees exactly what a human reading the manual would see — then generates a clean SVG diagram for the user based on what it found.

### Agentic loop

```
User question + conversation history
  → search_manual(query)          # find relevant pages by text
  → get_page_image(doc, page)     # read diagrams on those pages
  → stream response tokens        # HTML/SVG/artifact appears word-by-word
```

The agent streams status updates in real time so users see which pages are being read while the response is forming. The final response streams token-by-token (Claude Streaming API), appearing as a plain-text preview with a blinking cursor that transitions to fully rendered HTML when generation completes.

### Multi-turn conversation

Each request includes the last 6 exchanges as conversation history, prepended to the messages array before the current user message. History is stored in memory only (never persisted to `localStorage`). The "New conversation" button clears history and resets the UI without a page reload.

### Visual-first output

The system prompt instructs the agent to never describe a visual when it can render one. It embeds concrete templates for three output tiers:

**Tier 1 — Interactive artifacts**: For calculators, configurators, and flowcharts, the agent generates a complete sandboxed HTML/JS application wrapped in `<artifact>` tags. The frontend renders it in a sandboxed iframe with a fullscreen expand button. Examples:
- Duty cycle calculator with process/voltage dropdowns and amperage slider
- Settings wizard: process → material → thickness → recommended parameters
- Troubleshooting flowchart: click-through decision tree

**Tier 2 — Manual page images**: After fetching a page that contains a diagram the user needs to see, the agent embeds it directly in the response using `<img class="manual-page" src="/pages/FILENAME">`.

**Tier 3 — Inline HTML**: For simple structured data, HTML tables and SVG wiring diagrams with shared CSS classes for consistent styling.

### Prompt caching

The system prompt (including SVG and table templates) is marked `cache_control: ephemeral`. The Anthropic API caches it between requests, reducing latency and cost on the first tool-call round-trip.

### Artifact iframe sizing

Interactive artifacts are rendered in sandbox iframes (`sandbox="allow-scripts"`). A `postMessage` resize reporter is injected into each artifact at render time — it fires on `load`, on a `ResizeObserver`, and twice on a timeout to catch late-rendering JS. The parent window listens and sets the iframe height to the artifact's actual rendered size, eliminating the blank space that plagues `onload` + `scrollHeight` approaches.

---

## Architecture

```
prox-challenge/
├── agent.py          Claude agent — tool definitions, agentic loop, streaming generator
├── server.py         FastAPI — local dev server with static file serving
├── preprocess.py     PDF → page PNGs + TF-IDF index (run once locally)
│
├── api/
│   └── index.py      Vercel serverless entry point (same routes, no static serving)
│
├── public/
│   ├── index.html    Frontend — chat UI, artifact iframe rendering, settings panel
│   └── pages/        Rendered manual page PNGs (51 pages, committed to repo)
│
├── data/
│   ├── index.json    Page text + headings index
│   └── idf.json      TF-IDF weights
│
├── files/
│   ├── owner-manual.pdf
│   ├── quick-start-guide.pdf
│   └── selection-chart.pdf
│
└── vercel.json       Vercel routing config
```

---

## Settings

The live demo has a ⚙ Settings panel (top right). Paste your [Anthropic API key](https://console.anthropic.com) and choose a model. Your key is stored only in your browser's `localStorage` — it never touches any server other than Anthropic's API directly.

Model options:
- **Sonnet** — fast and capable, good default
- **Opus** — most powerful, better for complex diagrams and cross-referencing
- **Haiku** — fastest, good for simple lookups

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent | Anthropic Claude API (claude-sonnet-4-6 default) |
| PDF processing | PyMuPDF |
| Search | TF-IDF (no embedding model) |
| Backend | FastAPI + Uvicorn |
| Hosting | Vercel (serverless Python) |
| Frontend | Vanilla HTML/CSS/JS — no build step |

---

## Running preprocess.py

Only needed if you modify the source PDFs:

```bash
source .venv/bin/activate
python preprocess.py
# outputs: public/pages/*.png + data/index.json + data/idf.json
```
