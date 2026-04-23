"""
Vulcan OmniPro 220 multimodal reasoning agent.
Uses Claude with tool_use to search the manual and read page images,
then generates rich HTML/SVG responses.
"""
import anthropic
import base64
import json
import os
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "claude-sonnet-4-6")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --------------------------------------------------------------------------- #
# Knowledge base (loaded once at import)
# --------------------------------------------------------------------------- #
with open("data/index.json") as f:
    PAGES: list[dict] = json.load(f)

with open("data/idf.json") as f:
    IDF: dict[str, float] = json.load(f)


def _tfidf_score(page: dict, query_terms: list[str]) -> float:
    text = (page["text"] + " " + " ".join(page.get("headings", []))).lower()
    words = text.split()
    tf = Counter(words)
    total = max(len(words), 1)
    return sum((tf[t] / total) * IDF.get(t, 0) for t in query_terms if t in tf)


def search_manual(query: str, top_k: int = 6) -> list[dict]:
    terms = query.lower().split()
    scored = [(_tfidf_score(p, terms), p) for p in PAGES]
    scored.sort(key=lambda x: -x[0])
    return [p for score, p in scored if score > 0][:top_k]


def get_page_image_b64(doc: str, page: int) -> str | None:
    path = f"data/pages/{doc}_p{page}.png"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are the Vulcan OmniPro 220 technical expert. You have full access to the \
owner's manual (48 pages), quick-start guide, and selection chart. Users are \
typically in their garage, hands dirty, needing accurate help fast.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL-FIRST RULE — most important instruction:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Never describe a visual when you can render it. Specifically:

• Duty cycle data           → always produce an HTML table
• Polarity / wiring setup   → always produce an SVG connection diagram
• Settings parameters       → HTML table with process/voltage columns
• Troubleshooting steps     → numbered HTML list with decision points
• Process comparisons       → HTML comparison table

Your HTML and SVG render directly in the browser (innerHTML). Write complete,
self-contained HTML fragments. Do NOT use markdown code fences around HTML.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CSS CLASSES AVAILABLE (already styled in the UI):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<div class="viz">…</div>           — wraps any visualization
<table class="dtable">…</table>   — data tables
<div class="warn-box">…</div>     — red safety warning box
<div class="tip-box">…</div>      — green tip / note box
<div class="step-list">…</div>    — numbered step container

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SVG WIRING DIAGRAM — base template to customize:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<div class="viz">
<svg width="580" height="360" viewBox="0 0 580 360"
     xmlns="http://www.w3.org/2000/svg"
     style="background:#1a1a2e;border-radius:10px;display:block">
  <defs>
    <marker id="ah" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0,8 3,0 6" fill="#94a3b8"/>
    </marker>
  </defs>
  <!-- Welder body -->
  <rect x="190" y="60" width="200" height="210" rx="10"
        fill="#0f172a" stroke="#f59e0b" stroke-width="2"/>
  <text x="290" y="92" text-anchor="middle" fill="#f59e0b"
        font-family="monospace" font-size="13" font-weight="bold">OmniPro 220</text>
  <!-- Positive output -->
  <circle cx="230" cy="250" r="18" fill="#7f1d1d" stroke="#ef4444" stroke-width="2"/>
  <text x="230" y="255" text-anchor="middle" fill="white"
        font-family="monospace" font-size="15" font-weight="bold">+</text>
  <text x="230" y="282" text-anchor="middle" fill="#94a3b8"
        font-family="monospace" font-size="10">OUTPUT (+)</text>
  <!-- Negative output -->
  <circle cx="350" cy="250" r="18" fill="#1e3a8a" stroke="#3b82f6" stroke-width="2"/>
  <text x="350" y="255" text-anchor="middle" fill="white"
        font-family="monospace" font-size="15" font-weight="bold">−</text>
  <text x="350" y="282" text-anchor="middle" fill="#94a3b8"
        font-family="monospace" font-size="10">OUTPUT (−)</text>
  <!-- Add cables, components, and labels below this line -->
  <!-- Red cable: stroke="#ef4444" stroke-width="4" -->
  <!-- Black cable: stroke="#475569" stroke-width="4" -->
</svg>
</div>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HTML TABLE — base template:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<div class="viz">
<table class="dtable">
  <thead><tr><th>Parameter</th><th>120V</th><th>240V</th></tr></thead>
  <tbody>
    <tr><td>Max Output</td><td>130A</td><td class="good">220A</td></tr>
    <tr><td>Duty @ max</td><td class="warn">20%</td><td class="good">60%</td></tr>
  </tbody>
</table>
</div>
CSS value classes: good (green), warn (amber), bad (red)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE STRUCTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Direct answer — 1-2 sentences
2. Visual — table or SVG diagram (required whenever data is involved)
3. Safety warnings using <div class="warn-box">⚠ …</div>
4. Tips using <div class="tip-box">💡 …</div>
5. Source: <em>Owner's Manual p.XX</em> or Quick-Start Guide p.XX

When the user uploads a weld photo: identify the specific defect visible \
(porosity, spatter, undercut, cold lap, burn-through, lack of fusion, etc.) \
and produce a causes-and-fixes table.

Be precise. Cross-reference multiple manual sections when needed. \
Always cite page numbers.\
"""

# --------------------------------------------------------------------------- #
# Tool definitions
# --------------------------------------------------------------------------- #
TOOLS = [
    {
        "name": "search_manual",
        "description": (
            "TF-IDF search over the full Vulcan OmniPro 220 documentation "
            "(owner's manual, quick-start guide, selection chart). Returns text "
            "excerpts and page numbers. Always call this first to locate relevant pages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query using welding terminology",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of pages to return (default 5, max 8)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_page_image",
        "description": (
            "Fetch a rendered image of a specific manual page. Use this to examine "
            "diagrams, wiring schematics, duty cycle matrices, and process selection "
            "charts that are embedded as images in the PDF."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc": {
                    "type": "string",
                    "description": "Document name",
                    "enum": ["owner-manual", "quick-start-guide", "selection-chart"],
                },
                "page": {
                    "type": "integer",
                    "description": "Page number (1-indexed)",
                },
            },
            "required": ["doc", "page"],
        },
    },
]


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #
def run_agent(message: str, images_b64: list[str] | None = None) -> str:
    """Run the agentic tool-use loop and return final HTML response."""

    user_content: list = []
    for img in (images_b64 or []):
        media_type = "image/png" if img.startswith("iVBOR") else "image/jpeg"
        user_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": img},
        })
    user_content.append({"type": "text", "text": message})

    messages = [{"role": "user", "content": user_content}]

    # Cache the system prompt across requests
    system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            return "".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            if tu.name == "search_manual":
                results = search_manual(tu.input["query"], tu.input.get("top_k", 5))
                if results:
                    body = "\n\n---\n\n".join(
                        f"[{r['doc']} — page {r['page']}]\n"
                        f"Headings: {', '.join(r['headings'][:6]) or '(none)'}\n\n"
                        f"{r['text'][:1800].strip()}"
                        for r in results
                    )
                else:
                    body = "No relevant pages found."
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": body,
                })

            elif tu.name == "get_page_image":
                doc, page = tu.input["doc"], tu.input["page"]
                img_b64 = get_page_image_b64(doc, page)
                if img_b64:
                    content = [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": f"[{doc}, page {page}]"},
                    ]
                else:
                    content = f"Page {page} not found in {doc}."
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": content,
                })

        messages.append({"role": "user", "content": tool_results})
