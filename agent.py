"""
Vulcan OmniPro 220 multimodal reasoning agent.
Uses Claude with tool_use to search the manual and read page images,
then generates rich HTML/SVG/interactive-artifact responses.
"""
import anthropic
import base64
import json
import os
from collections import Counter
from dotenv import load_dotenv
from typing import Generator

load_dotenv()

MODEL = os.getenv("MODEL", "claude-sonnet-4-6")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --------------------------------------------------------------------------- #
# Knowledge base
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


def page_img_filename(doc: str, page: int) -> str:
    return f"{doc}_p{page}.png"


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are the Vulcan OmniPro 220 technical expert. You have full access to the \
owner's manual (48 pages), quick-start guide, and selection chart via your tools. \
Users are typically in their garage setting up or troubleshooting their welder.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIER 1 — INTERACTIVE ARTIFACTS (highest priority):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For questions involving calculation, configuration, or multi-step troubleshooting,
generate a FULLY INTERACTIVE HTML/JS application as an artifact. This is the most
important part of your job — interactive tools beat static text every time.

Wrap artifacts exactly like this:
<artifact title="Descriptive Name">
<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #07070f; color: #e2e8f0;
         font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         padding: 20px; min-height: 100vh; }
  /* amber accent: #f59e0b | surface: #111120 | border: #ffffff10 */
</style>
</head>
<body>
  <!-- interactive content -->
  <script>/* inline JS only, no external deps */</script>
</body>
</html>
</artifact>

WHEN TO CREATE AN ARTIFACT:
• "What's the duty cycle at X amps / X voltage?" → Duty Cycle Calculator (sliders + visual gauge)
• "What settings for [material] [thickness]?" → Settings Configurator (step-through wizard)
• Troubleshooting questions → Interactive Flowchart (click-through decision tree)
• Polarity / wiring setup → Animated SVG diagram with labeled cable routing
• Anything comparing specs across processes/voltages → Interactive comparison tool

ARTIFACT DESIGN RULES:
- Colors: bg #07070f, surface #111120, accent #f59e0b, text #e2e8f0, green #22c55e, red #ef4444
- All CSS and JS inline, zero external URLs
- Use REAL data from the manual (actual duty cycle %, actual wire sizes, actual amp ranges)
- Add genuine interactivity: sliders, dropdowns, step buttons, hover states
- Artifacts can coexist with text explanation in the same response

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIER 2 — SHOW MANUAL PAGES (for diagrams the user must see):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After fetching a page image that contains a DIAGRAM, SCHEMATIC, or VISUAL TABLE
the user needs to see, include it using:
<img class="manual-page" src="/pages/FILENAME" alt="Manual page N">
(where FILENAME is e.g. owner-manual_p12.png)
Do this for wiring schematics, process selection charts, weld diagnosis photos,
wire feed diagrams, front panel diagrams. Do NOT show pages that are text-only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIER 3 — INLINE HTML (for simple structured data):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CSS classes available in the UI:
  <div class="viz">…</div>            — wraps any visualization
  <table class="dtable">…</table>    — data tables
  <div class="warn-box">⚠ …</div>   — red safety warning
  <div class="tip-box">💡 …</div>   — green tip
  .good (green text) .warn (amber) .bad (red) — table cell classes

SVG WIRING DIAGRAM base (customize cables and labels per process):
<div class="viz"><svg width="580" height="360" viewBox="0 0 580 360"
  xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e;border-radius:10px;display:block">
  <rect x="190" y="60" width="200" height="210" rx="10" fill="#0f172a" stroke="#f59e0b" stroke-width="2"/>
  <text x="290" y="92" text-anchor="middle" fill="#f59e0b" font-family="monospace" font-size="13" font-weight="bold">OmniPro 220</text>
  <circle cx="230" cy="250" r="18" fill="#7f1d1d" stroke="#ef4444" stroke-width="2"/>
  <text x="230" y="255" text-anchor="middle" fill="white" font-family="monospace" font-size="15" font-weight="bold">+</text>
  <circle cx="350" cy="250" r="18" fill="#1e3a8a" stroke="#3b82f6" stroke-width="2"/>
  <text x="350" y="255" text-anchor="middle" fill="white" font-family="monospace" font-size="15" font-weight="bold">−</text>
  <!-- add cables, components, labels here -->
</svg></div>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Direct answer (1-2 sentences)
2. Artifact OR inline visualization (always when data is involved)
3. Manual page image (if a relevant diagram exists)
4. Safety warnings: <div class="warn-box">⚠ …</div>
5. Tips: <div class="tip-box">💡 …</div>
6. Source: <em>Owner's Manual p.XX</em>

For weld photo diagnosis: identify the specific defect, give a causes-and-fixes table.\
"""

# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
TOOLS = [
    {
        "name": "search_manual",
        "description": (
            "TF-IDF search over the Vulcan OmniPro 220 owner's manual, quick-start guide, "
            "and selection chart. Returns text excerpts and page numbers. Call this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query using welding terms"},
                "top_k": {"type": "integer", "description": "Pages to return (default 5)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_page_image",
        "description": (
            "Get a rendered image of a specific manual page to examine diagrams, schematics, "
            "duty cycle tables, selection charts, and weld diagnosis photos. Use after "
            "search_manual to visually inspect pages with diagrams."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc": {
                    "type": "string",
                    "enum": ["owner-manual", "quick-start-guide", "selection-chart"],
                },
                "page": {"type": "integer", "description": "Page number (1-indexed)"},
            },
            "required": ["doc", "page"],
        },
    },
]


# --------------------------------------------------------------------------- #
# Agent — streaming generator
# --------------------------------------------------------------------------- #
def run_agent_stream(
    message: str, images_b64: list[str] | None = None
) -> Generator[dict, None, None]:
    """
    Yields event dicts:
      {"type": "status", "text": "..."}
      {"type": "response", "text": "..."}
      {"type": "error", "text": "..."}
    """
    user_content: list = []
    for img in (images_b64 or []):
        media_type = "image/png" if img.startswith("iVBOR") else "image/jpeg"
        user_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": img},
        })
    user_content.append({"type": "text", "text": message})

    messages = [{"role": "user", "content": user_content}]
    system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    try:
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=8192,
                system=system,
                tools=TOOLS,
                messages=messages,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                final = "".join(b.text for b in response.content if b.type == "text")
                yield {"type": "response", "text": final}
                return

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tu in tool_uses:
                if tu.name == "search_manual":
                    yield {"type": "status", "text": f'Searching: "{tu.input["query"]}"…'}
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
                    yield {"type": "status", "text": f"Reading {doc} page {page}…"}
                    img_b64 = get_page_image_b64(doc, page)
                    if img_b64:
                        content = [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"[{doc}, page {page}] "
                                    f"Image filename: {page_img_filename(doc, page)}"
                                ),
                            },
                        ]
                    else:
                        content = f"Page {page} not found in {doc}."
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": content,
                    })

            messages.append({"role": "user", "content": tool_results})

    except Exception as exc:
        yield {"type": "error", "text": str(exc)}
