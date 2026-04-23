"""
Run once to extract text and render page images from the PDFs.
Output: data/index.json + data/pages/*.png
"""
import fitz  # PyMuPDF
import json
import os
import math
from pathlib import Path
from collections import Counter

DOCS = [
    ("files/owner-manual.pdf", "owner-manual"),
    ("files/quick-start-guide.pdf", "quick-start-guide"),
    ("files/selection-chart.pdf", "selection-chart"),
]

def extract_headings(page) -> list[str]:
    headings = []
    try:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0)
                    flags = span.get("flags", 0)
                    text = span.get("text", "").strip()
                    if text and (size > 11 or flags & 16):  # bold or large
                        headings.append(text)
    except Exception:
        pass
    return headings


def preprocess():
    os.makedirs("data/pages", exist_ok=True)
    index = []

    for pdf_path, doc_name in DOCS:
        if not os.path.exists(pdf_path):
            print(f"  skipping {pdf_path} (not found)")
            continue

        doc = fitz.open(pdf_path)
        print(f"Processing {pdf_path} — {len(doc)} pages")

        for i in range(len(doc)):
            page = doc[i]
            page_num = i + 1

            text = page.get_text("text")
            headings = extract_headings(page)

            # Render at 1.5x for clarity without giant files
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            img_filename = f"{doc_name}_p{page_num}.png"
            pix.save(f"data/pages/{img_filename}")

            index.append({
                "doc": doc_name,
                "page": page_num,
                "text": text,
                "headings": headings,
                "image": img_filename,
            })

        doc.close()

    # Build IDF table for better search scoring
    N = len(index)
    df: Counter = Counter()
    for entry in index:
        terms = set(entry["text"].lower().split())
        df.update(terms)

    idf = {t: math.log((N + 1) / (c + 1)) for t, c in df.items()}

    with open("data/index.json", "w") as f:
        json.dump(index, f)

    with open("data/idf.json", "w") as f:
        json.dump(idf, f)

    print(f"\nDone — {len(index)} pages indexed, images in data/pages/")


if __name__ == "__main__":
    preprocess()
