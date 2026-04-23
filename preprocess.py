"""
Run once to extract text and render page images from the PDFs.
Output: data/index.json + data/idf.json + public/pages/*.png
"""
import fitz  # PyMuPDF
import json
import math
import os
from collections import Counter
from pathlib import Path

DOCS = [
    ("files/owner-manual.pdf",    "owner-manual"),
    ("files/quick-start-guide.pdf", "quick-start-guide"),
    ("files/selection-chart.pdf",   "selection-chart"),
]


def extract_headings(page) -> list[str]:
    headings = []
    try:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip() and (
                        span.get("size", 0) > 11 or span.get("flags", 0) & 16
                    ):
                        headings.append(span["text"].strip())
    except Exception:
        pass
    return headings


def preprocess():
    Path("data").mkdir(exist_ok=True)
    Path("public/pages").mkdir(parents=True, exist_ok=True)
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

            # Render at 1.5x and save to public/pages/ for static serving
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            img_filename = f"{doc_name}_p{page_num}.png"
            pix.save(f"public/pages/{img_filename}")

            index.append({
                "doc": doc_name,
                "page": page_num,
                "text": text,
                "headings": headings,
                "image": img_filename,
            })

        doc.close()

    # Build TF-IDF index
    N = len(index)
    df: Counter = Counter()
    for entry in index:
        df.update(set(entry["text"].lower().split()))
    idf = {t: math.log((N + 1) / (c + 1)) for t, c in df.items()}

    with open("data/index.json", "w") as f:
        json.dump(index, f)
    with open("data/idf.json", "w") as f:
        json.dump(idf, f)

    print(f"\nDone — {len(index)} pages indexed, images in public/pages/")


if __name__ == "__main__":
    preprocess()
