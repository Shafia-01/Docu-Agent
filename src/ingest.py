# src/ingest.py
"""
PDF ingestion: extracts text, images, and tables from PDFs and produces
chunked passages ready for embedding.
"""
import os
import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm
import hashlib
import json
from typing import List, Dict

ASSET_DIR = "data/assets"
os.makedirs(ASSET_DIR, exist_ok=True)

def _save_image_from_page(doc, page, img_index, pdf_name):
    # Pull image xref and save to disk
    for img_index_tuple, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        ext = "png"
        if pix.n > 4:  # CMYK or with alpha
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img_name = f"{pdf_name}_page{page.number+1}_img{img_index_tuple}.{ext}"
        img_path = os.path.join(ASSET_DIR, img_name)
        pix.save(img_path)
        pix = None
        yield img_path

def extract_pdf(path: str) -> Dict:
    """
    Returns a dict with keys: id, path, title (best effort), pages (list),
    full_text, tables (list of saved csv paths).
    """
    doc = fitz.open(path)
    pdf_name = os.path.splitext(os.path.basename(path))[0]
    pages = []
    full_text = []
    all_tables = []
    for p in doc:
        text = p.get_text("text")
        # images
        images = list(_save_image_from_page(doc, p, 0, pdf_name))
        pages.append({"page": p.number+1, "text": text, "images": images})
        full_text.append(f"[PAGE {p.number+1}]\n{text}")

    # extract tables using pdfplumber (best-effort)
    try:
        with pdfplumber.open(path) as pdf:
            for i, p in enumerate(pdf.pages):
                tables = p.extract_tables()
                for t_idx, table in enumerate(tables):
                    # Save as simple CSV
                    table_path = os.path.join(ASSET_DIR, f"{pdf_name}_page{i+1}_table{t_idx}.csv")
                    with open(table_path, "w", encoding="utf-8") as fh:
                        for row in table:
                            row_clean = ["" if x is None else str(x).replace(",", " ") for x in row]
                            fh.write(",".join(row_clean) + "\n")
                    all_tables.append({"page": i+1, "csv": table_path})
    except Exception:
        # Silently continue; table extraction is optional
        pass

    full_text_str = "\n\n".join(full_text)
    # guess title: first line of text usually contains title
    title_guess = full_text_str.strip().split("\n", 1)[0]
    doc_id = hashlib.sha1(path.encode()).hexdigest()[:10]
    return {
        "id": doc_id,
        "path": path,
        "title": title_guess,
        "pages": pages,
        "full_text": full_text_str,
        "tables": all_tables
    }

def chunk_text(text: str, chunk_chars: int = 1500, overlap: int = 200) -> List[Dict]:
    """
    Simple character-based chunking with overlap. Returns list of dicts:
    {"id":..., "text":..., "meta":{...}}
    """
    text = text.replace("\r", "")
    chunks = []
    start = 0
    doc_len = len(text)
    chunk_id = 0
    while start < doc_len:
        end = min(start + chunk_chars, doc_len)
        chunk_text = text[start:end]
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": chunk_text,
            "meta": {"start": start, "end": end}
        })
        chunk_id += 1
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python src/ingest.py file.pdf")
        sys.exit(1)
    path = sys.argv[1]
    out = extract_pdf(path)
    chunks = chunk_text(out["full_text"])
    print("Title guess:", out["title"])
    print("Chunks:", len(chunks))
    # Optionally save a small metadata file
    meta_path = f"data/{os.path.splitext(os.path.basename(path))[0]}_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("Saved metadata to", meta_path)
