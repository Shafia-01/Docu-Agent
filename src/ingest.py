# src/ingest.py
"""
Multi-document, multi-format ingestion pipeline (PDF, DOCX, HTML).
Now includes debug logs to see where ingestion might hang.
"""

import os
import fitz
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
import hashlib
from tqdm import tqdm
from typing import List, Dict

ASSET_DIR = "data/assets"
os.makedirs(ASSET_DIR, exist_ok=True)

# ---------------------------
# PDF Extraction
# ---------------------------
def _save_image_from_page(doc, page, pdf_name):
    print(f"Extracting images from page {page.number+1}...")
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        ext = "png"
        if pix.n > 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img_name = f"{pdf_name}_page{page.number+1}_img{img_index}.{ext}"
        img_path = os.path.join(ASSET_DIR, img_name)
        pix.save(img_path)
        pix = None
        yield img_path

def extract_pdf(path: str) -> Dict:
    print(f"[PDF] Opening {path}...")
    doc = fitz.open(path)
    pdf_name = os.path.splitext(os.path.basename(path))[0]
    print(f"[PDF] Document has {len(doc)} pages.")

    pages = []
    full_text = []
    all_tables = []

    for p in doc:
        print(f"Extracting text from page {p.number+1}...")
        text = p.get_text("text")
        images = list(_save_image_from_page(doc, p, pdf_name))
        pages.append({"page": p.number+1, "text": text, "images": images})
        full_text.append(f"[PAGE {p.number+1}]\n{text}")

    # Table extraction (can be slow)
    print(f"[PDF] Attempting table extraction with pdfplumber...")
    try:
        with pdfplumber.open(path) as pdf:
            for i, p in enumerate(pdf.pages):
                print(f"Extracting tables from page {i+1}...")
                tables = p.extract_tables()
                for t_idx, table in enumerate(tables):
                    table_path = os.path.join(
                        ASSET_DIR, f"{pdf_name}_page{i+1}_table{t_idx}.csv"
                    )
                    with open(table_path, "w", encoding="utf-8") as fh:
                        for row in table:
                            row_clean = [
                                "" if x is None else str(x).replace(",", " ") for x in row
                            ]
                            fh.write(",".join(row_clean) + "\n")
                    all_tables.append({"page": i+1, "csv": table_path})
    except Exception as e:
        print(f"[WARN] Table extraction failed: {e}")

    full_text_str = "\n\n".join(full_text)
    title_guess = full_text_str.strip().split("\n", 1)[0]
    doc_id = hashlib.sha1(path.encode()).hexdigest()[:10]

    return {
        "id": doc_id,
        "path": path,
        "title": title_guess,
        "pages": pages,
        "full_text": full_text_str,
        "tables": all_tables,
    }

# ---------------------------
# DOCX Extraction
# ---------------------------
def extract_docx(path: str) -> Dict:
    print(f"[DOCX] Opening {path}...")
    doc = Document(path)
    text = [para.text for para in doc.paragraphs if para.text.strip()]
    full_text_str = "\n".join(text)
    doc_id = hashlib.sha1(path.encode()).hexdigest()[:10]
    return {
        "id": doc_id,
        "path": path,
        "title": os.path.basename(path),
        "pages": [],
        "full_text": full_text_str,
        "tables": [],
    }

# ---------------------------
# HTML Extraction
# ---------------------------
def extract_html(path: str) -> Dict:
    print(f"[HTML] Opening {path}...")
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        text = soup.get_text(" ", strip=True)
    doc_id = hashlib.sha1(path.encode()).hexdigest()[:10]
    return {
        "id": doc_id,
        "path": path,
        "title": os.path.basename(path),
        "pages": [],
        "full_text": text,
        "tables": [],
    }

# ---------------------------
# Dispatcher
# ---------------------------
def extract_document(path: str) -> Dict:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path)
    elif ext == ".docx":
        return extract_docx(path)
    elif ext in [".html", ".htm"]:
        return extract_html(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

# ---------------------------
# Chunking
# ---------------------------

def chunk_text(text: str, chunk_chars: int = 1500, overlap: int = 200, max_chunks: int = 500):
    """
    Split text into overlapping chunks. 
    Safety: caps at max_chunks to avoid infinite loops on corrupted text.
    """
    print(f"[Chunking] Splitting text into chunks (size={chunk_chars}, overlap={overlap})...")

    # Defensive cleaning
    if not text or len(text.strip()) == 0:
        print("[Chunking] Warning: Empty text received.")
        return []

    text = text.replace("\r", "")
    chunks = []
    start = 0
    doc_len = len(text)
    chunk_id = 0

    while start < doc_len and chunk_id < max_chunks:
        end = min(start + chunk_chars, doc_len)
        chunk_text = text[start:end]

        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": chunk_text,
            "meta": {"start": start, "end": end},
        })

        chunk_id += 1
        if end >= doc_len:
            break  # done

        start = max(0, end - overlap)

    print(f"[Chunking] Created {len(chunks)} chunks (capped at {max_chunks}).")
    return chunks


# ---------------------------
# Multi-document ingestion
# ---------------------------
def extract_documents(file_paths: List[str]) -> List[Dict]:
    all_chunks = []
    for path in tqdm(file_paths, desc="Ingesting documents"):
        try:
            doc = extract_document(path)
            print(f"[DEBUG] Full text length: {len(doc['full_text'])} characters")
            chunks = chunk_text(doc["full_text"], chunk_chars=1500, overlap=200)
            for c in chunks:
                c["meta"].update({
                    "source_doc": doc["id"],
                    "source_name": os.path.basename(path),
                    "title": doc.get("title", "")
                })
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")
    return all_chunks

# ---------------------------
# CLI Testing
# ---------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/ingest.py file1.pdf file2.docx ...")
        sys.exit(1)

    files = sys.argv[1:]
    chunks = extract_documents(files)
    print(f"\n[Done] Ingested {len(files)} documents → {len(chunks)} chunks total.")
    for f in files:
        print(" -", f)

