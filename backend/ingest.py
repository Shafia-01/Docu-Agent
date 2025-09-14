# ingest.py
"""
Multi-document, multi-format ingestion pipeline (PDF, DOCX, HTML).
Now includes debug logs to see where ingestion might hang.
Enhanced with better document naming logic.
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
# Document Name Utilities
# ---------------------------
def get_document_name(path: str, doc_content: Dict) -> str:
    """
    Extract a meaningful document name from various sources.
    Priority: PDF title > first line of content > filename
    """
    # Start with filename without extension as fallback
    base_name = os.path.splitext(os.path.basename(path))[0]
    
    # For PDFs, try to extract title from metadata or first meaningful line
    if path.lower().endswith('.pdf'):
        try:
            # Try PDF metadata first
            pdf_doc = fitz.open(path)
            metadata = pdf_doc.metadata
            if metadata and metadata.get('title') and metadata['title'].strip():
                pdf_title = metadata['title'].strip()
                pdf_doc.close()
                print(f"[DEBUG] Using PDF metadata title: {pdf_title}")
                return pdf_title
            pdf_doc.close()
        except Exception as e:
            print(f"[DEBUG] Could not extract PDF metadata: {e}")
    
    # Try to get title from first meaningful line of content
    if doc_content.get('full_text'):
        lines = doc_content['full_text'].strip().split('\n')
        for line in lines[:5]:  # Check first 5 lines
            clean_line = line.strip()
            if clean_line and len(clean_line) > 3 and len(clean_line) < 100:
                # Skip lines that look like headers, page numbers, etc.
                if not clean_line.lower().startswith(('page ', 'chapter ', 'section ')):
                    print(f"[DEBUG] Using content-based title: {clean_line}")
                    return clean_line
    
    print(f"[DEBUG] Using filename-based title: {base_name}")
    return base_name

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
    doc_id = hashlib.sha1(path.encode()).hexdigest()[:10]
    
    # Create document dict first, then get the smart title
    doc_dict = {
        "id": doc_id,
        "path": path,
        "title": "",  # Will be set below
        "pages": pages,
        "full_text": full_text_str,
        "tables": all_tables,
    }
    
    # Get smart document name
    doc_dict["title"] = get_document_name(path, doc_dict)

    return doc_dict

# ---------------------------
# DOCX Extraction
# ---------------------------
def extract_docx(path: str) -> Dict:
    print(f"[DOCX] Opening {path}...")
    doc = Document(path)
    
    # Try to get title from document properties first
    title = None
    if hasattr(doc.core_properties, 'title') and doc.core_properties.title:
        title = doc.core_properties.title.strip()
        print(f"[DEBUG] Found DOCX title property: {title}")
    
    text = [para.text for para in doc.paragraphs if para.text.strip()]
    full_text_str = "\n".join(text)
    doc_id = hashlib.sha1(path.encode()).hexdigest()[:10]
    
    doc_dict = {
        "id": doc_id,
        "path": path,
        "title": title if title else "",  # Will be set below if empty
        "pages": [],
        "full_text": full_text_str,
        "tables": [],
    }
    
    # If no title from properties, get smart document name
    if not doc_dict["title"]:
        doc_dict["title"] = get_document_name(path, doc_dict)
    
    return doc_dict

# ---------------------------
# HTML Extraction
# ---------------------------
def extract_html(path: str) -> Dict:
    print(f"[HTML] Opening {path}...")
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        
        # Try to get title from HTML title tag
        title = None
        title_tag = soup.find('title')
        if title_tag and title_tag.text.strip():
            title = title_tag.text.strip()
            print(f"[DEBUG] Found HTML title tag: {title}")
        
        text = soup.get_text(" ", strip=True)
    
    doc_id = hashlib.sha1(path.encode()).hexdigest()[:10]
    
    doc_dict = {
        "id": doc_id,
        "path": path,
        "title": title if title else "",  # Will be set below if empty
        "pages": [],
        "full_text": text,
        "tables": [],
    }
    
    # If no title from HTML, get smart document name
    if not doc_dict["title"]:
        doc_dict["title"] = get_document_name(path, doc_dict)
    
    return doc_dict

# ---------------------------
# Dispatcher
# ---------------------------
def extract_document(path: str) -> Dict:
    """Extract document with enhanced error handling and naming."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
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
            print(f"[DEBUG] Document title: '{doc['title']}'")
            print(f"[DEBUG] Full text length: {len(doc['full_text'])} characters")
            
            chunks = chunk_text(doc["full_text"], chunk_chars=1500, overlap=200)
            
            for c in chunks:
                c["meta"].update({
                    "source_doc": doc["id"],
                    "source_name": os.path.basename(path),  # Keep original filename
                    "document_name": doc["title"],  # Add the smart document name
                    "title": doc.get("title", ""),
                    "source_path": path  # Add full path for reference
                })
            
            all_chunks.extend(chunks)
            print(f"[DEBUG] Added {len(chunks)} chunks from '{doc['title']}'")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_chunks

# ---------------------------
# CLI Testing
# ---------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python backend/ingest.py file1.pdf file2.docx ...")
        sys.exit(1)

    files = sys.argv[1:]
    chunks = extract_documents(files)
    print(f"\n[Done] Ingested {len(files)} documents → {len(chunks)} chunks total.")
    
    # Show document names for verification
    doc_names = set()
    for chunk in chunks:
        doc_names.add(chunk["meta"]["document_name"])
    
    print("\nDocument names extracted:")
    for name in sorted(doc_names):
        print(f" - '{name}'")
    
    print("\nFiles processed:")
    for f in files:
        print(f" - {f}")