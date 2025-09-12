# src/arxiv_integration.py
"""
Simple arXiv API search. Returns a short list of entries.
Used as an optional function call from the agent.
"""
import requests
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"

def search_arxiv(query: str, max_results: int = 3):
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results
    }
    headers = {"User-Agent": "StochasticInc-Assessment/0.1 (Contact: you@example.com)"}
    r = requests.get(ARXIV_API, params=params, headers=headers, timeout=10)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}"}
    root = ET.fromstring(r.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()
        id_ = entry.find("atom:id", ns).text.strip()
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        pdf_link = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib.get("href")
        entries.append({"id": id_, "title": title, "summary": summary, "authors": authors, "pdf": pdf_link})
    return entries
