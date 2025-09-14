# vectorstore.py
"""
A tiny in-memory vector store using numpy for cosine similarity.
This is intentionally small & dependency-light so you can run locally.
Swap for FAISS/Pinecone/Chroma in production.
"""
import numpy as np
from typing import List, Dict

class InMemoryVectorStore:
    def __init__(self):
        self.vectors = []    # list of numpy arrays
        self.metadatas = []  # parallel list of metadata dicts
        self.ids = []        # stable ids

    def add(self, vec: List[float], metadata: Dict, id: str = None):
        vec = np.array(vec, dtype=np.float32)
        # normalize
        norm = np.linalg.norm(vec) + 1e-10
        vec = vec / norm
        self.vectors.append(vec)
        self.metadatas.append(metadata)
        self.ids.append(id or str(len(self.ids)))

    def similarity_search(self, query_vec: List[float], top_k: int = 5):
        """
        Returns top_k hits with score and metadata
        """
        if len(self.vectors) == 0:
            return []
        q = np.array(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-10)
        mat = np.stack(self.vectors, axis=0)  # (n, dim)
        scores = mat.dot(q)
        idx = np.argsort(-scores)[:top_k]
        results = []
        for i in idx:
            results.append({"id": self.ids[i], "score": float(scores[i]), "metadata": self.metadatas[i]})
        return results

    def __len__(self):
        return len(self.vectors)
