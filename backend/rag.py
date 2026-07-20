"""
FAISS-based vector store: build, save, load, and retrieve.
Author: Clara-xiao (xiaoxua@kean.edu)
GitHub: https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

from backend.embeddings import embed

INDEX_DIR = Path("data/faiss_index")
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "meta.pkl"


class VectorStore:
    """Thin wrapper around a FAISS IndexFlatIP (inner-product = cosine after L2-norm)."""

    def __init__(self):
        self.index = None
        self.chunks: List[str] = []
        self.sources: List[str] = []  # file name per chunk

    # ------------------------------------------------------------------
    def build(self, chunks: List[str], sources: List[str]) -> None:
        if not _FAISS_OK:
            raise RuntimeError("faiss-cpu not installed. Run: pip install faiss-cpu")
        vecs = embed(chunks)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # inner product on unit vecs = cosine
        self.index.add(vecs)
        self.chunks = chunks
        self.sources = sources

    # ------------------------------------------------------------------
    def save(self, index_dir: Path = INDEX_DIR) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        with open(index_dir / "meta.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "sources": self.sources}, f)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "VectorStore":
        vs = cls()
        vs.index = faiss.read_index(str(index_dir / "index.faiss"))
        with open(index_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        vs.chunks = meta["chunks"]
        vs.sources = meta["sources"]
        return vs

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Return list of (chunk_text, source, score)."""
        if self.index is None or self.index.ntotal == 0:
            return []
        q_vec = embed([query])
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], self.sources[idx], float(score)))
        return results
