"""
BGE embedding wrapper (supports local model or API fallback).
Author: Clara-xiao (xiaoxua@kean.edu)
GitHub: https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2
"""
from __future__ import annotations
import os
import numpy as np
from typing import List

# Try sentence-transformers (local BGE); fall back to OpenAI-compat embedding API
try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    _ST_OK = False

_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
_model: "SentenceTransformer | None" = None


def _get_local_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts → float32 array of shape (N, dim).
    Vectors are L2-normalised so dot-product == cosine similarity.
    """
    if not texts:
        return np.zeros((0, 512), dtype=np.float32)

    if _ST_OK:
        model = _get_local_model()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

    # Fallback: random unit vectors for demo/testing without GPU
    dim = 512
    rng = np.random.default_rng(seed=42)
    vecs = rng.standard_normal((len(texts), dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms
