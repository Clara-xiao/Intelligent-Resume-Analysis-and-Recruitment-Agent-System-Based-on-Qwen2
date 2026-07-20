"""
PDF / text parser and chunker.
Author: Clara-xiao (xiaoxua@kean.edu)
GitHub: https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List

try:
    import pdfplumber
    _PDF_OK = True
except ImportError:
    _PDF_OK = False


def extract_text(path: str | Path) -> str:
    """Extract plain text from a PDF or txt file."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        if not _PDF_OK:
            raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks by sentence boundaries.
    chunk_size: target characters per chunk
    overlap:    characters of overlap between consecutive chunks
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += sent + " "
        else:
            if current:
                chunks.append(current.strip())
            # Start next chunk with overlap
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + sent + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks
