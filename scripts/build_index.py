"""
One-time script: parse all resumes in data/sample_resumes/ and build the FAISS index.
Author: Clara-xiao (xiaoxua@kean.edu)
GitHub: https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2

Run from project root:
    python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.parser import extract_text, chunk_text
from backend.rag import VectorStore

RESUME_DIR = Path("data/sample_resumes")
INDEX_DIR = Path("data/faiss_index")


def main():
    files = list(RESUME_DIR.glob("*.txt")) + list(RESUME_DIR.glob("*.pdf"))
    if not files:
        print(f"No files found in {RESUME_DIR}. Add .txt or .pdf resumes and rerun.")
        return

    all_chunks, all_sources = [], []
    for fp in files:
        print(f"  Parsing {fp.name} …")
        text = extract_text(fp)
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        all_chunks.extend(chunks)
        all_sources.extend([fp.name] * len(chunks))
        print(f"    → {len(chunks)} chunks")

    print(f"\nBuilding FAISS index over {len(all_chunks)} chunks …")
    vs = VectorStore()
    vs.build(all_chunks, all_sources)
    vs.save(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
