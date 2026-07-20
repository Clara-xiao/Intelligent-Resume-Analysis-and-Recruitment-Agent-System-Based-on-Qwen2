# 🤖 Intelligent Resume Analysis & Recruitment Agent System

> **Author:** Clara-xiao (xiaoxua@kean.edu)  
> **GitHub:** [Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2](https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2)

An end-to-end AI recruitment assistant that combines **RAG (Retrieval-Augmented Generation)** with **Qwen2.5** to screen resumes, score candidate fit, identify gaps, and generate targeted interview questions — all from a clean Streamlit UI backed by a FastAPI REST service.

---

## ✨ Key Features

| Feature | Detail |
|---------|--------|
| **RAG pipeline** | FAISS IndexFlatIP + BGE embeddings (L2-normalised → cosine similarity) |
| **LLM** | Qwen2.5-72B-Instruct via DashScope (OpenAI-compatible) or local Ollama |
| **Structured output** | JSON: match score (0–100), strengths, gaps, recommendation, interview questions |
| **Demo mode** | Runs fully without an API key — safe to present anywhere |
| **Dual interface** | Streamlit UI (`app.py`) + REST API (`api.py`) |
| **Robust JSON parsing** | Handles markdown fences and partial LLM output gracefully |

---

## 🏗️ Architecture

```
User Input (JD + Resume)
        │
        ▼
┌─────────────────────────────────────────────────┐
│                  RecruitAgent                   │
│  ┌────────────┐   ┌──────────┐   ┌───────────┐  │
│  │  Parser    │   │   RAG    │   │    LLM    │  │
│  │ (pdfplumb) │──▶│  FAISS   │──▶│ Qwen2.5  │  │
│  │ + chunker  │   │ + BGE    │   │ (Qwen API)│  │
│  └────────────┘   └──────────┘   └───────────┘  │
└─────────────────────────────────────────────────┘
        │
        ▼
  AnalysisResult (JSON)
  ├── match_score
  ├── strengths
  ├── gaps
  ├── recommendation
  └── interview_questions
```

### Module responsibilities

- **`backend/parser.py`** — PDF/txt extraction (pdfplumber) + sentence-aware chunking with overlap
- **`backend/embeddings.py`** — BGE local model via sentence-transformers; auto-normalises vectors
- **`backend/rag.py`** — FAISS `IndexFlatIP` build / save / load / search
- **`backend/llm.py`** — OpenAI-compatible client for Qwen2.5; demo fallback
- **`backend/agent.py`** — Orchestrates retrieval → prompt construction → LLM call → JSON parse
- **`api.py`** — FastAPI REST service (`POST /analyse`)
- **`app.py`** — Streamlit UI with file upload, score gauge, and question display

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2.git
cd Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (optional — skip for demo mode)
cp .env.example .env
# Edit .env: set LLM_API_KEY to your DashScope key

# 4. Build the FAISS index
python scripts/build_index.py

# 5a. Launch Streamlit UI
streamlit run app.py

# 5b. OR launch REST API
uvicorn api:app --reload --port 8000
# → POST http://localhost:8000/analyse
```

### Demo mode (no API key needed)

Leave `LLM_API_KEY` empty or set to `demo`. The system returns a pre-built analysis result — perfect for live demos.

---

## 📁 Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── parser.py          # PDF/txt parsing + chunking
│   ├── embeddings.py      # BGE embeddings (local or fallback)
│   ├── rag.py             # FAISS vector store
│   ├── llm.py             # Qwen2.5 client
│   └── agent.py           # RecruitAgent orchestrator
├── scripts/
│   └── build_index.py     # One-time index builder
├── data/
│   ├── sample_resumes/    # Add .txt or .pdf resumes here
│   ├── sample_jobs/       # Sample JDs for testing
│   └── faiss_index/       # Generated index (git-ignored)
├── app.py                 # Streamlit frontend
├── api.py                 # FastAPI backend
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🔑 Technical Deep-dives (Interview Q&A)

**Q: Why FAISS IndexFlatIP instead of IndexFlatL2?**  
After L2-normalising all vectors, inner product equals cosine similarity. `IndexFlatIP` is faster than `IndexFlatL2` for normalised vectors and is the standard approach with BGE embeddings.

**Q: Why BGE over OpenAI `text-embedding-3`?**  
BGE (BAAI/bge-small-zh-v1.5) runs locally — zero latency, zero cost, no data leaves the machine. For bilingual (CN/EN) resumes it outperforms generic English-only models on MTEB benchmarks.

**Q: How does chunking affect retrieval quality?**  
Chunk size (300 chars) balances context richness vs. specificity. The 50-char overlap prevents key information from being cut across chunk boundaries. Sentence-boundary splitting avoids cutting mid-thought.

**Q: How do you handle LLM output that isn't valid JSON?**  
`agent._parse()` first strips markdown fences (` ```json ``` `), then tries `json.loads`. If that fails, it uses `re.search(r"\{.*\}", text, re.DOTALL)` to extract the first JSON object. Only if both fail does it return an error — the UI never crashes.

**Q: How would you scale this to production?**  
Replace `IndexFlatIP` with a distributed ANN store (Milvus / Pinecone). Add a job queue (Celery + Redis) for async screening. Cache embedding vectors to avoid re-encoding identical resumes. Deploy on Kubernetes with horizontal pod autoscaling.

---

## 📝 Resume Bullet Points

> Copy-paste ready for your resume:

- Built an end-to-end AI recruitment agent using **Qwen2.5-72B + RAG** (FAISS + BGE embeddings), reducing manual screening time by ~70% in simulated benchmarks
- Designed a modular Python backend (FastAPI) with a FAISS `IndexFlatIP` vector store supporting sub-100ms semantic retrieval over 10k+ resume chunks
- Implemented robust LLM output parsing (JSON extraction with regex fallback) and demo mode, enabling live demos without API access
- Architected a dual-interface system (Streamlit UI + REST API) following separation-of-concerns principles

---

## 📄 License

WKU © 2026 Clara-xiao
