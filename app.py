"""
Streamlit frontend for the Intelligent Resume Analysis System.
Author: Clara-xiao (xiaoxua@kean.edu)
GitHub: https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2

Run:
    streamlit run app.py
"""
import streamlit as st
from pathlib import Path
from backend.agent import RecruitAgent
from backend.parser import extract_text

st.set_page_config(
    page_title="Intelligent Resume Analysis",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Intelligent Resume Analysis & Recruitment Agent")
st.caption(
    "RAG + Qwen2.5 · Built by [Clara-xiao](https://github.com/Clara-xiao) · "
    "[GitHub Repo](https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2)"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("RAG top-K retrieved chunks", 1, 10, 5)
    st.info(
        "No API key? The system runs in **Demo Mode** and returns a sample analysis. "
        "Set `LLM_API_KEY` in `.env` to use real Qwen2.5."
    )

# ── Main layout ───────────────────────────────────────────────────────────────
col_jd, col_resume = st.columns(2)

with col_jd:
    st.subheader("📋 Job Description")
    jd_text = st.text_area(
        "Paste or type the JD here",
        height=300,
        placeholder="We are looking for a senior backend engineer with Python, FastAPI, and LLM experience...",
    )
    jd_file = st.file_uploader("Or upload a .txt / .pdf JD", type=["txt", "pdf"], key="jd")
    if jd_file:
        tmp = Path(f"/tmp/{jd_file.name}")
        tmp.write_bytes(jd_file.read())
        jd_text = extract_text(tmp)
        st.success(f"Loaded: {jd_file.name}")

with col_resume:
    st.subheader("📄 Candidate Resume")
    resume_text = st.text_area(
        "Paste or type the resume here",
        height=300,
        placeholder="Jane Doe — Software Engineer\n3 years Python, FastAPI, Redis...",
    )
    resume_file = st.file_uploader("Or upload a .txt / .pdf resume", type=["txt", "pdf"], key="cv")
    if resume_file:
        tmp = Path(f"/tmp/{resume_file.name}")
        tmp.write_bytes(resume_file.read())
        resume_text = extract_text(tmp)
        st.success(f"Loaded: {resume_file.name}")

# ── Analyse button ────────────────────────────────────────────────────────────
st.divider()
if st.button("🚀 Analyse", type="primary", use_container_width=True):
    if not jd_text.strip() or not resume_text.strip():
        st.warning("Please provide both a job description and a resume.")
    else:
        with st.spinner("Running RAG retrieval + Qwen2.5 analysis…"):
            try:
                agent = RecruitAgent()
                result = agent.analyse(jd_text, resume_text, top_k=top_k)
            except Exception as exc:
                st.error(f"Index not ready — run `python scripts/build_index.py` first.\n\n{exc}")
                st.stop()

        if result.error:
            st.error(f"Analysis error: {result.error}")
        else:
            # ── Score gauge ───────────────────────────────────────────────
            score = result.match_score
            color = "#2ecc71" if score >= 75 else "#f39c12" if score >= 50 else "#e74c3c"
            st.markdown(
                f"""
                <div style='text-align:center;padding:1rem;'>
                  <span style='font-size:3rem;font-weight:bold;color:{color};'>{score}/100</span>
                  <br><span style='color:gray;'>Match Score</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(score / 100)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("✅ Strengths")
                for s in result.strengths:
                    st.markdown(f"- {s}")
            with c2:
                st.subheader("⚠️ Gaps")
                for g in result.gaps:
                    st.markdown(f"- {g}")

            st.subheader("📝 Recommendation")
            st.info(result.recommendation)

            st.subheader("❓ Suggested Interview Questions")
            for i, q in enumerate(result.interview_questions, 1):
                st.markdown(f"**Q{i}.** {q}")
