"""
FastAPI REST service for the Recruit Agent.
Author: Clara-xiao (xiaoxua@kean.edu)
GitHub: https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2

Run:
    uvicorn api:app --reload --port 8000
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.agent import RecruitAgent, AnalysisResult

app = FastAPI(
    title="Intelligent Resume Analysis API",
    description="RAG + Qwen2.5 powered resume screening",
    version="1.0.0",
)

_agent: RecruitAgent | None = None


def _get_agent() -> RecruitAgent:
    global _agent
    if _agent is None:
        try:
            _agent = RecruitAgent()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Index not ready: {exc}")
    return _agent


class AnalyseRequest(BaseModel):
    job_description: str
    resume_text: str
    top_k: int = 5


class AnalyseResponse(BaseModel):
    match_score: int
    strengths: list[str]
    gaps: list[str]
    recommendation: str
    interview_questions: list[str]
    error: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest):
    agent = _get_agent()
    result: AnalysisResult = agent.analyse(
        job_description=req.job_description,
        resume_text=req.resume_text,
        top_k=req.top_k,
    )
    return AnalyseResponse(
        match_score=result.match_score,
        strengths=result.strengths,
        gaps=result.gaps,
        recommendation=result.recommendation,
        interview_questions=result.interview_questions,
        error=result.error,
    )
