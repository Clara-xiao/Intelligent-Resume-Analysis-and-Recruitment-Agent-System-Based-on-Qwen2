"""
RecruitAgent: orchestrates RAG retrieval → LLM analysis → structured output.
Author: Clara-xiao (xiaoxua@kean.edu)
GitHub: https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.rag import VectorStore
from backend.llm import chat

SYSTEM_PROMPT = """You are an expert technical recruiter and HR analyst.
Given a job description and candidate resume excerpts retrieved from a vector database,
analyse the match and output ONLY valid JSON with these keys:
- match_score (int 0-100)
- strengths (list of strings)
- gaps (list of strings)
- recommendation (string)
- interview_questions (list of 3 strings)
Output JSON only, no markdown fences."""


@dataclass
class AnalysisResult:
    match_score: int = 0
    strengths: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    recommendation: str = ""
    interview_questions: List[str] = field(default_factory=list)
    raw: str = ""
    error: Optional[str] = None


class RecruitAgent:
    def __init__(self, index_dir: str = "data/faiss_index"):
        from pathlib import Path
        self.vs = VectorStore.load(Path(index_dir))

    # ------------------------------------------------------------------
    def analyse(self, job_description: str, resume_text: str, top_k: int = 5) -> AnalysisResult:
        # 1. Retrieve relevant chunks from the indexed resume corpus
        retrieved = self.vs.search(job_description, top_k=top_k)
        context_parts = [f"[{src}] {chunk}" for chunk, src, _ in retrieved]

        # 2. Build prompt
        user_prompt = (
            f"## Job Description\n{job_description}\n\n"
            f"## Candidate Resume (submitted)\n{resume_text[:3000]}\n\n"
            f"## Retrieved Context from Resume DB (top-{top_k})\n"
            + "\n---\n".join(context_parts)
        )

        # 3. Call LLM
        raw = chat(SYSTEM_PROMPT, user_prompt)

        # 4. Parse JSON robustly
        return self._parse(raw)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(raw: str) -> AnalysisResult:
        """Extract JSON even if the model wraps it in markdown fences."""
        text = raw.strip()
        # Strip ```json ... ``` fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # Try direct parse
        try:
            data: Dict[str, Any] = json.loads(text)
        except json.JSONDecodeError:
            # Try to find the first {...} block
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    return AnalysisResult(raw=raw, error="JSON parse failed")
            else:
                return AnalysisResult(raw=raw, error="No JSON found in response")

        return AnalysisResult(
            match_score=int(data.get("match_score", 0)),
            strengths=data.get("strengths", []),
            gaps=data.get("gaps", []),
            recommendation=data.get("recommendation", ""),
            interview_questions=data.get("interview_questions", []),
            raw=raw,
        )
