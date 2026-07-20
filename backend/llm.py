"""
Qwen2.5 LLM client (OpenAI-compatible interface).
Supports DashScope, local vLLM/Ollama, or demo mode (no key needed).
Author: Clara-xiao (xiaoxua@kean.edu)
GitHub: https://github.com/Clara-xiao/Intelligent-Resume-Analysis-and-Recruitment-Agent-System-Based-on-Qwen2
"""
from __future__ import annotations
import os

_API_KEY = os.getenv("LLM_API_KEY", "")
_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")

DEMO_RESPONSE = """{
  "match_score": 82,
  "strengths": [
    "3 years Python / FastAPI backend experience matches JD requirement",
    "Familiar with vector databases (Milvus), relevant to RAG stack",
    "Led a team of 4, demonstrates engineering leadership"
  ],
  "gaps": [
    "No direct LLM fine-tuning experience mentioned",
    "CI/CD pipeline experience not clearly stated"
  ],
  "recommendation": "Strong candidate — recommend technical interview. Focus on LLM deployment and system-design questions.",
  "interview_questions": [
    "Describe how you would optimise retrieval latency in a RAG pipeline under high QPS.",
    "How would you evaluate the output quality of an LLM-based screening system?",
    "Walk me through a time you debugged a production performance issue end-to-end."
  ]
}"""


def chat(system_prompt: str, user_prompt: str) -> str:
    """
    Send a chat request to Qwen2.5 and return the raw text response.
    Falls back to DEMO_RESPONSE if no API key is configured.
    """
    if not _API_KEY or _API_KEY.lower() in ("", "demo", "your_api_key_here"):
        return DEMO_RESPONSE

    try:
        from openai import OpenAI
        client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        # Degrade gracefully so the UI never crashes
        return f'{{"error": "{exc}", "match_score": 0}}'
