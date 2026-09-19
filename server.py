"""Jev-compatible local server: POST /v1/systemone over Ollama / LM Studio / Atomic.

Run:
    uv run uvicorn server:app --host 127.0.0.1 --port 8010

Point it at any OpenAI-compatible backend with env vars:
    JEV_BASE_URL=http://127.0.0.1:11434/v1   # Ollama
    JEV_BASE_URL=http://127.0.0.1:1234/v1    # LM Studio
    JEV_BASE_URL=http://127.0.0.1:1337/v1    # Atomic Chat
    JEV_MODEL=qwen2.5:7b
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from jev_local import JevError, JevLocal

app = FastAPI(title="jev-local", version="0.1.0")


def make_jev(model: str | None = None) -> JevLocal:
    return JevLocal(
        base_url=os.environ.get("JEV_BASE_URL", "http://127.0.0.1:11434/v1"),
        model=model or os.environ.get("JEV_MODEL", "qwen2.5-coder:7b"),
        api_key=os.environ.get("JEV_API_KEY", "local"),
    )


class SystemOneRequest(BaseModel):
    model: str | None = None
    state: str
    questions: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "base_url": app.state.base_url, "model": app.state.model}


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest) -> dict:
    jev = make_jev(req.model)
    try:
        result = jev.decide(req.state, req.questions)
    except JevError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    result["usage"] = {"input_tokens": 0, "output_tokens": 0}
    return result


@app.on_event("startup")
def _startup() -> None:
    jev = make_jev()
    app.state.base_url = jev.base_url
    app.state.model = jev.model
