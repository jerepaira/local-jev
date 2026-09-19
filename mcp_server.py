"""Local Jev MCP server (stdio).

Turns any OpenAI-compatible LLM you already run (Ollama, LM Studio, ...) into a
typed decision function and exposes it over MCP:

    decide    typed decisions (choice / score / noul)
    classify  single best label + calibrated confidence + per-label scores
    score     rating on an ordered scale
    check     yes/no question with P(yes)

Configure the backend with environment variables:
    JEV_BASE_URL / JEV_MODEL                 default LM Studio qwen2.5-coder-14b-instruct
    JEV_FALLBACK_URL / JEV_FALLBACK_MODEL    default Ollama qwen2.5-coder:7b
"""

from __future__ import annotations

import json
import os

from mcp.server.mcpserver import MCPServer

from jev_local import JevError, JevLocal

mcp = MCPServer("jev-local")

PRIMARY_URL = os.environ.get("JEV_BASE_URL", "http://127.0.0.1:1234/v1")
PRIMARY_MODEL = os.environ.get("JEV_MODEL", "qwen2.5-coder-14b-instruct")
FALLBACK_URL = os.environ.get("JEV_FALLBACK_URL", "http://127.0.0.1:11434/v1")
FALLBACK_MODEL = os.environ.get("JEV_FALLBACK_MODEL", "qwen2.5-coder:7b")


def _clients() -> list[JevLocal]:
    return [
        JevLocal(PRIMARY_URL, PRIMARY_MODEL),
        JevLocal(FALLBACK_URL, FALLBACK_MODEL),
    ]


def _decide(state: str, questions: dict) -> dict:
    last: Exception | None = None
    for jev in _clients():
        try:
            return jev.decide(state, questions)
        except JevError as exc:
            last = exc
    raise RuntimeError(f"no backend available: {last}")


@mcp.tool()
def decide(state: str, questions: dict) -> str:
    """Make typed decisions about a state. questions maps an id to
    {type: "choice"|"score"|"noul", instructions, criteria}. Returns each answer
    with a probability distribution."""
    return json.dumps(_decide(state, questions)["answers"], ensure_ascii=False)


@mcp.tool()
def classify(text: str, labels: list[str], instructions: str = "") -> str:
    """Pick the single best label for a text, with calibrated confidence and a
    score for every label. Use for routing, tagging, moderation, triage."""
    q = {"label": {"type": "choice",
                   "instructions": instructions or "Pick the single best label for this text.",
                   "criteria": {label: label for label in labels}}}
    ans = _decide(text, q)["answers"]["label"]
    return json.dumps({"label": ans["choice"], "confidence": ans["confidence"],
                       "scores": ans["probabilities"]}, ensure_ascii=False)


@mcp.tool()
def score(text: str, criteria: list[str], instructions: str = "") -> str:
    """Rate a text on an ordered scale given by criteria (low to high). Returns
    the probability-weighted score plus the distribution."""
    q = {"score": {"type": "score",
                   "instructions": instructions or "Rate this text on the given scale.",
                   "criteria": criteria}}
    return json.dumps(_decide(text, q)["answers"]["score"], ensure_ascii=False)


@mcp.tool()
def check(state: str, question: str) -> str:
    """Answer a yes/no question about a state. Returns P(yes) in [0,1]."""
    q = {"check": {"type": "noul", "instructions": question}}
    ans = _decide(state, q)["answers"]["check"]
    return json.dumps({"yes": ans["noul"], "no": round(1 - ans["noul"], 6)})


if __name__ == "__main__":
    mcp.run()
