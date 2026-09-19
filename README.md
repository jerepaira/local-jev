# local-jev

A local, OpenAI-compatible **decision layer** that turns any LLM you already run
(Ollama, LM Studio, ...) into a fast, typed decision function — and exposes it
over **MCP**.

Inspired by TypeSafe AI's *Jev* ("System One" models). It never generates text:
for each question it asks the model for one token and reads the probability mass
on every option letter via `logprobs`, in a single pass. That gives a typed answer
(choice / score / yes-no) **with a probability distribution**.

```
state + typed question  ->  choice | score | P(yes)   (no generated text)
```

## Why

LLMs are great at fuzzy judgment but wasteful as an `if` in your code. This makes
that judgment cheap, structured and local: classify, route, score, verify, and
gate — with a confidence you can threshold.

## MCP tools

| tool | what it does |
| --- | --- |
| `decide` | several typed decisions about one state (choice / score / noul) |
| `classify` | single best label + confidence + score per label |
| `score` | rating on an ordered scale (low → high) |
| `check` | yes/no question with `P(yes)` |

## Use it (MCP)

Add to your client (opencode, Cline, Claude, Codex, ...). Example for opencode:

```json
{
  "mcp": {
    "jev-local": {
      "type": "local",
      "command": ["/path/to/.venv/bin/python", "/path/to/mcp_server.py"],
      "enabled": true
    }
  }
}
```

Then ask in natural language, e.g.
*"classify these tickets into bug / feature / praise"* or
*"is this shell command risky? use check"*.

## Use it (HTTP)

```bash
pip install fastapi uvicorn
uvicorn server:app --port 8010
curl localhost:8010/v1/systemone -H 'content-type: application/json' -d '{
  "state": "the checkout button does nothing",
  "questions": {"kind": {"type": "choice", "instructions": "Which kind?",
    "criteria": {"bug": "a defect", "feature": "a request", "praise": "positive"}}}
}'
```

## Backend

Any OpenAI-compatible endpoint that returns `logprobs` works. Tested with
**Ollama** and **LM Studio**. Configure with env vars:

```
JEV_BASE_URL=http://127.0.0.1:1234/v1   JEV_MODEL=qwen2.5-coder-14b-instruct
JEV_FALLBACK_URL=http://127.0.0.1:11434/v1  JEV_FALLBACK_MODEL=qwen2.5-coder:7b
```

> Note: reasoning models that emit a thinking token first (e.g. Qwen3.5 via
> Ollama's OpenAI endpoint) can't be used with the letter-logprob trick — use a
> non-thinking instruct model.

## Confidence is a ranking signal

The returned distributions are normalized candidate scores, not perfectly
calibrated probabilities. Use them to rank and to set thresholds; don't read the
third decimal as truth.

## Requirements

Python 3.10+, `mcp` (MCP 2.x). HTTP server additionally needs `fastapi` + `uvicorn`.

## Files

- `jev_local.py` — the core (stdlib only)
- `mcp_server.py` — MCP stdio server
- `server.py` — Jev-compatible HTTP `POST /v1/systemone`
