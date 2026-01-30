# Segment 10 — Agentic AI

This segment demonstrates a simple but production-aligned agent architecture:

Planner → Tool → Verifier

## What it shows
- Decision making before calling tools
- Tool orchestration (RAG reused as a tool)
- Guardrails and refusal logic
- Grounded answers with citations

## Run
```powershell
python 10_agents/agent.py "What does Grant Thornton do?"
python 10_agents/agent.py "What is the capital of France?"
