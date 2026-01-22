# 09 — LLM RAG (Retrieval-Augmented Q&A with Citations + Guardrail)

A minimal RAG (Retrieval-Augmented Generation) system that answers questions **only from your provided documents**, returns **citations**, and uses a **score-based guardrail** to refuse out-of-scope questions.

This segment demonstrates:
- Document-grounded Q&A (no free-form guessing)
- Citations from retrieved chunks
- A simple “answer / refuse” guardrail using retrieval score
- Saved debug artifacts for inspection

---

## Quickstart (60 seconds)

### 1) Install dependencies
From repo root:
```bash
pip install -r requirements.txt
