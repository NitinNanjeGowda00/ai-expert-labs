"""
Segment 09 - Eval Harness
UPGRADE 3: Run a batch of questions and save a report.

Run:
  python 09_llm_rag/eval.py

Reads:
  artifacts/seg09/vector_store.json
  artifacts/seg09/assistant.json (created by ask.py)

Writes:
  artifacts/seg09/eval_report.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ART_DIR = Path("artifacts/seg09")
VS_PATH = ART_DIR / "vector_store.json"
ASSIST_PATH = ART_DIR / "assistant.json"
OUT_EVAL = ART_DIR / "eval_report.json"


QUESTIONS = [
    "What does Grant Thornton do?",
    "List 3 things an agentic AI system can help with in this context.",
    "Does the document mention compliance guardrails or approvals?",
    "What tools can the agentic system use for structured data?",
    "If the documents don't contain an answer, what should you do?",
]


def wait_for_run(client: OpenAI, thread_id: str, run_id: str, poll_sec: float = 1.0):
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run.status in ("completed", "failed", "cancelled", "expired"):
            return run
        time.sleep(poll_sec)


def safe_json(obj) -> dict:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return {"repr": repr(obj)}


def extract_citations_from_steps(steps_obj: dict) -> list[dict]:
    citations: list[dict] = []
    for step in steps_obj.get("data", []):
        details = step.get("step_details") or {}
        if details.get("type") != "tool_calls":
            continue
        for call in details.get("tool_calls", []):
            if call.get("type") != "file_search":
                continue
            fs = call.get("file_search") or {}
            for r in fs.get("results", []) or []:
                citations.append(
                    {
                        "file_name": r.get("file_name"),
                        "score": r.get("score"),
                    }
                )
    return citations


def sum_token_usage(steps_obj: dict) -> dict:
    prompt = 0
    completion = 0
    total = 0
    for step in steps_obj.get("data", []):
        usage = step.get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Add it to .env")

    if not VS_PATH.exists():
        raise RuntimeError("vector_store.json not found. Run ingest.py first.")
    if not ASSIST_PATH.exists():
        raise RuntimeError("assistant.json not found. Run ask.py once to create/reuse assistant.")

    client = OpenAI()

    assistant_id = json.loads(ASSIST_PATH.read_text(encoding="utf-8"))["assistant_id"]

    rows = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for q in QUESTIONS:
        thread = client.beta.threads.create(messages=[{"role": "user", "content": q}])
        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
        run = wait_for_run(client, thread_id=thread.id, run_id=run.id)

        steps = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id)
        steps_obj = safe_json(steps)

        citations = extract_citations_from_steps(steps_obj)
        usage = sum_token_usage(steps_obj)

        totals["prompt_tokens"] += usage["prompt_tokens"]
        totals["completion_tokens"] += usage["completion_tokens"]
        totals["total_tokens"] += usage["total_tokens"]

        answer_text = ""
        if run.status == "completed":
            msgs = client.beta.threads.messages.list(thread_id=thread.id)
            for m in msgs.data:
                if m.role == "assistant":
                    parts = []
                    for c in m.content:
                        if c.type == "text":
                            parts.append(c.text.value)
                    answer_text = "\n".join(parts).strip()
                    break

        answerable = bool(citations) and bool(answer_text) and ("don't know" not in answer_text.lower())

        rows.append(
            {
                "question": q,
                "status": run.status,
                "answerable": answerable,
                "answer": answer_text[:400],
                "citations": citations,
                "token_usage": usage,
            }
        )

    answerable_rate = sum(1 for r in rows if r["answerable"]) / max(len(rows), 1)

    report = {
        "n_questions": len(rows),
        "answerable_rate": answerable_rate,
        "totals_token_usage": totals,
        "rows": rows,
    }

    OUT_EVAL.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Eval Summary ===")
    print(json.dumps({"n_questions": len(rows), "answerable_rate": answerable_rate, "totals": totals}, indent=2))
    print(f"\nSaved: {OUT_EVAL}")


if __name__ == "__main__":
    main()
