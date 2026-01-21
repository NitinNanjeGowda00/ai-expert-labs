"""
Segment 09 - Advanced RAG QA (Assistants + file_search)
Upgrades:
1) Reuse assistant_id (persist to artifacts/seg09/assistant.json)
2) Save structured citations + token usage (last_answer.json)
3) Debug artifact for failures (last_run_debug.json)
4) Guardrails: enforce "don't know" if no citations or low confidence retrieval

Run:
  python 09_llm_rag/ask.py "Your question here"
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ART_DIR = Path("artifacts/seg09")
VS_PATH = ART_DIR / "vector_store.json"
ASSIST_PATH = ART_DIR / "assistant.json"
OUT_ANSWER = ART_DIR / "last_answer.json"
OUT_DEBUG = ART_DIR / "last_run_debug.json"

# --- UPGRADE 4: retrieval confidence threshold ---
# If best retrieval score is below this, we treat the answer as unknown.
# Tune between ~0.2 to 0.6 based on your docs quality.
SCORE_THRESHOLD = 0.35


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


def load_assistant_id() -> str | None:
    if not ASSIST_PATH.exists():
        return None
    data = json.loads(ASSIST_PATH.read_text(encoding="utf-8"))
    return data.get("assistant_id")


def save_assistant_id(assistant_id: str) -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    ASSIST_PATH.write_text(json.dumps({"assistant_id": assistant_id}, indent=2), encoding="utf-8")


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
                        "file_id": r.get("file_id"),
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
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python 09_llm_rag/ask.py "Your question here"')

    question = sys.argv[1]

    if not VS_PATH.exists():
        raise RuntimeError("vector_store.json not found. Run: python 09_llm_rag/ingest.py")

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Add it to .env")

    client = OpenAI()
    vs = json.loads(VS_PATH.read_text(encoding="utf-8"))
    vector_store_id = vs["vector_store_id"]

    # Upgrade 1: reuse assistant
    assistant_id = load_assistant_id()
    if assistant_id:
        assistant = client.beta.assistants.retrieve(assistant_id)
        print(f"Reusing assistant: {assistant.id}")
    else:
        assistant = client.beta.assistants.create(
            name="Segment 09 RAG Assistant",
            model="gpt-4o-mini",
            instructions=(
                "Answer ONLY using the provided documents. "
                "If the documents do not contain the answer, say you don't know. "
                "Always cite which document the info comes from."
            ),
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        )
        save_assistant_id(assistant.id)
        print(f"Created assistant: {assistant.id}")

    # Thread + run
    thread = client.beta.threads.create(messages=[{"role": "user", "content": question}])
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
    run = wait_for_run(client, thread_id=thread.id, run_id=run.id)

    # Upgrade 2: steps -> citations + token usage
    steps = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id)
    steps_obj = safe_json(steps)
    citations = extract_citations_from_steps(steps_obj)
    token_usage = sum_token_usage(steps_obj)

    best_score = max((c.get("score") or 0.0) for c in citations) if citations else 0.0

    debug = {
        "question": question,
        "assistant_id": assistant.id,
        "thread_id": thread.id,
        "run_id": run.id,
        "run_status": run.status,
        "run_last_error": safe_json(run.last_error) if getattr(run, "last_error", None) else None,
        "vector_store_id": vector_store_id,
        "citations": citations,
        "token_usage": token_usage,
        "best_score": best_score,
        "score_threshold": SCORE_THRESHOLD,
    }
    OUT_DEBUG.write_text(json.dumps(debug, indent=2), encoding="utf-8")

    if run.status != "completed":
        print("\n=== RUN FAILED ===")
        if debug["run_last_error"]:
            print(json.dumps(debug["run_last_error"], indent=2))
        else:
            print("No last_error available.")
        print(f"\nSaved debug: {OUT_DEBUG}")
        raise RuntimeError(f"Run did not complete: status={run.status}")

    # Messages (answer)
    msgs = client.beta.threads.messages.list(thread_id=thread.id)
    answer_text = ""
    for m in msgs.data:
        if m.role == "assistant":
            parts = []
            for c in m.content:
                if c.type == "text":
                    parts.append(c.text.value)
            answer_text = "\n".join(parts).strip()
            break

    # Upgrade 4: guardrail enforcement
    if len(citations) == 0 or best_score < SCORE_THRESHOLD:
        answer_text = "I don't know based on the provided documents."

    out = {
        "question": question,
        "answer": answer_text,
        "assistant_id": assistant.id,
        "thread_id": thread.id,
        "run_id": run.id,
        "vector_store_id": vector_store_id,
        "citations": citations,
        "token_usage": token_usage,
        "best_score": best_score,
        "score_threshold": SCORE_THRESHOLD,
    }
    OUT_ANSWER.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Answer ===\n")
    print(answer_text if answer_text else "(No assistant text parsed)")
    print("\n=== Citations (retrieval results) ===")
    print(json.dumps(citations, indent=2))
    print("\n=== Guardrail ===")
    print(json.dumps({"best_score": best_score, "threshold": SCORE_THRESHOLD}, indent=2))
    print("\n=== Token usage (summed across steps) ===")
    print(json.dumps(token_usage, indent=2))
    print(f"\nSaved: {OUT_ANSWER}")
    print(f"Saved debug: {OUT_DEBUG}")


if __name__ == "__main__":
    main()
