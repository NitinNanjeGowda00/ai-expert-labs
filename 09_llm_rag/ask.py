import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main():
    if len(sys.argv) < 2:
        print("Usage: python ask.py \"your question\"")
        sys.exit(1)

    question = sys.argv[1]
    print("\n=== QUESTION ===")
    print(question)

    # --------------------------------------------------
    # Resolve paths RELATIVE to this file (IMPORTANT)
    # --------------------------------------------------
    ROOT = Path(__file__).resolve().parents[1]
    META_PATH = ROOT / "artifacts" / "seg09" / "vector_store.json"

    if not META_PATH.exists():
        raise RuntimeError(
            f"Vector store metadata not found at {META_PATH}. "
            "Run ingest.py first."
        )

    meta = json.loads(META_PATH.read_text())
    vector_store_id = meta.get("vector_store_id")

    if not vector_store_id:
        raise RuntimeError("vector_store_id missing in metadata file")

    print(f"\nUsing vector store: {vector_store_id}")

    client = OpenAI()

    # --------------------------------------------------
    # RAG query using Responses API
    # --------------------------------------------------
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=question,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }
        ],
    )

    print("\n=== ANSWER ===")
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    print(content.text)

    # --------------------------------------------------
    # Show citations (retrieval evidence)
    # --------------------------------------------------
    print("\n=== CITATIONS ===")
    for item in response.output:
        if item.type == "file_search":
            for r in item.results:
                print(f"- {r.file_name} (score={r.score:.3f})")


if __name__ == "__main__":
    main()

