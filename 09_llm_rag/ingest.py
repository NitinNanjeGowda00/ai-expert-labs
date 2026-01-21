"""
Segment 09 - RAG Ingestion (OpenAI Vector Store)

- Reads files from 09_llm_rag/docs/
- Uploads supported non-empty files
- Creates a Vector Store and attaches the uploaded files
- Writes vector store info to artifacts/seg09/vector_store.json

Run:
  python 09_llm_rag/ingest.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

DOCS_DIR = Path("09_llm_rag/docs")
ART_DIR = Path("artifacts/seg09")
ART_DIR.mkdir(parents=True, exist_ok=True)

# Keep it simple: ingest only these types (add more if you want)
ALLOWED_EXTS = {".txt", ".md", ".pdf", ".csv", ".json"}


def is_allowed(path: Path) -> bool:
    # Skip hidden/dot files and placeholders
    if path.name.startswith("."):
        return False
    if path.suffix.lower() not in ALLOWED_EXTS:
        return False
    # Skip empty files (OpenAI rejects application/x-empty)
    if path.stat().st_size == 0:
        return False
    return True


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Add it to .env")

    client = OpenAI()

    if not DOCS_DIR.exists():
        raise RuntimeError(f"Docs folder not found: {DOCS_DIR}. Create it and add some .txt/.md/.pdf files.")

    all_files = [p for p in DOCS_DIR.rglob("*") if p.is_file()]
    files = [p for p in all_files if is_allowed(p)]

    if not files:
        raise RuntimeError(
            f"No supported non-empty files found in {DOCS_DIR}.\n"
            f"Allowed extensions: {sorted(ALLOWED_EXTS)}\n"
            f"Tip: add a .txt or .md file with content."
        )

    print(f"Found {len(files)} supported files in {DOCS_DIR} (from {len(all_files)} total files)")

    # 1) Upload files
    uploaded_ids = []
    for path in files:
        with open(path, "rb") as f:
            up = client.files.create(file=f, purpose="assistants")
        uploaded_ids.append(up.id)
        print(f"Uploaded: {path.name} -> {up.id}")

    # 2) Create vector store
    vs = client.vector_stores.create(name="seg09_vector_store")
    print(f"Created vector store: {vs.id}")

    # 3) Add files to vector store
    batch = client.vector_stores.file_batches.create(
        vector_store_id=vs.id,
        file_ids=uploaded_ids,
    )
    batch_id = batch.id  # keep stable; must begin with vsfb_
    print(f"File batch: {batch_id} (status={batch.status})")

    # 4) Poll until complete
    while True:
        batch = client.vector_stores.file_batches.retrieve(
            vector_store_id=vs.id,
            batch_id=batch_id,
        )
        if batch.status in ("completed", "failed", "cancelled"):
            print(f"Batch finished: status={batch.status}")
            break
        print(f"Waiting... status={batch.status}")
        time.sleep(2)

    if batch.status != "completed":
        raise RuntimeError(f"Vector store ingestion failed: status={batch.status}")

    out = {
        "vector_store_id": vs.id,
        "file_batch_id": batch_id,
        "file_ids": uploaded_ids,
        "docs_dir": str(DOCS_DIR),
        "allowed_exts": sorted(ALLOWED_EXTS),
    }

    (ART_DIR / "vector_store.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nSaved: {ART_DIR / 'vector_store.json'}")
    print('Next: python 09_llm_rag/ask.py "your question"')


if __name__ == "__main__":
    main()
