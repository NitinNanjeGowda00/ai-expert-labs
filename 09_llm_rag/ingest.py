import time
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ------------------------
# Config
# ------------------------
DOCS_DIR = Path("09_llm_rag/docs")
ARTIFACT_DIR = Path("artifacts/seg09")
ALLOWED_EXTS = {".md", ".txt", ".pdf", ".csv", ".json"}

load_dotenv()


def main():
    client = OpenAI()

    files = [
        p for p in DOCS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    ]

    print(f"Found {len(files)} supported files in {DOCS_DIR} (from {len(list(DOCS_DIR.iterdir()))} total files)")

    if not files:
        raise RuntimeError("No supported documents found")

    # ------------------------
    # Upload files
    # ------------------------
    uploaded_file_ids = []

    for path in files:
        with open(path, "rb") as f:
            uploaded = client.files.create(
                file=f,
                purpose="assistants",
            )
            uploaded_file_ids.append(uploaded.id)
            print(f"Uploaded: {path.name} -> {uploaded.id}")

    # ------------------------
    # Create vector store
    # ------------------------
    vector_store = client.vector_stores.create(
        name="seg09_vector_store"
    )
    print(f"Created vector store: {vector_store.id}")

    # ------------------------
    # Create file batch (IMPORTANT)
    # ------------------------
    file_batch = client.vector_stores.file_batches.create(
        vector_store_id=vector_store.id,
        file_ids=uploaded_file_ids,
    )

    file_batch_id = file_batch.id  # <-- must be vsfb_*
    print(f"File batch created: {file_batch_id}")

    # ------------------------
    # Poll until indexing completes
    # ------------------------
    while True:
        status = client.vector_stores.file_batches.retrieve(
            vector_store_id=vector_store.id,
            batch_id=file_batch_id,
        )

        print(f"Indexing status: {status.status}")

        if status.status in ("completed", "failed"):
            break

        time.sleep(2)

    if status.status != "completed":
        raise RuntimeError("Vector store indexing failed")

    # ------------------------
    # Save metadata
    # ------------------------
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    meta = {
        "vector_store_id": vector_store.id,
        "file_batch_id": file_batch_id,
        "file_ids": uploaded_file_ids,
        "docs_dir": str(DOCS_DIR),
        "allowed_exts": sorted(ALLOWED_EXTS),
    }

    out = ARTIFACT_DIR / "vector_store.json"
    out.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved vector store metadata to {out}")


if __name__ == "__main__":
    main()
