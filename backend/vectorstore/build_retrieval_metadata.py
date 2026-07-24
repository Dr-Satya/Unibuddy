"""
build_retrieval_metadata.py
===========================
Converts gdgu_embeddings.json into a retrieval_metadata.json that is
drop-in compatible with the existing metadata.json format expected by
the retrieval pipeline.

Input  : data/processed/gdgu_embeddings.json
Output : vectorstore/retrieval_metadata.json

Output format (keys are FAISS row indices as strings):
{
  "0": {
    "chunk_id":    str,
    "content":     str,   <- mapped from text
    "source":      str,   <- mapped from metadata.page_url
    "title":       str,   <- mapped from metadata.page_title
    "content_type":str    <- mapped from metadata.category
  },
  "1": { ... },
  ...
}

Key ordering matches the order of records in gdgu_embeddings.json so that
FAISS row index N maps directly to key "N" in this file.

Usage:
    python vectorstore/build_retrieval_metadata.py
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
INPUT_FILE  = BACKEND_DIR / "data" / "processed" / "gdgu_embeddings.json"
OUTPUT_FILE = SCRIPT_DIR  / "retrieval_metadata.json"


def main():
    # Load embeddings
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Build metadata dict keyed by FAISS row index
    retrieval_meta: dict = {}
    for pos, rec in enumerate(records):
        m = rec.get("metadata", {})
        retrieval_meta[str(pos)] = {
            "chunk_id":    rec.get("chunk_id", ""),
            "content":     rec.get("text", ""),
            "source":      m.get("page_url",   ""),
            "title":       m.get("page_title", ""),
            "content_type": m.get("category",  "general"),
        }

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(retrieval_meta, f, ensure_ascii=False, indent=2)

    # Validation
    total      = len(retrieval_meta)
    size_mb    = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    first      = retrieval_meta["0"]
    last       = retrieval_meta[str(total - 1)]

    assert total == len(records),        "Record count mismatch"
    assert first["content"],             "First record has empty content"
    assert last["content"],              "Last record has empty content"
    assert first["source"].startswith("http"), "First record source is not a URL"
    assert last["source"].startswith("http"),  "Last record source is not a URL"
    assert all(str(i) in retrieval_meta for i in range(total)), "Key sequence broken"

    print(f"Total records : {total}")
    print(f"Output file   : {OUTPUT_FILE.resolve()}")
    print(f"Output size   : {size_mb:.2f} MB")
    print("Validation Passed")


if __name__ == "__main__":
    main()
