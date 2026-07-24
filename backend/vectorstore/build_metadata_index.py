"""
build_metadata_index.py — Phase 7a
Reads  : data/processed/gdgu_embeddings.json
Writes : vectorstore/metadata_index.json

Each record in metadata_index.json:
  {
    "chunk_id":           str   — original chunk ID (MD5)
    "url":                str   — page_url from metadata
    "page_title":         str   — page_title from metadata
    "category":           str   — category from metadata
    "content_hash":       str   — SHA-256 of chunk text (UTF-8)
    "embedding_hash":     str   — SHA-256 of raw embedding bytes (float32 LE)
    "embedding_position": int   — 0-based index in the embeddings array
    "last_updated":       str   — ISO-8601 UTC timestamp of this build
  }

Usage:
    python vectorstore/build_metadata_index.py
"""

import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent                  # backend/vectorstore/
BACKEND_DIR = SCRIPT_DIR.parent                      # backend/
INPUT_FILE  = BACKEND_DIR / "data" / "processed" / "gdgu_embeddings.json"
OUTPUT_FILE = SCRIPT_DIR / "metadata_index.json"


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------
def _content_hash(text: str) -> str:
    """SHA-256 of the chunk text encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embedding_hash(embedding: list) -> str:
    """
    SHA-256 of the embedding serialized as packed float32 little-endian bytes.
    Deterministic regardless of JSON formatting.
    """
    raw = struct.pack(f"<{len(embedding)}f", *embedding)
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*65}")
    print("  Phase 7a — Build Metadata Index")
    print(f"  Input  : {INPUT_FILE}")
    print(f"  Output : {OUTPUT_FILE}")
    print(f"{'='*65}\n")

    # Load embeddings
    print("Loading embeddings... ", end="", flush=True)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"{len(records)} records.")

    now_iso = datetime.now(timezone.utc).isoformat()
    index   = []

    print("Building metadata index...", flush=True)
    for pos, rec in enumerate(records):
        chunk_id  = rec["chunk_id"]
        text      = rec["text"]
        embedding = rec["embedding"]
        meta      = rec.get("metadata", {})

        entry = {
            "chunk_id":           chunk_id,
            "url":                meta.get("page_url",   ""),
            "page_title":         meta.get("page_title", ""),
            "category":           meta.get("category",   ""),
            "content_hash":       _content_hash(text),
            "embedding_hash":     _embedding_hash(embedding),
            "embedding_position": pos,
            "last_updated":       now_iso,
        }
        index.append(entry)

        if (pos + 1) % 200 == 0 or (pos + 1) == len(records):
            print(f"  {pos + 1:>5} / {len(records)} processed...", flush=True)

    # Write
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    out_size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\n{'='*65}")
    print(f"  METADATA INDEX BUILT")
    print(f"{'='*65}")
    print(f"  Total entries    : {len(index)}")
    print(f"  Output size      : {out_size_kb:.1f} KB")
    print(f"  Output path      : {OUTPUT_FILE.resolve()}")
    print(f"  Timestamp        : {now_iso}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
