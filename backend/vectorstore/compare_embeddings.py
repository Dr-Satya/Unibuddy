"""
compare_embeddings.py — Phase 7b
Compares OLD  : vectorstore/metadata_index.json   (previous state)
vs      NEW   : data/processed/gdgu_embeddings.json (current crawl)

Produces : vectorstore/sync_report.json

Sync logic:
  - A chunk is UNCHANGED   if chunk_id exists in both AND content_hash
                            AND embedding_hash are identical.
  - A chunk is MODIFIED    if chunk_id exists in both BUT content_hash
                            OR embedding_hash differ.
  - A chunk is DELETED     if chunk_id exists in old index but NOT in
                            the new embeddings.
  - A chunk is NEW         if chunk_id exists in new embeddings but NOT
                            in the old index.

sync_report.json structure:
  {
    "generated_at":    ISO-8601 UTC,
    "old_source":      path to metadata_index.json,
    "new_source":      path to gdgu_embeddings.json,
    "summary": {
      "total_old":     int,
      "total_new":     int,
      "unchanged":     int,
      "modified":      int,
      "deleted":       int,
      "new":           int,
      "pct_changed":   float
    },
    "new_chunks":      [ {chunk_id, url, page_title, category,
                          content_hash, embedding_hash, embedding_position} ]
    "modified_chunks": [ {chunk_id, url, page_title, category,
                          old_content_hash, new_content_hash,
                          old_embedding_hash, new_embedding_hash,
                          embedding_position} ]
    "deleted_chunks":  [ {chunk_id, url, page_title, category,
                          old_embedding_position} ]
    "unchanged_chunks":[ {chunk_id, url, page_title, category} ]
  }

Usage:
    python vectorstore/compare_embeddings.py
    python vectorstore/compare_embeddings.py --old path/to/old_index.json
"""

import argparse
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
BACKEND_DIR  = SCRIPT_DIR.parent
OLD_INDEX    = SCRIPT_DIR / "metadata_index.json"
NEW_EMB      = BACKEND_DIR / "data" / "processed" / "gdgu_embeddings.json"
REPORT_OUT   = SCRIPT_DIR / "sync_report.json"


# ---------------------------------------------------------------------------
# Hash helpers (must match build_metadata_index.py exactly)
# ---------------------------------------------------------------------------
def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embedding_hash(embedding: list) -> str:
    raw = struct.pack(f"<{len(embedding)}f", *embedding)
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare old metadata index vs new embeddings."
    )
    parser.add_argument("--old", type=str, default=str(OLD_INDEX),
                        help="Path to old metadata_index.json")
    parser.add_argument("--new", type=str, default=str(NEW_EMB),
                        help="Path to new gdgu_embeddings.json")
    parser.add_argument("--output", type=str, default=str(REPORT_OUT),
                        help="Path to write sync_report.json")
    args = parser.parse_args()

    old_path    = Path(args.old)
    new_path    = Path(args.new)
    output_path = Path(args.output)

    print(f"\n{'='*65}")
    print("  Phase 7b — Embedding Synchronization Comparator")
    print(f"  Old index : {old_path}")
    print(f"  New embs  : {new_path}")
    print(f"  Report    : {output_path}")
    print(f"{'='*65}\n")

    # ── Load old index ────────────────────────────────────────────────────────
    if not old_path.exists():
        print(f"ERROR: Old index not found at {old_path}")
        print("Run build_metadata_index.py first to create the baseline.")
        raise SystemExit(1)

    print("Loading old metadata index... ", end="", flush=True)
    with open(old_path, "r", encoding="utf-8") as f:
        old_list = json.load(f)
    # Build lookup by chunk_id
    old_map: dict = {e["chunk_id"]: e for e in old_list}
    print(f"{len(old_map)} entries.")

    # ── Load new embeddings ───────────────────────────────────────────────────
    print("Loading new embeddings...     ", end="", flush=True)
    with open(new_path, "r", encoding="utf-8") as f:
        new_list = json.load(f)
    print(f"{len(new_list)} records.")

    # ── Compare ───────────────────────────────────────────────────────────────
    print("Comparing...", flush=True)

    new_chunks       = []
    modified_chunks  = []
    unchanged_chunks = []

    new_ids: set = set()

    for pos, rec in enumerate(new_list):
        cid      = rec["chunk_id"]
        text     = rec["text"]
        embedding= rec["embedding"]
        meta     = rec.get("metadata", {})

        new_ids.add(cid)

        c_hash = _content_hash(text)
        e_hash = _embedding_hash(embedding)

        base = {
            "chunk_id":           cid,
            "url":                meta.get("page_url",   ""),
            "page_title":         meta.get("page_title", ""),
            "category":           meta.get("category",   ""),
            "content_hash":       c_hash,
            "embedding_hash":     e_hash,
            "embedding_position": pos,
        }

        if cid not in old_map:
            # Brand-new chunk
            new_chunks.append(base)

        else:
            old_entry = old_map[cid]
            content_changed   = old_entry.get("content_hash",   "") != c_hash
            embedding_changed = old_entry.get("embedding_hash", "") != e_hash

            if content_changed or embedding_changed:
                modified_chunks.append({
                    **base,
                    "old_content_hash":   old_entry.get("content_hash",   ""),
                    "new_content_hash":   c_hash,
                    "old_embedding_hash": old_entry.get("embedding_hash", ""),
                    "new_embedding_hash": e_hash,
                    "content_changed":    content_changed,
                    "embedding_changed":  embedding_changed,
                })
            else:
                unchanged_chunks.append({
                    "chunk_id":   cid,
                    "url":        meta.get("page_url",   ""),
                    "page_title": meta.get("page_title", ""),
                    "category":   meta.get("category",   ""),
                })

    # Deleted: in old but not in new
    deleted_chunks = []
    for cid, old_entry in old_map.items():
        if cid not in new_ids:
            deleted_chunks.append({
                "chunk_id":            cid,
                "url":                 old_entry.get("url",        ""),
                "page_title":          old_entry.get("page_title", ""),
                "category":            old_entry.get("category",   ""),
                "old_embedding_position": old_entry.get("embedding_position", -1),
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    total_old  = len(old_map)
    total_new  = len(new_list)
    n_changed  = len(new_chunks) + len(modified_chunks) + len(deleted_chunks)
    pct_changed= round(n_changed / max(total_old, total_new) * 100, 2)

    summary = {
        "total_old":  total_old,
        "total_new":  total_new,
        "unchanged":  len(unchanged_chunks),
        "modified":   len(modified_chunks),
        "deleted":    len(deleted_chunks),
        "new":        len(new_chunks),
        "pct_changed": pct_changed,
    }

    now_iso = datetime.now(timezone.utc).isoformat()
    report  = {
        "generated_at":    now_iso,
        "old_source":      str(old_path),
        "new_source":      str(new_path),
        "summary":         summary,
        "new_chunks":      new_chunks,
        "modified_chunks": modified_chunks,
        "deleted_chunks":  deleted_chunks,
        "unchanged_chunks":unchanged_chunks,
    }

    # ── Write report ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    out_size_kb = output_path.stat().st_size / 1024

    print(f"\n{'='*65}")
    print(f"  SYNC REPORT")
    print(f"{'='*65}")
    print(f"  Old chunks          : {total_old}")
    print(f"  New chunks          : {total_new}")
    print(f"  Unchanged           : {len(unchanged_chunks)}")
    print(f"  Modified            : {len(modified_chunks)}")
    print(f"  Deleted             : {len(deleted_chunks)}")
    print(f"  New (added)         : {len(new_chunks)}")
    print(f"  % changed           : {pct_changed}%")
    print(f"  Report size         : {out_size_kb:.1f} KB")
    print(f"  Report path         : {output_path.resolve()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
