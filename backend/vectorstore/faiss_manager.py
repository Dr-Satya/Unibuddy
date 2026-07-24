"""
faiss_manager.py — Phase 8: Incremental FAISS Manager
======================================================

Persists:
    vectorstore/gdgu.index      — FAISS IndexFlatIP (L2-normalised embeddings)
    vectorstore/id_map.json     — {faiss_position: chunk_id} mapping

First run  (no gdgu.index exists):
    Builds a fresh index from ALL records in gdgu_embeddings.json.

Subsequent runs (gdgu.index already exists):
    Reads sync_report.json and applies only the diff:
      - new_chunks      → add vectors
      - modified_chunks → remove old + add new vector
      - deleted_chunks  → remove vector
      - unchanged_chunks→ untouched

Because IndexFlatIP does NOT support in-place removal, the manager uses a
"logical delete + periodic rebuild" strategy:

    id_map.json tracks every (faiss_row → chunk_id) mapping.
    Rows that belong to deleted/modified chunks are marked as DELETED
    (chunk_id set to null) so they are excluded from future searches.
    A full physical rebuild is triggered only when the dead-row ratio
    exceeds REBUILD_THRESHOLD (default 20%).

id_map.json structure:
    {
      "dim":      int,
      "total":    int,               ← ntotal including dead rows
      "live":     int,               ← rows with chunk_id != null
      "dead":     int,               ← rows with chunk_id == null
      "rows":     [                  ← indexed by FAISS row position
        {
          "chunk_id":    str | null,
          "url":         str,
          "page_title":  str,
          "category":    str,
          "content_hash":str,
          "embedding_hash": str,
          "last_updated":str
        },
        ...
      ]
    }

Usage:
    python vectorstore/faiss_manager.py               ← auto-detect first/incremental
    python vectorstore/faiss_manager.py --force-rebuild  ← always full rebuild
    python vectorstore/faiss_manager.py --dry-run        ← show what would change, no writes
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent                 # backend/vectorstore/
BACKEND_DIR  = SCRIPT_DIR.parent                     # backend/
EMB_FILE     = BACKEND_DIR / "data" / "processed" / "gdgu_embeddings.json"
SYNC_REPORT  = SCRIPT_DIR / "sync_report.json"
INDEX_FILE   = SCRIPT_DIR / "gdgu.index"
ID_MAP_FILE  = SCRIPT_DIR / "id_map.json"

# If dead rows exceed this fraction of total, trigger a physical rebuild
REBUILD_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_embeddings(path: Path) -> dict:
    """Return {chunk_id: record} from gdgu_embeddings.json."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["chunk_id"]: r for r in records}


def _vec(embedding: list) -> np.ndarray:
    """Convert list[float] → float32 row vector (1, dim)."""
    return np.array([embedding], dtype=np.float32)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_id_map(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_id_map(id_map: dict, path: Path):
    live = sum(1 for r in id_map["rows"] if r["chunk_id"] is not None)
    dead = len(id_map["rows"]) - live
    id_map["total"] = len(id_map["rows"])
    id_map["live"]  = live
    id_map["dead"]  = dead
    with open(path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)


def _row_entry(chunk_id: str | None, rec: dict | None) -> dict:
    """Build one id_map row entry."""
    if rec is None:
        return {"chunk_id": None, "url": "", "page_title": "",
                "category": "", "content_hash": "", "embedding_hash": "",
                "last_updated": _now()}
    meta = rec.get("metadata", {})
    return {
        "chunk_id":      chunk_id,
        "url":           meta.get("page_url",   ""),
        "page_title":    meta.get("page_title", ""),
        "category":      meta.get("category",   ""),
        "content_hash":  rec.get("content_hash",  ""),
        "embedding_hash":rec.get("embedding_hash",""),
        "last_updated":  _now(),
    }


# ---------------------------------------------------------------------------
# First-run: build fresh index
# ---------------------------------------------------------------------------

def build_fresh(emb_map: dict, dry_run: bool) -> dict:
    """Build FAISS IndexFlatIP from all embeddings. Returns stats dict."""
    print(f"\n  [FIRST RUN] Building fresh index from {len(emb_map)} vectors...")

    sample = next(iter(emb_map.values()))
    dim    = len(sample["embedding"])

    # Build matrix in embedding_position order if available, else dict order
    records_ordered = sorted(
        emb_map.values(),
        key=lambda r: r.get("metadata", {}).get("embedding_position",
                      list(emb_map.values()).index(r))
    )

    matrix = np.array([r["embedding"] for r in records_ordered], dtype=np.float32)
    print(f"  Matrix shape : {matrix.shape}")

    # Build id_map rows
    rows = []
    for rec in records_ordered:
        rows.append(_row_entry(rec["chunk_id"], rec))

    id_map = {
        "dim":   dim,
        "total": len(rows),
        "live":  len(rows),
        "dead":  0,
        "rows":  rows,
    }

    if dry_run:
        print("  [DRY RUN] Would write index and id_map — skipping writes.")
        return {"added": len(rows), "modified": 0, "deleted": 0,
                "unchanged": 0, "rebuilt": True, "dim": dim,
                "ntotal": len(rows)}

    t0 = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    build_time = time.time() - t0

    t1 = time.time()
    faiss.write_index(index, str(INDEX_FILE))
    _save_id_map(id_map, ID_MAP_FILE)
    save_time = time.time() - t1

    print(f"  Index built  : {index.ntotal} vectors  ({build_time:.2f}s)")
    print(f"  Saved        : {INDEX_FILE.name} + {ID_MAP_FILE.name}  ({save_time:.2f}s)")

    # Write a fresh sync_report.json reflecting the rebuild so inspect_faiss.py
    # always shows current state instead of a stale incremental report.
    fresh_sync = {
        "generated_at": _now(),
        "old_source":   str(EMB_FILE),
        "new_source":   str(EMB_FILE),
        "summary": {
            "total_old":    len(rows),
            "total_new":    len(rows),
            "unchanged":    0,
            "modified":     0,
            "deleted":      0,
            "new":          len(rows),
            "pct_changed":  100.0,
        },
        "rebuild_type": "force_rebuild" if INDEX_FILE.exists() else "first_run",
        "new_chunks":       [],
        "modified_chunks":  [],
        "deleted_chunks":   [],
        "unchanged_chunks": [],
    }
    with open(SYNC_REPORT, "w", encoding="utf-8") as f:
        json.dump(fresh_sync, f, ensure_ascii=False, indent=2)
    print(f"  Sync report  : {SYNC_REPORT.name} updated")

    return {"added": len(rows), "modified": 0, "deleted": 0,
            "unchanged": 0, "rebuilt": True, "dim": dim,
            "ntotal": index.ntotal, "build_time": build_time,
            "save_time": save_time}


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def incremental_update(emb_map: dict, sync_report: dict,
                       dry_run: bool, force_rebuild: bool) -> dict:
    """Apply sync_report diff to existing index. Returns stats dict."""

    new_list  = sync_report.get("new_chunks",       [])
    mod_list  = sync_report.get("modified_chunks",  [])
    del_list  = sync_report.get("deleted_chunks",   [])
    unch_list = sync_report.get("unchanged_chunks", [])

    n_new  = len(new_list)
    n_mod  = len(mod_list)
    n_del  = len(del_list)
    n_unch = len(unch_list)

    print(f"\n  [INCREMENTAL] new={n_new}  modified={n_mod}  "
          f"deleted={n_del}  unchanged={n_unch}")

    if n_new == 0 and n_mod == 0 and n_del == 0:
        print("  Nothing to do — index is already up to date.")
        id_map = _load_id_map(ID_MAP_FILE)
        idx    = faiss.read_index(str(INDEX_FILE))
        return {"added": 0, "modified": 0, "deleted": 0,
                "unchanged": n_unch, "rebuilt": False,
                "dim": id_map["dim"], "ntotal": idx.ntotal,
                "build_time": 0.0, "save_time": 0.0}

    # Load current state
    t_load = time.time()
    index  = faiss.read_index(str(INDEX_FILE))
    id_map = _load_id_map(ID_MAP_FILE)
    load_time = time.time() - t_load
    dim    = id_map["dim"]
    rows   = id_map["rows"]   # list indexed by FAISS row position

    print(f"  Loaded index : {index.ntotal} vectors  ({load_time:.2f}s)")

    # Build chunk_id → row-position lookup from current id_map
    cid_to_row: dict[str, int] = {
        row["chunk_id"]: pos
        for pos, row in enumerate(rows)
        if row["chunk_id"] is not None
    }

    # ------------------------------------------------------------------
    # Step 1: Mark deleted + modified-old rows as dead (logical delete)
    # ------------------------------------------------------------------
    deleted_rows: set[int] = set()

    for entry in del_list:
        cid = entry["chunk_id"]
        if cid in cid_to_row:
            pos = cid_to_row[cid]
            rows[pos] = _row_entry(None, None)   # null = dead
            deleted_rows.add(pos)

    for entry in mod_list:
        cid = entry["chunk_id"]
        if cid in cid_to_row:
            pos = cid_to_row[cid]
            rows[pos] = _row_entry(None, None)   # old version → dead
            deleted_rows.add(pos)

    # ------------------------------------------------------------------
    # Step 2: Decide whether to do a physical rebuild
    # ------------------------------------------------------------------
    live_count  = sum(1 for r in rows if r["chunk_id"] is not None)
    dead_count  = len(rows) - live_count
    dead_ratio  = dead_count / len(rows) if rows else 0
    needs_rebuild = force_rebuild or (dead_ratio > REBUILD_THRESHOLD)

    if needs_rebuild:
        print(f"  Dead-row ratio {dead_ratio:.1%} ≥ threshold "
              f"{REBUILD_THRESHOLD:.0%} — triggering physical rebuild.")

    # ------------------------------------------------------------------
    # Step 3: Collect vectors to add (new + modified-new)
    # ------------------------------------------------------------------
    to_add_chunks: list[dict] = []
    for entry in new_list:
        cid = entry["chunk_id"]
        if cid in emb_map:
            to_add_chunks.append(emb_map[cid])
    for entry in mod_list:
        cid = entry["chunk_id"]
        if cid in emb_map:
            to_add_chunks.append(emb_map[cid])

    if dry_run:
        print(f"  [DRY RUN] Would mark {len(deleted_rows)} rows dead, "
              f"add {len(to_add_chunks)} vectors, "
              f"rebuild={needs_rebuild}")
        return {"added": len(to_add_chunks), "modified": n_mod,
                "deleted": n_del, "unchanged": n_unch,
                "rebuilt": needs_rebuild, "dim": dim,
                "ntotal": index.ntotal + len(to_add_chunks) - len(deleted_rows),
                "build_time": 0.0, "save_time": 0.0}

    t_build = time.time()

    if needs_rebuild:
        # Physical rebuild: extract all live vectors + new vectors
        # Pull live rows from the existing FAISS index
        live_positions = [
            pos for pos, row in enumerate(rows)
            if row["chunk_id"] is not None
        ]

        if live_positions:
            live_matrix = np.vstack([
                faiss.rev_swig_ptr(index.get_xb(), index.ntotal * dim)
                .reshape(index.ntotal, dim)[live_positions]
            ])
        else:
            live_matrix = np.empty((0, dim), dtype=np.float32)

        new_matrix = (
            np.array([c["embedding"] for c in to_add_chunks], dtype=np.float32)
            if to_add_chunks else np.empty((0, dim), dtype=np.float32)
        )

        if len(live_matrix) > 0 and len(new_matrix) > 0:
            full_matrix = np.vstack([live_matrix, new_matrix])
        elif len(live_matrix) > 0:
            full_matrix = live_matrix
        else:
            full_matrix = new_matrix

        # Rebuild id_map rows: live first, then new
        new_rows = [rows[pos] for pos in live_positions]
        for chunk in to_add_chunks:
            new_rows.append(_row_entry(chunk["chunk_id"], chunk))

        # Rebuild index
        new_index = faiss.IndexFlatIP(dim)
        if len(full_matrix) > 0:
            new_index.add(full_matrix)
        index  = new_index
        rows   = new_rows

    else:
        # Append-only: just add the new vectors at the end
        if to_add_chunks:
            add_matrix = np.array(
                [c["embedding"] for c in to_add_chunks], dtype=np.float32
            )
            index.add(add_matrix)
            for chunk in to_add_chunks:
                rows.append(_row_entry(chunk["chunk_id"], chunk))

    build_time = time.time() - t_build

    # ------------------------------------------------------------------
    # Step 4: Save
    # ------------------------------------------------------------------
    id_map["dim"]  = dim
    id_map["rows"] = rows

    t_save = time.time()
    faiss.write_index(index, str(INDEX_FILE))
    _save_id_map(id_map, ID_MAP_FILE)
    save_time = time.time() - t_save

    live_final = sum(1 for r in rows if r["chunk_id"] is not None)
    print(f"  Index saved  : {index.ntotal} rows total, "
          f"{live_final} live  ({save_time:.2f}s)")

    return {
        "added":      len(to_add_chunks),
        "modified":   n_mod,
        "deleted":    n_del,
        "unchanged":  n_unch,
        "rebuilt":    needs_rebuild,
        "dim":        dim,
        "ntotal":     index.ntotal,
        "live":       live_final,
        "build_time": build_time,
        "save_time":  save_time,
        "load_time":  load_time,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Incremental FAISS Manager for UnibudDY"
    )
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Force full physical rebuild even if diff is small")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing any files")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print("  Phase 8 — Incremental FAISS Manager")
    print(f"  Embeddings : {EMB_FILE.name}")
    print(f"  Index      : {INDEX_FILE}")
    print(f"  ID map     : {ID_MAP_FILE}")
    if args.dry_run:
        print("  MODE       : DRY RUN (no files will be written)")
    if args.force_rebuild:
        print("  MODE       : FORCE REBUILD")
    print(f"{'='*65}")

    # Load embeddings
    print("\nLoading embeddings... ", end="", flush=True)
    emb_map = _load_embeddings(EMB_FILE)
    print(f"{len(emb_map)} records.")

    is_first_run = not INDEX_FILE.exists() or not ID_MAP_FILE.exists()

    if is_first_run or args.force_rebuild:
        if not is_first_run and args.force_rebuild:
            print("Force-rebuild requested — treating as first run.")
        stats = build_fresh(emb_map, dry_run=args.dry_run)
    else:
        # Load sync report
        if not SYNC_REPORT.exists():
            print(f"ERROR: {SYNC_REPORT} not found.")
            print("Run compare_embeddings.py first to generate the sync report.")
            raise SystemExit(1)
        with open(SYNC_REPORT, "r", encoding="utf-8") as f:
            sync_report = json.load(f)
        stats = incremental_update(
            emb_map, sync_report,
            dry_run=args.dry_run,
            force_rebuild=args.force_rebuild,
        )

    # Final report
    index_size_mb = INDEX_FILE.stat().st_size / (1024*1024) if INDEX_FILE.exists() else 0
    id_map_size_kb= ID_MAP_FILE.stat().st_size / 1024        if ID_MAP_FILE.exists() else 0

    print(f"\n{'='*65}")
    print(f"  FAISS MANAGER COMPLETE")
    print(f"{'='*65}")
    print(f"  Index type             : IndexFlatIP")
    print(f"  Embedding dimension    : {stats.get('dim', 384)}")
    print(f"  Total rows in index    : {stats.get('ntotal', 0)}")
    print(f"  Live vectors           : {stats.get('live', stats.get('ntotal',0))}")
    print(f"  Vectors added          : {stats.get('added', 0)}")
    print(f"  Vectors modified       : {stats.get('modified', 0)}")
    print(f"  Vectors removed        : {stats.get('deleted', 0)}")
    print(f"  Unchanged vectors      : {stats.get('unchanged', 0)}")
    print(f"  Physical rebuild       : {'YES' if stats.get('rebuilt') else 'NO'}")
    print(f"  Build time             : {stats.get('build_time', 0):.2f}s")
    print(f"  Save time              : {stats.get('save_time',  0):.2f}s")
    if 'load_time' in stats:
        print(f"  Load time              : {stats.get('load_time', 0):.2f}s")
    print(f"  Index size on disk     : {index_size_mb:.2f} MB")
    print(f"  ID map size on disk    : {id_map_size_kb:.1f} KB")
    print(f"  Index path             : {INDEX_FILE.resolve()}")
    print(f"  ID map path            : {ID_MAP_FILE.resolve()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
