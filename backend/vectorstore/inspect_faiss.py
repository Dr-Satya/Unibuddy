"""
inspect_faiss.py — Phase 8 validation report
Reads vectorstore/gdgu.index and vectorstore/id_map.json

Usage:
    python vectorstore/inspect_faiss.py
"""

import json
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

SCRIPT_DIR  = Path(__file__).parent
INDEX_FILE  = SCRIPT_DIR / "gdgu.index"
ID_MAP_FILE = SCRIPT_DIR / "id_map.json"
SYNC_FILE   = SCRIPT_DIR / "sync_report.json"

if not INDEX_FILE.exists():
    print(f"ERROR: {INDEX_FILE} not found. Run faiss_manager.py first.")
    raise SystemExit(1)

# ── Load index ────────────────────────────────────────────────────────────────
print("Loading index... ", end="", flush=True)
t0    = time.time()
index = faiss.read_index(str(INDEX_FILE))
load_time = time.time() - t0
print(f"OK  ({load_time:.3f}s)")

# ── Load id_map ───────────────────────────────────────────────────────────────
with open(ID_MAP_FILE, "r", encoding="utf-8") as f:
    id_map = json.load(f)

rows = id_map.get("rows", [])
dim  = id_map.get("dim",  index.d)

live_rows = [r for r in rows if r.get("chunk_id") is not None]
dead_rows = [r for r in rows if r.get("chunk_id") is None]

print(f"\n{'='*65}")
print("  Phase 8 — FAISS Index Inspection Report")
print(f"{'='*65}")

# ── Index basics ──────────────────────────────────────────────────────────────
print(f"\n  {'─'*55}")
print(f"  INDEX")
print(f"  {'─'*55}")
print(f"  Index type             : {type(index).__name__}")
print(f"  Embedding dimension    : {index.d}")
print(f"  Total rows (ntotal)    : {index.ntotal}")
print(f"  Live vectors           : {len(live_rows)}")
print(f"  Dead rows              : {len(dead_rows)}")
if rows:
    dead_pct = len(dead_rows) / len(rows) * 100
    print(f"  Dead-row ratio         : {dead_pct:.1f}%")
    print(f"  Rebuild threshold      : 20.0%  "
          f"{'← rebuild due' if dead_pct >= 20 else '← OK'}")

# ── Timing ────────────────────────────────────────────────────────────────────
print(f"\n  {'─'*55}")
print(f"  TIMING")
print(f"  {'─'*55}")
print(f"  Load time              : {load_time:.3f}s")

# Measure save time with a temp write
import tempfile, os
t1 = time.time()
with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
    tmp_path = tmp.name
faiss.write_index(index, tmp_path)
save_time = time.time() - t1
os.unlink(tmp_path)
print(f"  Save time (measured)   : {save_time:.3f}s")

# ── Embedding value sanity ────────────────────────────────────────────────────
print(f"\n  {'─'*55}")
print(f"  EMBEDDING SANITY")
print(f"  {'─'*55}")

# Sample first 100 vectors for norm check
n_sample = min(100, index.ntotal)
raw_ptr  = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * dim)
matrix   = raw_ptr.reshape(index.ntotal, dim)[:n_sample]
norms    = np.linalg.norm(matrix, axis=1)
nan_count= int(np.isnan(matrix).sum())
inf_count= int(np.isinf(matrix).sum())

print(f"  Sample size            : {n_sample} vectors")
print(f"  NaN values             : {nan_count}")
print(f"  Inf values             : {inf_count}")
print(f"  Avg norm (sample)      : {norms.mean():.4f}")
print(f"  Min norm (sample)      : {norms.min():.4f}")
print(f"  Max norm (sample)      : {norms.max():.4f}")

# ── Category distribution (live rows only) ────────────────────────────────────
cat_counts = Counter(r.get("category", "unknown") for r in live_rows)
print(f"\n  {'─'*55}")
print(f"  CATEGORY DISTRIBUTION  ({len(live_rows)} live vectors)")
print(f"  {'─'*55}")
for cat, count in cat_counts.most_common():
    pct = count / len(live_rows) * 100 if live_rows else 0
    bar = "█" * (count // 10)
    print(f"  {cat:<14} {count:>5}  ({pct:4.1f}%)  {bar}")

# ── Sync summary (if report exists) ──────────────────────────────────────────
if SYNC_FILE.exists():
    with open(SYNC_FILE, "r", encoding="utf-8") as f:
        sync = json.load(f)
    s = sync.get("summary", {})
    print(f"\n  {'─'*55}")
    print(f"  LAST SYNC REPORT  ({sync.get('generated_at','')[:19]})")
    print(f"  {'─'*55}")
    print(f"  Added       : {s.get('new',       0)}")
    print(f"  Modified    : {s.get('modified',  0)}")
    print(f"  Deleted     : {s.get('deleted',   0)}")
    print(f"  Unchanged   : {s.get('unchanged', 0)}")
    print(f"  % changed   : {s.get('pct_changed', 0.0):.1f}%")

# ── Disk sizes ────────────────────────────────────────────────────────────────
index_mb  = INDEX_FILE.stat().st_size  / (1024 * 1024)
idmap_kb  = ID_MAP_FILE.stat().st_size / 1024

print(f"\n  {'─'*55}")
print(f"  DISK")
print(f"  {'─'*55}")
print(f"  gdgu.index size        : {index_mb:.2f} MB")
print(f"  id_map.json size       : {idmap_kb:.1f} KB")
print(f"  Index path             : {INDEX_FILE.resolve()}")
print(f"  ID map path            : {ID_MAP_FILE.resolve()}")

# ── Quick search test ─────────────────────────────────────────────────────────
print(f"\n  {'─'*55}")
print(f"  SEARCH TEST  (query = first vector in index)")
print(f"  {'─'*55}")
query = matrix[0:1].copy()
D, I  = index.search(query, 5)
print(f"  Top-5 FAISS rows : {I[0].tolist()}")
print(f"  Top-5 scores     : {[round(float(s),4) for s in D[0]]}")
for rank, (row_idx, score) in enumerate(zip(I[0], D[0])):
    if 0 <= row_idx < len(rows):
        cid   = rows[row_idx].get("chunk_id", "DEAD")
        title = rows[row_idx].get("page_title", "")[:40]
        print(f"    #{rank+1}  row={row_idx}  score={score:.4f}  "
              f"cid={cid[:12] if cid else 'DEAD'}  {title}")

print(f"\n{'='*65}\n")
