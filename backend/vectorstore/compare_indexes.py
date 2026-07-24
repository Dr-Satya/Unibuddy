"""Compare gdgu_without_sitemap.index vs gdgu.index"""
import json, faiss, numpy as np
from pathlib import Path
from collections import Counter

VS = Path(__file__).parent

idx1_path = VS / "gdgu_without_sitemap.index"
idx2_path = VS / "gdgu.index"
meta_path = VS / "retrieval_metadata.json"
id_map_path = VS / "id_map.json"

# Check which files exist
for p in [idx1_path, idx2_path, meta_path, id_map_path]:
    print(f"  {'EXISTS' if p.exists() else 'MISSING':<8}  {p.name}  "
          f"({p.stat().st_size/1024/1024:.2f} MB)" if p.exists() else f"  MISSING   {p.name}")

print()

if not idx1_path.exists():
    print("gdgu_without_sitemap.index not found — listing all .index files in vectorstore:")
    for f in VS.glob("*.index"):
        print(f"  {f.name}  {f.stat().st_size/1024/1024:.2f} MB")
    raise SystemExit(1)

# Load both indexes
idx1 = faiss.read_index(str(idx1_path))
idx2 = faiss.read_index(str(idx2_path))

print(f"{'='*60}")
print(f"  INDEX COMPARISON")
print(f"{'='*60}")
print(f"  {'Metric':<30} {'WITHOUT SITEMAP':>18}  {'CURRENT (gdgu)':>18}")
print(f"  {'-'*30} {'-'*18}  {'-'*18}")
print(f"  {'Type':<30} {type(idx1).__name__:>18}  {type(idx2).__name__:>18}")
print(f"  {'Vectors (ntotal)':<30} {idx1.ntotal:>18,}  {idx2.ntotal:>18,}")
print(f"  {'Dimension':<30} {idx1.d:>18}  {idx2.d:>18}")
print(f"  {'Difference in vectors':<30} {idx2.ntotal - idx1.ntotal:>+17,}")

sz1 = idx1_path.stat().st_size / (1024*1024)
sz2 = idx2_path.stat().st_size / (1024*1024)
print(f"  {'File size (MB)':<30} {sz1:>17.2f}  {sz2:>17.2f}")

# Load id_map for category distribution of current index
if id_map_path.exists():
    with open(id_map_path, encoding='utf-8') as f:
        id_map = json.load(f)
    rows = id_map.get("rows", [])
    live = [r for r in rows if r.get("chunk_id")]
    cats = Counter(r.get("category","unknown") for r in live)
    print(f"\n  Current index category distribution ({len(live)} live vectors):")
    for cat, n in cats.most_common():
        pct = n/len(live)*100
        print(f"    {cat:<20} {n:>5}  ({pct:4.1f}%)")

# Load retrieval_metadata for current index
if meta_path.exists():
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    print(f"\n  retrieval_metadata.json entries: {len(meta)}")

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
diff = idx2.ntotal - idx1.ntotal
if diff > 0:
    print(f"  Current gdgu.index has {diff:,} MORE vectors than gdgu_without_sitemap.index")
elif diff < 0:
    print(f"  Current gdgu.index has {abs(diff):,} FEWER vectors than gdgu_without_sitemap.index")
else:
    print(f"  Both indexes have the same number of vectors ({idx1.ntotal:,})")
