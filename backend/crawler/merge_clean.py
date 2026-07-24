"""
merge_clean.py
==============
Merges two cleaned page datasets into one.

Input 1 : data/processed/gdgu_clean.json          (existing)
Input 2 : data/processed/gdgu_clean_missing.json  (new pages)
Output  : data/processed/gdgu_clean_merged.json   (NEW file — never overwrites inputs)

Rules:
  - Deduplicate by URL (strip trailing slash for comparison).
  - If the same URL exists in both files, keep the record from
    gdgu_clean_missing.json (newer crawl wins).
  - Preserves the exact schema: {url, title, content, crawled_at, cleaned_at}
  - Order: existing pages first (preserving their order), then new-only pages.

Does NOT modify any existing file.
"""

import json
import sys
from pathlib import Path
from urllib.parse import urldefrag

BASE     = Path(__file__).parent.parent / "data" / "processed"
FILE_V1  = BASE / "gdgu_clean.json"
FILE_V2  = BASE / "gdgu_clean_missing.json"
OUT_FILE = BASE / "gdgu_clean_merged.json"

# ---------------------------------------------------------------------------
# Guard: do not overwrite inputs
# ---------------------------------------------------------------------------
if OUT_FILE == FILE_V1 or OUT_FILE == FILE_V2:
    print("ERROR: output path would overwrite an input file. Aborting.")
    sys.exit(1)


def norm_url(url: str) -> str:
    u, _ = urldefrag(url)
    return u.strip().rstrip("/")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"Loading {FILE_V1.name} ...", end=" ", flush=True)
with open(FILE_V1, encoding="utf-8") as f:
    v1 = json.load(f)
print(f"{len(v1)} pages")

if not FILE_V2.exists():
    print(f"\nERROR: {FILE_V2} not found.")
    print("Run the cleaning step on gdgu_crawl_missing.json first, then retry.")
    sys.exit(1)

print(f"Loading {FILE_V2.name} ...", end=" ", flush=True)
with open(FILE_V2, encoding="utf-8") as f:
    v2 = json.load(f)
print(f"{len(v2)} pages")

# ---------------------------------------------------------------------------
# Build lookup from v2 (missing pages) — these win on conflict
# ---------------------------------------------------------------------------
v2_map = {norm_url(p["url"]): p for p in v2}

# ---------------------------------------------------------------------------
# Merge: v1 pages first, overriding with v2 where URL matches
# ---------------------------------------------------------------------------
merged  = []
updated = 0
kept_v1 = 0

for page in v1:
    key = norm_url(page["url"])
    if key in v2_map:
        merged.append(v2_map[key])   # newer version wins
        updated += 1
    else:
        merged.append(page)
        kept_v1 += 1

# Add v2 pages that are completely new (not in v1)
v1_keys   = {norm_url(p["url"]) for p in v1}
new_only  = [p for p in v2 if norm_url(p["url"]) not in v1_keys]
merged   += new_only

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

size_mb = OUT_FILE.stat().st_size / (1024 * 1024)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"  MERGE COMPLETE")
print(f"{'='*55}")
print(f"  v1 pages (gdgu_clean.json)         : {len(v1)}")
print(f"  v2 pages (gdgu_clean_missing.json) : {len(v2)}")
print(f"  Kept from v1 unchanged             : {kept_v1}")
print(f"  Updated from v2 (same URL)         : {updated}")
print(f"  Added from v2 (new URLs)           : {len(new_only)}")
print(f"  Total merged pages                 : {len(merged)}")
print(f"  Output size                        : {size_mb:.2f} MB")
print(f"  Output file                        : {OUT_FILE.resolve()}")
print(f"{'='*55}")
print(f"\n  Inputs untouched:")
print(f"    {FILE_V1}")
print(f"    {FILE_V2}")
