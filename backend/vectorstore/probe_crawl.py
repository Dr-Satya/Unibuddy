import json
from pathlib import Path

base = Path(__file__).parent.parent

# raw crawl
with open(base / 'data/raw/gdgu_crawl.json', encoding='utf-8') as f:
    raw = json.load(f)

# clean pages
with open(base / 'data/processed/gdgu_clean.json', encoding='utf-8') as f:
    clean = json.load(f)

print(f"Raw crawl records   : {len(raw)}")
print(f"Clean page records  : {len(clean)}")
print(f"\nFirst 5 raw URLs:")
for p in raw[:5]:
    print(f"  {p['url']}")
print(f"\nLast 5 raw URLs:")
for p in raw[-5:]:
    print(f"  {p['url']}")

# Check for path segments that exist in raw
from urllib.parse import urlparse
from collections import Counter
paths = [urlparse(p['url']).path.strip('/').split('/')[0] for p in raw if urlparse(p['url']).path.strip('/')]
print(f"\nTop URL path segments in raw crawl:")
for seg, n in Counter(paths).most_common(20):
    print(f"  {n:>4}x  /{seg}/")

# How many URLs contain 'faculty' or 'staff' or 'people'
fac = [p['url'] for p in raw if any(k in p['url'].lower() for k in ['faculty','staff','people','dr-','professor'])]
print(f"\nFaculty-style URLs in raw crawl: {len(fac)}")
for u in fac[:10]:
    print(f"  {u}")
