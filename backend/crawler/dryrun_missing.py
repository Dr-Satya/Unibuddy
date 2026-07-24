import json
from pathlib import Path

AUDIT = Path(__file__).parent.parent / "vectorstore" / "crawl_coverage_audit.json"
with open(AUDIT, encoding="utf-8") as f:
    audit = json.load(f)

missing = []
for category, items in audit.get("missing_urls", {}).items():
    for entry in items:
        reason = entry.get("reason", "")
        url    = entry.get("url", "").strip()
        if "NOT_DISCOVERED" not in reason:
            continue
        if not url or "<" in url or " " in url or "\\" in url:
            continue
        if not url.startswith(("http://", "https://")):
            continue
        missing.append(url)

seen = set(); unique = []
for u in missing:
    if u not in seen:
        seen.add(u); unique.append(u)

print(f"Missing URLs to crawl: {len(unique)}")
print("First 5:")
for u in unique[:5]: print(" ", u)
print("Last 5:")
for u in unique[-5:]: print(" ", u)

from collections import Counter
cats = Counter()
for category, items in audit.get("missing_urls", {}).items():
    for entry in items:
        if "NOT_DISCOVERED" in entry.get("reason",""):
            url = entry.get("url","")
            if url and "<" not in url and " " not in url and "\\" not in url:
                cats[category] += 1
print("\nBy category:")
for cat, n in cats.most_common():
    print(f"  {cat:<25} {n}")

