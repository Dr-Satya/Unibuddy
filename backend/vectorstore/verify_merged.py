import json
from collections import Counter
from pathlib import Path

FILE = Path(__file__).parent.parent / "data/processed/gdgu_clean_merged.json"
REQUIRED_KEYS = {"url", "title", "content", "crawled_at", "cleaned_at"}

try:
    with open(FILE, encoding="utf-8") as f:
        data = json.load(f)
    print(f"JSON valid          : YES")
except json.JSONDecodeError as e:
    print(f"JSON valid          : NO — {e}")
    raise SystemExit(1)

total          = len(data)
urls           = [p.get("url","") for p in data]
url_counts     = Counter(urls)
dup_urls       = {u: n for u, n in url_counts.items() if n > 1}
empty_content  = [p for p in data if not (p.get("content") or "").strip()]
short_content  = [p for p in data if 0 < len((p.get("content") or "").strip()) < 100]
schema_errors  = [p for p in data if not REQUIRED_KEYS.issubset(p.keys())]
extra_keys     = [p for p in data if set(p.keys()) - REQUIRED_KEYS]

print(f"Total pages         : {total}")
print(f"Duplicate URLs      : {len(dup_urls)}")
if dup_urls:
    for u, n in list(dup_urls.items())[:5]:
        print(f"  {n}x  {u}")
print(f"Empty content pages : {len(empty_content)}")
print(f"Content < 100 chars : {len(short_content)}")
if short_content:
    for p in short_content[:5]:
        print(f"  {len(p.get('content','')):<5} chars  {p['url']}")
print(f"Schema errors       : {len(schema_errors)}")
if schema_errors:
    for p in schema_errors[:3]:
        print(f"  missing keys: {REQUIRED_KEYS - set(p.keys())}  url: {p.get('url','?')}")
print(f"Extra keys present  : {len(extra_keys)}")
if extra_keys:
    extra = set()
    for p in extra_keys:
        extra |= set(p.keys()) - REQUIRED_KEYS
    print(f"  extra key names: {extra}")
