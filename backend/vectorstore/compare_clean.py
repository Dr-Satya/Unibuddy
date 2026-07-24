"""Compare gdgu_clean.json vs gdgu_clean_v2.json"""
import json
from pathlib import Path

BASE = Path(__file__).parent.parent / "data" / "processed"

with open(BASE / "gdgu_clean.json",    encoding="utf-8") as f: v1 = json.load(f)
with open(BASE / "gdgu_clean_v2.json", encoding="utf-8") as f: v2 = json.load(f)

v1_map = {p["url"]: p for p in v1}
v2_map = {p["url"]: p for p in v2}

v1_urls = set(v1_map)
v2_urls = set(v2_map)

only_v1 = v1_urls - v2_urls
only_v2 = v2_urls - v1_urls
common  = v1_urls & v2_urls

print(f"{'='*60}")
print(f"  gdgu_clean.json    : {len(v1)} pages")
print(f"  gdgu_clean_v2.json : {len(v2)} pages")
print(f"  Only in v1         : {len(only_v1)}")
print(f"  Only in v2         : {len(only_v2)}")
print(f"  In both            : {len(common)}")
print(f"{'='*60}")

# Content differences in common pages
different_content = []
same_content      = 0
for url in common:
    c1 = (v1_map[url].get("content") or "").strip()
    c2 = (v2_map[url].get("content") or "").strip()
    if c1 != c2:
        different_content.append({
            "url":      url,
            "v1_len":   len(c1),
            "v2_len":   len(c2),
            "delta":    len(c2) - len(c1),
        })
    else:
        same_content += 1

different_content.sort(key=lambda x: abs(x["delta"]), reverse=True)

print(f"\n  Content identical  : {same_content}")
print(f"  Content different  : {len(different_content)}")

if different_content:
    print(f"\n  Top 20 content differences (by delta size):")
    print(f"  {'URL':<65} {'v1':>7} {'v2':>7} {'delta':>7}")
    print(f"  {'-'*65} {'-'*7} {'-'*7} {'-'*7}")
    for d in different_content[:20]:
        sign = "+" if d["delta"] > 0 else ""
        print(f"  {d['url'][:65]:<65} {d['v1_len']:>7} {d['v2_len']:>7} {sign}{d['delta']:>6}")

if only_v1:
    print(f"\n  URLs only in v1 (present in old clean, missing in v2):")
    for u in sorted(only_v1)[:20]:
        print(f"    {u}")

if only_v2:
    print(f"\n  URLs only in v2 (new pages not in old clean):")
    for u in sorted(only_v2)[:20]:
        print(f"    {u}")

# Avg content size comparison
v1_avg = sum(len(p.get("content","") or "") for p in v1) // len(v1) if v1 else 0
v2_avg = sum(len(p.get("content","") or "") for p in v2) // len(v2) if v2 else 0
print(f"\n  Avg content size v1 : {v1_avg:,} chars")
print(f"  Avg content size v2 : {v2_avg:,} chars")
print(f"  Delta               : {v2_avg - v1_avg:+,} chars")
