"""
The nav block appears to be a large repeated block at the start.
Find the exact text that signals end of the nav mega-menu.
Look for lines that appear in ALL or most pages.
"""
import json, re
from pathlib import Path
from collections import Counter

CRAWL_FILE = Path(__file__).parent.parent / "data" / "raw" / "gdgu_crawl.json"

with open(CRAWL_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Count how often each line (normalized) appears across all pages
line_counts = Counter()
for p in data:
    lines = set(p.get("markdown", "").splitlines())
    for line in lines:
        s = line.strip()
        if s:
            line_counts[s] += 1

# Lines that appear in 150+ of 200 pages = boilerplate/nav/footer
print("Lines appearing in 150+ pages (nav/footer candidates):")
for line, count in line_counts.most_common(60):
    if count >= 150:
        print(f"  {count:>3}x  {line[:100]}")

print()

# Now find what comes IMMEDIATELY AFTER the nav block
# on 3 sample pages - look for the first line that appears in < 50 pages
print("\n--- First unique content line per page (appears in <30 pages) ---")
for idx in [1, 65, 108, 54]:
    p = data[idx]
    lines = p.get("markdown", "").splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s and line_counts.get(s, 0) < 30:
            print(f"\nPage {idx} ({p['url'][-50:]}) — line {i}:")
            print(f"  COUNT={line_counts.get(s,0)}  TEXT: {s[:120]}")
            # Print 10 lines of context
            print("  --- next 10 lines ---")
            for l in lines[i:i+10]:
                print(f"    {l[:100]}")
            break
