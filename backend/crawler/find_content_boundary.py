"""
Find exactly where the nav block ends on each page.
The nav is a massive bullet list. Content starts after it.
Strategy: find the LAST occurrence of the nav sentinel line
'Admission Enquiry' or the last mega-menu bullet before real text.
"""
import json, re
from pathlib import Path

CRAWL_FILE = Path(__file__).parent.parent / "data" / "raw" / "gdgu_crawl.json"

with open(CRAWL_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Sentinel phrases that mark end of nav / start of content
NAV_END_MARKERS = [
    "Admission Enquiry",
    "[Apply Now]",
    "Apply Now",
]

for idx in [1, 65, 108, 54, 91]:
    p = data[idx]
    md = p.get("markdown", "")
    lines = md.splitlines()

    # Find last nav-end marker
    nav_end_line = 0
    for i, line in enumerate(lines):
        for marker in NAV_END_MARKERS:
            if marker in line:
                nav_end_line = i

    print(f"\n{'='*60}")
    print(f"URL  : {p['url'][:70]}")
    print(f"Lines: {len(lines)}  Nav ends at line: {nav_end_line}")
    print(f"--- Content after nav (lines {nav_end_line+1} to {nav_end_line+25}) ---")
    content_lines = lines[nav_end_line+1:nav_end_line+25]
    print("\n".join(content_lines[:20]))
