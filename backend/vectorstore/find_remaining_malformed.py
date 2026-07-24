"""Find the exact 4 pages with remaining malformed URLs and show every occurrence."""
import json, re
from pathlib import Path

with open(Path(__file__).parent.parent / 'data/processed/gdgu_clean.json',
          encoding='utf-8') as f:
    pages = json.load(f)

_RE = re.compile(r'<(https?:/[^>\s]+)>')

for page in pages:
    text = page.get('content', '')
    hits = list(_RE.finditer(text))
    if not hits:
        continue
    print(f"\nURL: {page['url']}")
    print(f"Occurrences: {len(hits)}")
    for h in hits:
        # Show 60 chars of context around the match
        start = max(0, h.start() - 40)
        end   = min(len(text), h.end() + 40)
        print(f"  MATCH: {repr(h.group(0))}")
        print(f"  CTX  : {repr(text[start:end])}")
