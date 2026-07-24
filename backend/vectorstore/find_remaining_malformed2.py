"""Use the EXACT same regex as audit_clean_content.py to find remaining hits."""
import json, re
from pathlib import Path

with open(Path(__file__).parent.parent / 'data/processed/gdgu_clean.json',
          encoding='utf-8') as f:
    pages = json.load(f)

# Exact pattern used in audit_clean_content.py
_RE_MALFORMED = re.compile(r'<https?:/[^>]+>')

for page in pages:
    text = page.get('content', '')
    hits = list(_RE_MALFORMED.finditer(text))
    if not hits:
        continue
    print(f"\nURL: {page['url']}")
    print(f"Occurrences: {len(hits)}")
    for h in hits[:8]:
        start = max(0, h.start() - 60)
        end   = min(len(text), h.end() + 60)
        print(f"  MATCH: {repr(h.group(0))}")
        print(f"  CTX  : {repr(text[start:end])}")
