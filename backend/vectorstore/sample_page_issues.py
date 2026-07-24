"""Show real examples of the three main issues from the audit."""
import json, re
from pathlib import Path

with open(Path(__file__).parent.parent / 'data/processed/gdgu_clean.json', encoding='utf-8') as f:
    pages = json.load(f)

# Pick the about-gd-goenka-group page (rank 1 worst)
target_urls = [
    "https://www.gdgoenkauniversity.com/about-us/about-gd-goenka-group",
    "https://www.gdgoenkauniversity.com/programmes/undergraduate",
    "https://www.gdgoenkauniversity.com/admissions/fee-structure",
]

_RE_MALFORMED = re.compile(r'<https?:/[^>]+>')
_RE_FOOTER    = re.compile(r'© Copyright|All Rights Reserved|GD Goenka Group|About Group|Follow Us|Quick Links', re.I)
_RE_IMAGE     = re.compile(r'!\[([^\]]*)\]\([^)]+\)')

for url in target_urls:
    page = next((p for p in pages if p['url'] == url), None)
    if not page:
        print(f"NOT FOUND: {url}")
        continue
    text = page.get('content','')
    lines = text.splitlines()
    
    print(f"\n{'='*70}")
    print(f"URL: {url}")
    print(f"Total chars: {len(text)}  Lines: {len(lines)}")
    
    # Malformed URLs
    bad = _RE_MALFORMED.findall(text)
    print(f"\nMALFORMED URLs ({len(bad)} found) — first 5:")
    for b in bad[:5]:
        print(f"  {b[:100]}")
    
    # Footer lines
    footer_lines = [l for l in lines if _RE_FOOTER.search(l)]
    print(f"\nFOOTER lines ({len(footer_lines)} found) — first 3:")
    for l in footer_lines[:3]:
        print(f"  {l[:100]}")
    
    # Image tags
    imgs = _RE_IMAGE.findall(text)
    print(f"\nIMAGE tags ({len(imgs)} found) — first 3 full matches:")
    for m in _RE_IMAGE.finditer(text):
        print(f"  {m.group(0)[:100]}")
        if len(imgs) > 3: break
    
    # Show last 300 chars (footer area)
    print(f"\nLAST 400 CHARS (footer area):")
    print(repr(text[-400:]))
