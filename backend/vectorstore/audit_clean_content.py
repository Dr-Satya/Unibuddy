"""Audit gdgu_clean.json for nav artifacts and malformed URLs. Read-only."""
import json, re
from pathlib import Path
from collections import Counter

with open(Path(__file__).parent.parent / 'data/processed/gdgu_clean.json',
          encoding='utf-8') as f:
    pages = json.load(f)

# ── Patterns ──────────────────────────────────────────────────────────────
_RE_MALFORMED  = re.compile(r'<https?:/[^>]+>')
_RE_NAV_HEADING= re.compile(r'^#{1,4} \[.+\]\(.+\)\s*$', re.MULTILINE)
_RE_ANGLE_LINK = re.compile(r'<https?://')  # even partially fixed
_RE_PROSE      = re.compile(r'[A-Za-z]{4,}.*[A-Za-z]{4,}')  # real sentence

# ── Per-page checks ────────────────────────────────────────────────────────
malformed_pages   = 0
nav_heading_pages = 0
no_prose_pages    = 0
total_malformed   = 0
total_nav_headings= 0

starts_with_nav = 0  # pages whose first non-blank line is a nav heading

for page in pages:
    text = page.get('content', '')
    lines = [l for l in text.splitlines() if l.strip()]

    # Malformed URLs
    mf = _RE_MALFORMED.findall(text)
    if mf:
        malformed_pages += 1
        total_malformed += len(mf)

    # Nav heading links (### [Text](url))
    nh = _RE_NAV_HEADING.findall(text)
    if nh:
        nav_heading_pages += 1
        total_nav_headings += len(nh)

    # First line check
    if lines:
        first = lines[0].strip()
        if _RE_NAV_HEADING.match(first):
            starts_with_nav += 1

    # Pages with no prose
    if not _RE_PROSE.search(text):
        no_prose_pages += 1

print(f"Total clean pages       : {len(pages)}")
print(f"\nMALFORMED URLs (<https:/...>)")
print(f"  Pages affected        : {malformed_pages} / {len(pages)}  ({malformed_pages/len(pages)*100:.0f}%)")
print(f"  Total occurrences     : {total_malformed}")

print(f"\nNAV HEADING LINKS (### [Text](url))")
print(f"  Pages affected        : {nav_heading_pages} / {len(pages)}  ({nav_heading_pages/len(pages)*100:.0f}%)")
print(f"  Total occurrences     : {total_nav_headings}")
print(f"  Pages starting with nav: {starts_with_nav}")

print(f"\nNO PROSE pages          : {no_prose_pages}")

# ── Show first 200 chars of 3 affected pages ─────────────────────────────
print(f"\n─── SAMPLE: 3 pages starting with nav headings ───")
shown = 0
for page in pages:
    text  = page.get('content', '')
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and _RE_NAV_HEADING.match(lines[0].strip()):
        print(f"\nURL: {page['url']}")
        print(f"First 300 chars:\n{text[:300]}")
        shown += 1
        if shown >= 3:
            break

# ── Where does real content start on a sample page? ──────────────────────
print(f"\n─── SAMPLE: where real content starts (5 pages) ───")
for page in pages[1:6]:
    text  = page.get('content', '')
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if (s and not _RE_NAV_HEADING.match(s)
                and not s.startswith('###')
                and not _RE_MALFORMED.search(s)
                and len(s) > 40
                and _RE_PROSE.search(s)):
            print(f"\nURL: {page['url'][:65]}")
            print(f"  First prose at line {i}: {s[:120]}")
            break

# ── Count lines that are nav headings across all pages ────────────────────
all_lines_total = 0
nav_lines_total = 0
for page in pages:
    lines = page.get('content', '').splitlines()
    all_lines_total += len(lines)
    nav_lines_total += sum(1 for l in lines if _RE_NAV_HEADING.match(l.strip()))

print(f"\n─── OVERALL LINE ANALYSIS ───")
print(f"  Total lines across all pages  : {all_lines_total:,}")
print(f"  Nav heading link lines        : {nav_lines_total:,}  ({nav_lines_total/all_lines_total*100:.1f}%)")
print(f"  Effective content lines       : {all_lines_total - nav_lines_total:,}")
