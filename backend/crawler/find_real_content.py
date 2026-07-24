"""
On GD Goenka pages the structure is:
  [NAV BLOCK]  — from line 0 to some line
  [PAGE CONTENT] — the actual unique content
  [FOOTER NAV] — repeated footer links at the end

The nav block is ALL bullet-list lines + a few ## headings.
The content starts at the FIRST non-nav line AFTER the main nav.

Key observation: the nav mega-menu ends with one of these patterns
before the first real prose paragraph:
  - A standalone # heading (page title)
  - A non-bulleted, non-image, non-link paragraph

Let's find pages where content is clearly visible.
"""
import json, re
from pathlib import Path

CRAWL_FILE = Path(__file__).parent.parent / "data" / "raw" / "gdgu_crawl.json"

with open(CRAWL_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

def is_nav_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    # Bullet list items
    if s.startswith(('* [', '* !', '    * ', '  * ')):
        return True
    # Standalone link lines like [text](url)
    if re.match(r'^\[.*\]\(.*\)\s*$', s):
        return True
    # Image-only lines
    if re.match(r'^!?\[.*\]\(.*\)\s*$', s):
        return True
    # Lines that are just markdown link text with images inside
    if s.startswith('[ !'):
        return True
    # CRC/nav heading words that appear in the menu
    nav_headings = {'Placements', 'CRC', 'Happenings', 'IQAC', 'IIQA',
                    'Rankings', 'GD Goenka Group', 'Facilities', 'Research',
                    'Campus Life', 'Internationalisation', 'Admissions',
                    'Programmes', 'Schools', 'Academics'}
    if s in nav_headings:
        return True
    return False

# Look for first real content line after a sequence of nav lines
def find_content_start(lines):
    # Count consecutive nav lines from start
    nav_end = 0
    for i, line in enumerate(lines):
        if is_nav_line(line):
            nav_end = i
        else:
            # First non-nav line
            if i > 20:  # must be past the nav header area
                return i
    return nav_end + 1

for idx in [1, 2, 65, 108, 54, 91, 42]:
    p = data[idx]
    md = p.get("markdown", "")
    lines = md.splitlines()
    cs = find_content_start(lines)
    print(f"\n{'='*60}")
    print(f"URL  : {p['url'][:70]}")
    print(f"Lines: {len(lines)}   Content starts: {cs}")
    snippet = "\n".join(lines[cs:cs+30])
    print(snippet[:1000])
