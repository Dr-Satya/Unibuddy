"""
recleaner.py — Post-cleaning pass to fix three systematic artefacts
====================================================================
Read-only in --dry-run mode. Writes gdgu_clean_v2.json otherwise.

Fixes:
  1. Malformed URLs:  [text](base/<https:/real/url>)
                   → [text](https://real/url)
     Also strips bare <https:/...> angle-bracket tokens.

  2. Footer bottom-nav lines:
       ## [![](arrow.svg)Label](url/<#>)
       ## [![](arrow.svg)Label](url/<#>)  (any variant ending in <#>)
     Removed entirely.

  3. Arrow SVG image tags:
       ![](https://.../white-arrow.svg)
       ![](https://.../icons/...)        (any icon-only image)
     Removed entirely.

Input:  data/processed/gdgu_clean.json
Output: data/processed/gdgu_clean_v2.json

Usage:
    python vectorstore/recleaner.py --dry-run    ← print stats, do NOT write
    python vectorstore/recleaner.py              ← write gdgu_clean_v2.json
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
INPUT_FILE  = BACKEND_DIR / "data" / "processed" / "gdgu_clean.json"
OUTPUT_FILE = BACKEND_DIR / "data" / "processed" / "gdgu_clean_v2.json"

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Fix 1a — markdown link with embedded angle-bracket URL
# [text](https://host/path/<https:/real/url>) → [text](https://real/url)
# [text](https://host/path/<https:/real/url>)
_RE_LINK_EMBEDDED = re.compile(
    r'\[([^\]]*)\]\([^)]*<(https?:/[^>]+)>\)'
)

# Fix 1b — bare angle-bracket URL token remaining after other cleans
# <https:/www.gdgoenkauniversity.com/...>
_RE_BARE_ANGLE = re.compile(r'<(https?:/[^>\s]+)>')

# Fix 2 — footer bottom-nav heading-link lines
# ## [![](anything)Label](anything/<#>)
# ## [![](anything)Label](anything/<javascript:void(0)>)
# Matches the whole line
_RE_FOOTER_NAV_LINE = re.compile(
    r'^##\s+\[.*?\]\([^)]*(?:<#>|<javascript:[^>]*>)\)\s*$',
    re.MULTILINE
)

# Fix 3 — inline icon image tags (arrow SVG, any siteassets/images/icons/ image)
# ![](https://.../siteassets/images/icons/...)
# ![](https://.../siteassets/images/gd-goenka-img/...)
# ![](https://.../uploads/...)    (programme type images, not content images)
_RE_ICON_IMAGE = re.compile(
    r'!\[[^\]]*\]\(https?://[^\)]*(?:'
    r'siteassets/images/icons/|'
    r'siteassets/images/gd-goenka-img/|'
    r'uploads/program_type/'
    r')[^\)]*\)'
)

# ---------------------------------------------------------------------------
# Whitespace cleanup after removals
# ---------------------------------------------------------------------------

def _collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive blank lines to 2."""
    return re.sub(r'\n{3,}', '\n\n', text)


def _fix_single_slash_protocol(url: str) -> str:
    """https:/foo → https://foo (only if not already double-slash)."""
    return re.sub(r'^(https?):/([^/])', r'\1://\2', url)


# ---------------------------------------------------------------------------
# Per-page cleaner
# ---------------------------------------------------------------------------

def clean_page(text: str) -> tuple[str, dict]:
    """
    Apply all three fixes to a single page's content text.
    Returns (cleaned_text, stats_dict).
    """
    stats = {
        "embedded_links_fixed": 0,
        "bare_angles_fixed": 0,
        "footer_lines_removed": 0,
        "icon_images_removed": 0,
    }

    # Fix 1a — markdown links with embedded angle-bracket URLs
    def _fix_embedded(m):
        stats["embedded_links_fixed"] += 1
        label   = m.group(1)
        raw_url = m.group(2)
        url     = _fix_single_slash_protocol(raw_url)
        if label.strip():
            return f"[{label}]({url})"
        return url   # no label → just the URL

    text = _RE_LINK_EMBEDDED.sub(_fix_embedded, text)

    # Fix 1b — bare angle-bracket URL tokens
    def _fix_bare(m):
        stats["bare_angles_fixed"] += 1
        return _fix_single_slash_protocol(m.group(1))

    text = _RE_BARE_ANGLE.sub(_fix_bare, text)

    # Fix 2 — footer nav lines
    n_before = len(_RE_FOOTER_NAV_LINE.findall(text))
    text = _RE_FOOTER_NAV_LINE.sub('', text)
    stats["footer_lines_removed"] = n_before

    # Fix 3 — icon image tags
    n_before = len(_RE_ICON_IMAGE.findall(text))
    text = _RE_ICON_IMAGE.sub('', text)
    stats["icon_images_removed"] = n_before

    # Collapse excess blank lines left by removals
    text = _collapse_blank_lines(text).strip()

    return text, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats only, do NOT write output file")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  recleaner.py — Post-cleaning pass")
    print(f"  Input  : {INPUT_FILE}")
    print(f"  Output : {OUTPUT_FILE}")
    print(f"  Mode   : {'DRY RUN (no writes)' if args.dry_run else 'WRITE'}")
    print(f"{'='*65}\n")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    print(f"Loaded {len(pages)} pages. Processing...\n")

    cleaned_pages = []
    totals = {
        "embedded_links_fixed": 0,
        "bare_angles_fixed": 0,
        "footer_lines_removed": 0,
        "icon_images_removed": 0,
        "total_chars_before": 0,
        "total_chars_after": 0,
    }

    worst = []   # pages with most fixes needed

    for page in pages:
        text_before = page.get("content", "") or ""
        text_after, stats = clean_page(text_before)

        totals["embedded_links_fixed"]  += stats["embedded_links_fixed"]
        totals["bare_angles_fixed"]     += stats["bare_angles_fixed"]
        totals["footer_lines_removed"]  += stats["footer_lines_removed"]
        totals["icon_images_removed"]   += stats["icon_images_removed"]
        totals["total_chars_before"]    += len(text_before)
        totals["total_chars_after"]     += len(text_after)

        total_fixes = sum(stats.values())
        if total_fixes > 0:
            worst.append((total_fixes, page["url"], stats))

        new_page = dict(page)
        new_page["content"]    = text_after
        new_page["recleaned_at"] = datetime.now(timezone.utc).isoformat()
        cleaned_pages.append(new_page)

    # Sort by most fixes needed
    worst.sort(reverse=True)

    reduction = (1 - totals["total_chars_after"] / totals["total_chars_before"]) * 100
    print(f"{'='*65}")
    print(f"  FIX SUMMARY")
    print(f"{'='*65}")
    print(f"  Embedded links fixed      : {totals['embedded_links_fixed']}")
    print(f"  Bare angle URLs fixed     : {totals['bare_angles_fixed']}")
    print(f"  Footer nav lines removed  : {totals['footer_lines_removed']}")
    print(f"  Icon image tags removed   : {totals['icon_images_removed']}")
    print(f"  Total chars before        : {totals['total_chars_before']:,}")
    print(f"  Total chars after         : {totals['total_chars_after']:,}")
    print(f"  Size reduction            : {reduction:.1f}%")
    print(f"\n  Top 10 most-fixed pages:")
    for n, url, s in worst[:10]:
        print(f"    {n:>4} fixes  {url[:65]}")

    if args.dry_run:
        print(f"\n  DRY RUN — no files written.")
    else:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned_pages, f, ensure_ascii=False, indent=2)
        size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(f"\n  Written: {OUTPUT_FILE}  ({size_mb:.2f} MB)")

    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
