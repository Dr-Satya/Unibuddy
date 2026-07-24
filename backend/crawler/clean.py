"""
clean.py — Phase 3 preprocessing pipeline
Reads  : data/raw/gdgu_crawl.json
Writes : data/processed/gdgu_clean.json

What it does per page:
  1. Remove every line that is site-wide boilerplate (appears in 150+ pages).
  2. Remove navigation bullet lines  (* [text](url)).
  3. Remove image-only / link-only lines.
  4. Remove javascript: / mailto: link lines.
  5. Remove chatbot widget lines.
  6. Remove empty lines runs (collapse to single blank).
  7. Remove duplicate paragraphs within the same page (exact + near-duplicate).
  8. Normalize whitespace.
  9. Extract a clean title from the first H1 on the page.
 10. Drop pages with < 100 chars of content after cleaning.

Usage:
    python crawler/clean.py
"""

import json
import re
import hashlib
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
BACKEND_DIR  = SCRIPT_DIR.parent
INPUT_FILE   = BACKEND_DIR / "data" / "raw"       / "gdgu_crawl.json"
OUTPUT_FILE  = BACKEND_DIR / "data" / "processed" / "gdgu_clean.json"

# ---------------------------------------------------------------------------
# Step 1: Build boilerplate line set
# Collect every line that appears verbatim in 150+ of the 200 pages.
# These are nav/footer lines present on every page.
# ---------------------------------------------------------------------------
BOILERPLATE_THRESHOLD = 150   # line must appear in this many pages to be boilerplate


def build_boilerplate_set(data: list) -> set:
    counts = Counter()
    for page in data:
        seen = set()
        for line in page.get("markdown", "").splitlines():
            s = line.strip()
            if s and s not in seen:
                counts[s] += 1
                seen.add(s)
    boilerplate = {line for line, count in counts.items()
                   if count >= BOILERPLATE_THRESHOLD}
    return boilerplate


# ---------------------------------------------------------------------------
# Line-level filters
# ---------------------------------------------------------------------------

# Pure markdown link:         [text](url)
_RE_LINK_ONLY    = re.compile(r'^\[.*?\]\([^)]+\)\s*$')
# Image tag (with or without leading link bracket):  ![...](...)  or  [ ![...](...) ](...)
_RE_IMAGE_ONLY   = re.compile(r'^!?\[.*?\]\([^)]+\)\s*$')
_RE_IMAGE_LINK   = re.compile(r'^\[\s*!\[')
# Navigation bullet:          * [text](url)  or     * ![img](url)
_RE_NAV_BULLET   = re.compile(r'^\s*\*\s+[\[!]')
# Numbered nav link:          1. [text](url)
_RE_NUM_NAV      = re.compile(r'^\s*\d+\.\s+\[')
# Javascript / mailto links
_RE_JS_MAILTO    = re.compile(r'javascript:|mailto:|tel:', re.IGNORECASE)
# Chatbot widget lines
_RE_CHATBOT      = re.compile(r'chatcdn\.npfs|refreshChatBotSession|Chat with me', re.IGNORECASE)
# Lines that are only whitespace / markdown horizontal rules
_RE_HR           = re.compile(r'^[-*_]{3,}\s*$')
# Lines that are only punctuation/symbols
_RE_SYMBOL_ONLY  = re.compile(r'^[|\\/*\-–—=+~^`#>]{1,5}\s*$')
# Heading navigation link:    ## [Text](url)  or  ### [![img](...)Text](url)
# A heading line whose ENTIRE non-whitespace content is a markdown link.
# Matches  ## [text](url)  and  ## [![img](url)text](url)  but NOT  ## About Us
_RE_HEADING_NAV  = re.compile(r'^#{1,4}\s+\[.+\]\([^)]*\)\s*$')
# Heading with embedded image nav:  ## [![](url)Label](url)
_RE_HEADING_IMG_NAV = re.compile(r'^#{1,4}\s+\[!\[')
# Malformed URL: angle-bracket embedded  base/<https:/real>  or bare <https:/...>
# The [^>]+ (not [^>\s]+) handles URLs with spaces in filenames (PDF links)
_RE_MALFORMED_URL = re.compile(r'<(https?:/[^>]+)>')

# ---------------------------------------------------------------------------
# Post-line-filter text transforms (applied to intermediate text as a whole)
# These strip UI artifacts while keeping the label text where useful.
# ---------------------------------------------------------------------------

# 1. Markdown links with label → keep label only:  [Apply Now](url) → Apply Now
#    Exception: keep table-cell links whose label is a course/prog name
_RE_MD_LINK_LABELLED = re.compile(r'\[([^\]]+)\]\([^)]+\)')

# 2. Empty / icon markdown links → remove entirely:  [](url)  or  [ ](url)
_RE_MD_LINK_EMPTY = re.compile(r'\[\s*\]\([^)]+\)')

# 3. Image markdown → remove entirely:  ![alt](url)
_RE_IMAGE_MD = re.compile(r'!\[[^\]]*\]\([^)]+\)')

# 4. Broken double-URL:  https://host/https://... or http://host/http://...
#    These are navigation links whose href was already absolute.
_RE_BROKEN_URL = re.compile(r'https?://[^\s)]*https?://[^\s)]*')

# 5. Standalone bare URLs (after all markdown has been stripped)
_RE_BARE_URL = re.compile(r'https?://\S+|www\.\S+|mailto:\S+|tel:\S+')

# 6. Footer attribution line (varies only in the leading path prefix)
_RE_FOOTER = re.compile(
    r'Website Design and Development by\s*\[?\s*Sterco Digitex\s*\]?[^\n]*',
    re.IGNORECASE
)

# 7. CTA / UI button text — standalone line OR isolated inline token
#    Pattern A: entire line is just a CTA phrase (with optional bullet/star)
#    Pattern B: CTA word surrounded by whitespace/punctuation on a short line
_RE_CTA_LINE = re.compile(
    r'^\s*[\*\-]?\s*(Apply Now|Read More|View More|Click Here|Download Brochure|'
    r'Download PDF|Register Now|Enquire Now|Book Now|Get Brochure|'
    r'Know More|Explore More|Learn More)\s*[!.]?\s*$',
    re.IGNORECASE | re.MULTILINE
)

# 8. Social media label lines
_RE_SOCIAL_LINE = re.compile(
    r'^\s*\*?\s*(Facebook|Instagram|LinkedIn|Twitter|YouTube|'
    r'Follow Us on|Social Wall)\s*$',
    re.IGNORECASE | re.MULTILINE
)

# 9. Standalone icon alt-text tokens left after image removal
_RE_ICON_TOKEN = re.compile(
    r'\b(pdf[\s\-]?icon|read[\s\-]?icon|arrow[\s\-]?icon|share[\s\-]?icon|'
    r'copy[\s\-]?icon|white[\s\-]?arrow|close[\s\-]?icon|'
    r'mail[\s\-]?icon|footer[\s\-]?logo)\b',
    re.IGNORECASE
)


def _apply_text_transforms(text: str) -> str:
    """
    Apply all post-filter text-level transforms in safe order.
    Each step is documented. Order matters — do not reorder.
    """
    # Step A: strip broken double-URL links first (before generic URL removal
    # overwrites them) — removes the whole [label](https://host/https://...) form
    text = _RE_BROKEN_URL.sub('', text)

    # Step B: empty markdown links → nothing  (includes [Text]() from URL stripping)
    text = _RE_MD_LINK_EMPTY.sub('', text)
    # Also catch links where the URL was already stripped, leaving [Text]()
    text = re.sub(r'\[([^\]]+)\]\(\s*\)', lambda m: m.group(1), text)

    # Step C: image markdown → nothing (decorative, no text value)
    text = _RE_IMAGE_MD.sub('', text)

    # Step D: labelled markdown links → keep label only
    #   [B.Tech CSE](url) → B.Tech CSE
    #   [Apply Now](url)  → Apply Now  (then caught by CTA line remover)
    text = _RE_MD_LINK_LABELLED.sub(lambda m: m.group(1), text)

    # Step E: footer attribution
    text = _RE_FOOTER.sub('', text)

    # Step F: CTA-only lines
    text = _RE_CTA_LINE.sub('', text)

    # Step G: social media label-only lines
    text = _RE_SOCIAL_LINE.sub('', text)

    # Step H: bare URLs (after markdown is stripped, some raw URLs survive)
    text = _RE_BARE_URL.sub('', text)

    # Step I: icon token cleanup
    text = _RE_ICON_TOKEN.sub('', text)

    # Step J: collapse runs of blank lines and strip trailing spaces
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)

    return text.strip()


def _is_junk_line(line: str, boilerplate: set) -> bool:
    s = line.strip()
    if not s:
        return False   # blank lines handled separately

    # Boilerplate
    if s in boilerplate:
        return True

    # Image-only or link-only
    if _RE_IMAGE_ONLY.match(s) or _RE_IMAGE_LINK.match(s):
        return True
    if _RE_LINK_ONLY.match(s):
        return True

    # Navigation bullets
    if _RE_NAV_BULLET.match(line) or _RE_NUM_NAV.match(line):
        return True

    # Javascript / mailto / chatbot
    if _RE_JS_MAILTO.search(s) or _RE_CHATBOT.search(s):
        return True

    # Horizontal rules / symbol-only
    if _RE_HR.match(s) or _RE_SYMBOL_ONLY.match(s):
        return True

    # Heading navigation links: ## [Text](url) or ## [![img](url)Text](url)
    # Only removes headings whose entire content IS a link — not content headings.
    if _RE_HEADING_NAV.match(s) or _RE_HEADING_IMG_NAV.match(s):
        return True

    return False


# ---------------------------------------------------------------------------
# Paragraph-level deduplication within a page
# Two paragraphs are "duplicate" if their normalized fingerprint matches.
# ---------------------------------------------------------------------------

def _fingerprint(text: str) -> str:
    """MD5 of lower-cased, whitespace-collapsed text."""
    normalized = re.sub(r'\s+', ' ', text.lower()).strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def _dedup_paragraphs(text: str) -> tuple[str, int]:
    """
    Split on double-blank-line paragraph boundaries.
    Remove exact duplicate paragraphs (by fingerprint).
    Returns (cleaned_text, n_removed).
    """
    paragraphs = re.split(r'\n{2,}', text)
    seen: set = set()
    unique: list = []
    removed = 0
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        fp = _fingerprint(stripped)
        if fp in seen:
            removed += 1
        else:
            seen.add(fp)
            unique.append(stripped)
    return '\n\n'.join(unique), removed


# ---------------------------------------------------------------------------
# Title extraction
# ---------------------------------------------------------------------------

def _extract_title(cleaned_text: str, url: str) -> str:
    """Return first H1 from cleaned text, fallback to URL slug."""
    for line in cleaned_text.splitlines():
        s = line.strip()
        if s.startswith('# ') and not s.startswith('## '):
            return s[2:].strip()
    # fallback: URL slug
    from urllib.parse import urlparse
    path = urlparse(url).path.rstrip('/')
    slug = path.split('/')[-1] if path else 'home'
    return slug.replace('-', ' ').replace('_', ' ').title() or 'GD Goenka University'


# ---------------------------------------------------------------------------
# Per-page cleaner
# ---------------------------------------------------------------------------

def clean_page(page: dict, boilerplate: set) -> dict | None:
    """
    Clean one raw page record.
    Returns a cleaned record dict, or None if the page should be dropped.
    """
    raw_md   = page.get("markdown", "")
    url      = page.get("url", "")
    crawled  = page.get("crawled_at", "")
    raw_len  = len(raw_md)

    # --- Fix malformed URLs before any line filtering ---
    # Pattern: base/<https:/real/url>  →  real/url (keep as plain URL)
    # Pattern: bare <https:/real/url>  →  https://real/url
    def _fix_url(m):
        inner = m.group(1)                         # e.g. https:/www.gdgu...
        fixed = re.sub(r'^(https?):/([^/])', r'\1://\2', inner)
        return fixed
    raw_md = _RE_MALFORMED_URL.sub(_fix_url, raw_md)

    # --- Line-level filtering ---
    lines = raw_md.splitlines()
    kept_lines = []
    for line in lines:
        if _is_junk_line(line, boilerplate):
            continue
        kept_lines.append(line)

    # --- Collapse multiple blank lines into one ---
    collapsed = []
    prev_blank = False
    for line in kept_lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        collapsed.append(line)
        prev_blank = is_blank

    intermediate = '\n'.join(collapsed).strip()

    # --- Paragraph deduplication ---
    deduped, n_para_removed = _dedup_paragraphs(intermediate)

    # --- UI artifact removal (URLs, images, CTAs, footer) ---
    deduped = _apply_text_transforms(deduped)

    # --- Final whitespace normalization ---
    # Normalize unicode spaces, tabs → single space within lines
    lines_final = []
    for line in deduped.splitlines():
        normalized = re.sub(r'[ \t]+', ' ', line).rstrip()
        lines_final.append(normalized)
    cleaned = '\n'.join(lines_final).strip()

    # --- Drop empty / near-empty pages ---
    if len(cleaned) < 100:
        return None

    title = _extract_title(cleaned, url)

    return {
        "url":             url,
        "title":           title,
        "content":         cleaned,
        "crawled_at":      crawled,
        "cleaned_at":      datetime.now(timezone.utc).isoformat(),
        "_raw_chars":      raw_len,
        "_clean_chars":    len(cleaned),
        "_paras_removed":  n_para_removed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clean raw Crawl4AI output."
    )
    parser.add_argument("--input",  default=None,
                        help="Input JSON file (default: data/raw/gdgu_crawl.json)")
    parser.add_argument("--output", default=None,
                        help="Output JSON file (default: data/processed/gdgu_clean.json)")
    args = parser.parse_args()

    input_file  = Path(args.input)  if args.input  else INPUT_FILE
    output_file = Path(args.output) if args.output else OUTPUT_FILE

    print(f"\n{'='*65}")
    print("  Phase 3 — Content Cleaning Pipeline")
    print(f"  Input  : {input_file}")
    print(f"  Output : {output_file}")
    print(f"{'='*65}\n")

    # Load raw crawl
    print("Loading raw crawl... ", end="", flush=True)
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"{len(raw_data)} pages loaded.")

    # Build boilerplate set
    print(f"Building boilerplate set (threshold={BOILERPLATE_THRESHOLD} pages)... ",
          end="", flush=True)
    boilerplate = build_boilerplate_set(raw_data)
    print(f"{len(boilerplate)} boilerplate lines identified.")

    # Clean each page
    print("Cleaning pages...\n")
    results     = []
    dropped     = []
    total_raw   = 0
    total_clean = 0
    total_paras = 0

    for i, page in enumerate(raw_data, 1):
        url = page.get("url", "")
        cleaned = clean_page(page, boilerplate)

        raw_len = len(page.get("markdown", ""))
        total_raw += raw_len

        if cleaned is None:
            dropped.append(url)
            reduction = 100.0
            print(f"  [SKIP] #{i:>3}  (empty after cleaning)  {url[:65]}")
        else:
            clean_len = cleaned["_clean_chars"]
            total_clean += clean_len
            total_paras += cleaned["_paras_removed"]
            reduction = (1 - clean_len / raw_len) * 100 if raw_len else 0
            print(f"  [OK  ] #{i:>3}  {raw_len:>7} → {clean_len:>6} chars "
                  f"({reduction:4.0f}% reduction)  {url[:55]}")
            results.append(cleaned)

    # Strip internal stats before saving
    for r in results:
        r.pop("_raw_chars",     None)
        r.pop("_clean_chars",   None)
        r.pop("_paras_removed", None)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    avg_reduction = (1 - total_clean / total_raw) * 100 if total_raw else 0
    out_size_mb   = output_file.stat().st_size / (1024 * 1024)

    print(f"\n{'='*65}")
    print(f"  CLEANING COMPLETE")
    print(f"{'='*65}")
    print(f"  Pages processed        : {len(raw_data)}")
    print(f"  Pages kept             : {len(results)}")
    print(f"  Pages removed          : {len(dropped)}")
    print(f"  Duplicate paragraphs   : {total_paras} removed")
    print(f"  Total raw chars        : {total_raw:,}")
    print(f"  Total clean chars      : {total_clean:,}")
    print(f"  Avg content reduction  : {avg_reduction:.1f}%")
    print(f"  Output file size       : {out_size_mb:.3f} MB")
    print(f"  Output path            : {output_file.resolve()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
