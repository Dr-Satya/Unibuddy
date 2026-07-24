"""
finalize_chunks.py — Phase 5 chunk dataset cleanup
Reads  : data/processed/gdgu_chunks.json
Writes : data/processed/gdgu_chunks_final.json

Three passes in order:
  Pass 1 — Remove navigation-only chunks
           A chunk is nav-only if > 60% of its non-blank lines are
           heading-links:  ## [...](...)  or  ### [...](...)
           Exception: keep if text contains important keywords
           (fee, admission, scholarship, contact, faq).

  Pass 2 — Remove tiny chunks (< 100 chars)
           Exception: keep if text contains important keywords.

  Pass 3 — Deduplicate across pages
           For each group of chunks with identical text fingerprint,
           keep the single best copy:
             1. Prefer shortest page_url (canonical URL tends to be shortest)
             2. Tiebreak: prefer page_title that is NOT generic
                ('Programmes', 'GD Goenka University', 'Home')
           Preserve original chunk_id for the kept copy.

Usage:
    python crawler/finalize_chunks.py
"""

import hashlib
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
INPUT_FILE  = BACKEND_DIR / "data" / "processed" / "gdgu_chunks.json"
OUTPUT_FILE = BACKEND_DIR / "data" / "processed" / "gdgu_chunks_final.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMPORTANT_KEYWORDS = re.compile(
    r'\b(fee|fees|admission|admissions|scholarship|contact|faq|hostel|'
    r'transport|eligibility|apply|application|tuition|charges|deposit)\b',
    re.IGNORECASE,
)

GENERIC_TITLES = {
    "programmes", "gd goenka university", "home", "",
    "gd goenka group", "about gd goenka university",
}

# A line is a nav-link line if it matches:  ## [...](...)  or  ### [...](...) 
# including the image-link variant  ## [![img](...)Label](...)
_RE_NAV_LINE  = re.compile(
    r'^\s*#{1,4}\s+\[.+\]\([^)]+\)\s*$'      # ### [text](url)
    r'|'
    r'^\s*#{1,4}\s+\[!\[.*?\]\(.*?\).*?\]\(.*?\)\s*$'  # ## [![img](url)text](url)
)
_RE_PLAIN_LINK = re.compile(r'^\s*\[.+\]\([^)]+\)\s*$')  # bare [text](url)


def _nav_ratio(text: str) -> float:
    """Fraction of non-blank lines that are pure nav-link lines."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 1.0
    nav = sum(1 for l in lines if _RE_NAV_LINE.match(l) or _RE_PLAIN_LINK.match(l))
    return nav / len(lines)


def _is_nav_only(chunk: dict) -> bool:
    text = chunk["text"]
    if IMPORTANT_KEYWORDS.search(text):
        return False
    return _nav_ratio(text) > 0.6


def _is_tiny(chunk: dict) -> bool:
    text = chunk["text"]
    if len(text) >= 100:
        return False
    if IMPORTANT_KEYWORDS.search(text):
        return False
    return True


# ---------------------------------------------------------------------------
# Text fingerprint for dedup
# ---------------------------------------------------------------------------
def _fingerprint(text: str) -> str:
    normalized = re.sub(r'\s+', ' ', text.lower()).strip()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# URL quality score — lower is better (shorter = more canonical)
# ---------------------------------------------------------------------------
def _url_score(url: str) -> int:
    return len(url)


# ---------------------------------------------------------------------------
# Title quality — lower is better (specific title > generic title)
# ---------------------------------------------------------------------------
def _title_score(title: str) -> int:
    return 0 if title.lower().strip() in GENERIC_TITLES else 1


def _best_in_group(group: list[dict]) -> dict:
    """
    Given a list of chunks with identical text, return the best one.
    Primary: prefer non-generic title (higher title_score).
    Secondary: prefer shorter URL (lower url_score).
    """
    return max(
        group,
        key=lambda c: (_title_score(c.get("page_title", "")),
                       -_url_score(c.get("page_url", "")))
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*65}")
    print("  Phase 5 — Chunk Dataset Finalization")
    print(f"  Input  : {INPUT_FILE}")
    print(f"  Output : {OUTPUT_FILE}")
    print(f"{'='*65}\n")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    original_count = len(chunks)
    print(f"Loaded {original_count} chunks.\n")

    # ------------------------------------------------------------------
    # Pass 1: Remove navigation-only chunks
    # ------------------------------------------------------------------
    nav_removed = []
    after_pass1 = []
    for c in chunks:
        if _is_nav_only(c):
            nav_removed.append(c)
        else:
            after_pass1.append(c)

    print(f"Pass 1 — Navigation removal : {len(nav_removed):>4} removed  "
          f"→ {len(after_pass1)} remaining")

    # ------------------------------------------------------------------
    # Pass 2: Remove tiny chunks
    # ------------------------------------------------------------------
    tiny_removed = []
    after_pass2  = []
    for c in after_pass1:
        if _is_tiny(c):
            tiny_removed.append(c)
        else:
            after_pass2.append(c)

    print(f"Pass 2 — Tiny chunk removal : {len(tiny_removed):>4} removed  "
          f"→ {len(after_pass2)} remaining")

    # ------------------------------------------------------------------
    # Pass 3: Cross-page text deduplication
    # ------------------------------------------------------------------
    # Group by text fingerprint
    groups: dict[str, list] = {}
    for c in after_pass2:
        fp = _fingerprint(c["text"])
        groups.setdefault(fp, []).append(c)

    dup_removed  = 0
    after_pass3  = []
    for fp, group in groups.items():
        if len(group) == 1:
            after_pass3.append(group[0])
        else:
            best = _best_in_group(group)
            after_pass3.append(best)
            dup_removed += len(group) - 1

    print(f"Pass 3 — Duplicate removal  : {dup_removed:>4} removed  "
          f"→ {len(after_pass3)} remaining")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(after_pass3, f, ensure_ascii=False, indent=2)

    out_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    total_final = len(after_pass3)
    avg_size    = sum(len(c["text"]) for c in after_pass3) // total_final if total_final else 0
    total_removed = original_count - total_final

    print(f"\n{'='*65}")
    print(f"  FINALIZATION COMPLETE")
    print(f"{'='*65}")
    print(f"  Original chunks        : {original_count}")
    print(f"  Navigation removed     : {len(nav_removed)}")
    print(f"  Tiny chunks removed    : {len(tiny_removed)}")
    print(f"  Duplicates removed     : {dup_removed}")
    print(f"  Total removed          : {total_removed}")
    print(f"  Final chunks           : {total_final}")
    print(f"  Avg chunk size         : {avg_size} chars")
    print(f"  Output file size       : {out_size_mb:.3f} MB")
    print(f"  Output path            : {OUTPUT_FILE.resolve()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
