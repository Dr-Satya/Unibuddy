"""
chunk.py — Phase 4 chunking pipeline
Reads  : data/processed/gdgu_clean.json
Writes : data/processed/gdgu_chunks.json

What it does:
  1. Loads the cleaned pages.
  2. Infers a category from each page URL.
  3. Splits content with RecursiveCharacterTextSplitter (1000/200).
     Headings are prepended to each chunk so context is preserved.
  4. Assigns stable chunk_id = MD5(url + chunk_text).
  5. Attaches full metadata to every chunk.
  6. Saves to data/processed/gdgu_chunks.json.

Usage:
    python crawler/chunk.py
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
INPUT_FILE  = BACKEND_DIR / "data" / "processed" / "gdgu_clean.json"
OUTPUT_FILE = BACKEND_DIR / "data" / "processed" / "gdgu_chunks.json"

# ---------------------------------------------------------------------------
# Splitter config
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    # Split on headings first, then paragraphs, then sentences, then chars
    separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
    keep_separator=True,
)

# ---------------------------------------------------------------------------
# Category inference from URL path
# ---------------------------------------------------------------------------
_CATEGORY_RULES = [
    # (url_substring,  category_label)
    ("course/",                  "course"),
    ("admissions/fee",           "fee"),
    ("admissions/scholarship",   "scholarship"),
    ("admissions/faq",           "faq"),
    ("admissions/hostel",        "facilities"),
    ("admissions/",              "admission"),
    ("corporate-resource-centre/placements", "placement"),
    ("corporate-resource-centre/",           "placement"),
    ("facilities/",              "facilities"),
    ("campus-life/",             "campus"),
    ("research-and-development/","research"),
    ("publications",             "research"),
    ("conferences",              "research"),
    ("happenings/news",          "event"),
    ("happenings/",              "event"),
    ("school/",                  "about"),
    ("programmes/",              "course"),
    ("about-us/",                "about"),
    ("internationalisation/",    "about"),
    ("iqac/",                    "about"),
    ("iiqa/",                    "about"),
    ("nirf",                     "about"),
    ("obe-ranking",              "about"),
    ("qs-ranking",               "about"),
    ("sustainability-ranking",   "about"),
    ("the-impact-ranking",       "about"),
    ("times-b-school",           "about"),
    ("contact-us",               "contact"),
    ("career",                   "about"),
    ("national-service-scheme",  "about"),
    ("examination",              "about"),
]


def infer_category(url: str) -> str:
    url_lower = url.lower()
    for fragment, label in _CATEGORY_RULES:
        if fragment in url_lower:
            return label
    return "general"


# ---------------------------------------------------------------------------
# Heading prepend helper
# Finds the nearest heading(s) above the chunk's position in the full text
# so each chunk carries its own heading context.
# ---------------------------------------------------------------------------
_RE_HEADING = re.compile(r'^#{1,4} .+', re.MULTILINE)


def _extract_preceding_headings(full_text: str, chunk_text: str) -> str:
    """
    Return the last heading that appears before the chunk in the full text.
    If the chunk already starts with a heading, return it unchanged.
    """
    # Already starts with a heading — no need to prepend
    if re.match(r'^#{1,4} ', chunk_text.strip()):
        return chunk_text

    pos = full_text.find(chunk_text[:80])  # use first 80 chars as anchor
    if pos == -1:
        return chunk_text

    # Scan backwards for the most recent heading
    preceding = full_text[:pos]
    headings  = _RE_HEADING.findall(preceding)
    if not headings:
        return chunk_text

    last_heading = headings[-1].strip()
    # Don't duplicate if chunk starts with that heading text
    if chunk_text.strip().startswith(last_heading):
        return chunk_text

    return f"{last_heading}\n\n{chunk_text}"


# ---------------------------------------------------------------------------
# Stable chunk ID
# ---------------------------------------------------------------------------
def _chunk_id(url: str, text: str) -> str:
    raw = f"{url}::{text}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*65}")
    print("  Phase 4 — Chunking Pipeline")
    print(f"  Input  : {INPUT_FILE}")
    print(f"  Output : {OUTPUT_FILE}")
    print(f"{'='*65}\n")

    # Load cleaned pages
    print("Loading cleaned pages... ", end="", flush=True)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)
    print(f"{len(pages)} pages loaded.")

    now_iso   = datetime.now(timezone.utc).isoformat()
    all_chunks: list[dict] = []
    seen_ids:  set         = set()

    print("\nChunking pages...\n")

    for i, page in enumerate(pages, 1):
        url        = page.get("url", "")
        title      = page.get("title", "")
        content    = page.get("content", "")
        crawled_at = page.get("crawled_at", "")
        category   = infer_category(url)

        if not content.strip():
            print(f"  [SKIP] #{i:>3}  (empty content)  {url[:60]}")
            continue

        # Split
        raw_chunks = splitter.split_text(content)

        # Attach heading context + build records
        page_chunk_count = 0
        for chunk_text in raw_chunks:
            # Prepend nearest heading for context
            enriched = _extract_preceding_headings(content, chunk_text)

            cid = _chunk_id(url, enriched)

            # Deduplicate by chunk_id (shouldn't happen but guard it)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            record = {
                "chunk_id":     cid,
                "page_url":     url,
                "page_title":   title,
                "category":     category,
                "source":       "crawl4ai",
                "last_crawled": crawled_at,
                "chunked_at":   now_iso,
                "text":         enriched,
            }
            all_chunks.append(record)
            page_chunk_count += 1

        print(f"  [OK  ] #{i:>3}  {len(content):>7} chars → {page_chunk_count:>3} chunks"
              f"  [{category:<11}]  {url[:50]}")

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    out_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    avg_size    = sum(len(c["text"]) for c in all_chunks) // len(all_chunks) if all_chunks else 0

    print(f"\n{'='*65}")
    print(f"  CHUNKING COMPLETE")
    print(f"{'='*65}")
    print(f"  Pages processed        : {len(pages)}")
    print(f"  Total chunks           : {len(all_chunks)}")
    print(f"  Avg chunks / page      : {len(all_chunks) / len(pages):.1f}")
    print(f"  Avg chunk size         : {avg_size} chars")
    print(f"  Output file size       : {out_size_mb:.3f} MB")
    print(f"  Output path            : {OUTPUT_FILE.resolve()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
