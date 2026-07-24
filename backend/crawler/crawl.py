"""
crawl.py — GD Goenka University website crawler
Phase 2: Discovery + extraction only. No chunking, no embeddings, no FAISS.

Usage:
    python crawler/crawl.py
    python crawler/crawl.py --max-pages 100
    python crawler/crawl.py --start-url https://www.gdgoenkauniversity.com/about-us/about-gd-goenka-university

Output:
    data/raw/gdgu_crawl.json
"""

import asyncio
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

from crawl4ai import AsyncWebCrawler

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so the script works from any cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent                          # backend/crawler/
BACKEND_DIR  = SCRIPT_DIR.parent                              # backend/
CONFIG_FILE  = SCRIPT_DIR / "config" / "crawler_config.json"
DEFAULT_OUT  = BACKEND_DIR / "data" / "raw" / "gdgu_crawl.json"


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# URL filters
# ---------------------------------------------------------------------------
def _normalise_url(url: str) -> str:
    """Strip fragment and trailing slash for dedup comparison."""
    url, _ = urldefrag(url)
    return url.rstrip("/")


def _is_allowed_url(url: str, allowed_domain: str,
                    ignore_exts: list, ignore_patterns: list) -> bool:
    """Return True if this URL should be crawled."""
    parsed = urlparse(url)

    # Must be http/https
    if parsed.scheme not in ("http", "https"):
        return False

    # Must belong to the allowed domain (subdomains included)
    if allowed_domain not in parsed.netloc:
        return False

    path_lower = parsed.path.lower()
    full_lower = url.lower()

    # Ignore by file extension
    for ext in ignore_exts:
        if path_lower.endswith(ext):
            return False

    # Ignore by URL pattern
    for pattern in ignore_patterns:
        if pattern.lower() in full_lower:
            return False

    return True


def _clean_href(href: str) -> str:
    """
    Normalise a raw href extracted from Crawl4AI markdown.

    The GD Goenka site produces links like:
        (https://www.gdgoenkauniversity.com/<https:/www.gdgoenkauniversity.com/about-us/...>)

    Steps applied in order:
      1. Strip leading/trailing whitespace.
      2. If the href contains an embedded absolute URL inside <...>, extract it.
         Pattern: anything followed by <http(s):/...>
      3. Strip any remaining leading/trailing angle brackets.
      4. Fix single-slash protocol:  https:/foo  →  https://foo
      5. Return the cleaned href (may still be relative — urljoin handles that).
    """
    href = href.strip()

    # Step 2: extract embedded absolute URL from angle-bracket wrapper
    # e.g.  "https://host/<https:/host/path>"  →  "https://host/path"
    embedded = re.search(r'<(https?:/[^>]+)>', href)
    if embedded:
        href = embedded.group(1)

    # Step 3: strip stray angle brackets
    href = href.strip('<>')

    # Step 4: fix single-slash protocol  (https:/path → https://path)
    href = re.sub(r'^(https?):/([^/])', r'\1://\2', href)

    return href


def _extract_links(markdown: str, base_url: str) -> list:
    """
    Pull all href targets from Crawl4AI markdown output.
    Markdown links look like:  [text](url)
    Also catches plain <a href="..."> that might survive as raw HTML.
    """
    raw_hrefs = []

    # Markdown link pattern: [label](url)
    for match in re.finditer(r'\[([^\]]*)\]\(([^)]+)\)', markdown):
        raw_hrefs.append(match.group(2).strip())

    # Raw HTML href pattern (fallback)
    for match in re.finditer(r'href=["\']([^"\']+)["\']', markdown, re.IGNORECASE):
        raw_hrefs.append(match.group(1).strip())

    # Clean, resolve, and normalise
    resolved = []
    for raw in raw_hrefs:
        # Skip non-navigable schemes before cleaning
        if raw.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue

        href = _clean_href(raw)

        # Skip again after cleaning (embedded href could expose these)
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue

        full = urljoin(base_url, href)
        resolved.append(_normalise_url(full))

    return resolved


# ---------------------------------------------------------------------------
# Progress printer
# ---------------------------------------------------------------------------
def _progress(crawled: int, queued: int, skipped: int,
               url: str, success: bool, elapsed: float):
    status = "OK " if success else "ERR"
    short  = url[:80] + "…" if len(url) > 81 else url
    print(f"  [{status}] #{crawled:>4}  q={queued:<4}  skip={skipped:<4}  "
          f"({elapsed:.1f}s)  {short}", flush=True)


# ---------------------------------------------------------------------------
# Core crawler
# ---------------------------------------------------------------------------
async def crawl(start_url: str, config: dict, max_pages: int, out_path: Path):
    allowed_domain   = config["allowed_domain"]
    ignore_exts      = [e.lower() for e in config["ignore_extensions"]]
    ignore_patterns  = config["ignore_url_patterns"]
    max_concurrent   = config.get("max_concurrent", 5)

    # State
    visited:   set  = set()   # normalised URLs already fetched or attempted
    queued:    list = [_normalise_url(start_url)]
    results:   list = []
    n_skipped: int  = 0
    n_crawled: int  = 0

    print(f"\n{'='*70}")
    print(f"  GD Goenka University Crawler — Phase 2")
    print(f"  Start URL : {start_url}")
    print(f"  Max pages : {max_pages}")
    print(f"  Output    : {out_path}")
    print(f"{'='*70}\n")

    session_start = time.time()

    async with AsyncWebCrawler(headless=config.get("headless", True),
                               verbose=False) as crawler:

        while queued and n_crawled < max_pages:
            # Take up to max_concurrent URLs from the front of the queue
            batch = []
            while queued and len(batch) < max_concurrent:
                url = queued.pop(0)
                if url in visited:
                    n_skipped += 1
                    continue
                if not _is_allowed_url(url, allowed_domain,
                                       ignore_exts, ignore_patterns):
                    n_skipped += 1
                    continue
                visited.add(url)
                batch.append(url)

            if not batch:
                continue

            # Crawl the batch concurrently
            tasks = [crawler.arun(url=u) for u in batch]
            t0    = time.time()
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - t0

            for url, result in zip(batch, batch_results):
                n_crawled += 1

                # Handle exceptions from asyncio.gather
                if isinstance(result, Exception):
                    _progress(n_crawled, len(queued), n_skipped,
                               url, False, elapsed)
                    continue

                success = getattr(result, "success", False)
                _progress(n_crawled, len(queued), n_skipped,
                           url, success, elapsed)

                if not success:
                    continue

                # Extract page data
                markdown = getattr(result, "markdown", "") or ""
                title    = _extract_title(markdown, url)

                page = {
                    "url":            url,
                    "title":          title,
                    "markdown":       markdown,
                    "crawled_at":     datetime.now(timezone.utc).isoformat(),
                }
                results.append(page)

                # Discover new links from this page's markdown
                new_links = _extract_links(markdown, url)
                for link in new_links:
                    if link not in visited and link not in queued:
                        queued.append(link)

                # Respect max_pages limit
                if n_crawled >= max_pages:
                    break

    total_elapsed = time.time() - session_start

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  CRAWL COMPLETE")
    print(f"{'='*70}")
    print(f"  Total pages discovered : {len(visited) + len(queued)}")
    print(f"  Total pages crawled    : {n_crawled}")
    print(f"  Successful pages saved : {len(results)}")
    print(f"  Total pages skipped    : {n_skipped}")
    print(f"  Elapsed time           : {total_elapsed:.1f}s")
    print(f"  Output file            : {out_path.resolve()}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Title extractor (from markdown h1 or URL slug fallback)
# ---------------------------------------------------------------------------
def _extract_title(markdown: str, url: str) -> str:
    """Try to get the page title from the first H1/H2 in markdown."""
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped.startswith("## "):
            return stripped[3:].strip()
    # Fallback: derive from URL path
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1] if path else "home"
    return slug.replace("-", " ").replace("_", " ").title() or "GD Goenka University"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Crawl GD Goenka University website and save pages to JSON."
    )
    parser.add_argument(
        "--start-url",
        default=None,
        help="Override the root URL from config (default: config root_url)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to crawl (default: config max_pages)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (default: data/raw/gdgu_crawl.json)"
    )
    args = parser.parse_args()

    config     = load_config()
    start_url  = args.start_url  or config["root_url"]
    max_pages  = args.max_pages  or config.get("max_pages", 500)
    out_path   = Path(args.output) if args.output else DEFAULT_OUT

    asyncio.run(crawl(start_url, config, max_pages, out_path))


if __name__ == "__main__":
    main()
