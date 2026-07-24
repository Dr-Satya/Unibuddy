"""
crawl_missing.py
================
Crawls only the URLs listed in data/raw/missing_urls.json.
Reuses _extract_title, _is_allowed_url, and the Crawl4AI config
from crawl.py without modifying it.

Output  : data/raw/gdgu_crawl_missing.json
Format  : identical to gdgu_crawl.json
          [{ "url", "title", "markdown", "crawled_at" }, ...]

Also writes:
    data/raw/failed_urls.json
    data/raw/crawl_summary.json

Usage:
    python crawler/crawl_missing.py
    python crawler/crawl_missing.py --batch-size 3
"""

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from crawl4ai import AsyncWebCrawler

# ---------------------------------------------------------------------------
# Import shared logic from crawl.py without modifying it
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_crawl_spec = _ilu.spec_from_file_location(
    "crawl", Path(__file__).parent / "crawl.py"
)
_crawl_mod = _ilu.module_from_spec(_crawl_spec)
_crawl_spec.loader.exec_module(_crawl_mod)

load_config     = _crawl_mod.load_config       # reads crawler_config.json
_extract_title  = _crawl_mod._extract_title    # H1/H2 → title
_is_allowed_url = _crawl_mod._is_allowed_url   # domain + ext + pattern filter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
INPUT_FILE  = BACKEND_DIR / "data" / "raw" / "missing_urls.json"
OUT_CRAWL   = BACKEND_DIR / "data" / "raw" / "gdgu_crawl_missing.json"
OUT_FAILED  = BACKEND_DIR / "data" / "raw" / "failed_urls.json"
OUT_SUMMARY = BACKEND_DIR / "data" / "raw" / "crawl_summary.json"

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
async def crawl_missing(urls: list[str], config: dict, batch_size: int):
    results = []
    failed  = []
    total   = len(urls)
    n_done  = 0
    t_start = time.time()

    allowed_domain  = config["allowed_domain"]
    ignore_exts     = [e.lower() for e in config["ignore_extensions"]]
    ignore_patterns = config["ignore_url_patterns"]

    print(f"\n{'='*65}")
    print(f"  crawl_missing.py — {total} URLs to crawl")
    print(f"  Batch size : {batch_size}")
    print(f"  Output     : {OUT_CRAWL.name}")
    print(f"{'='*65}\n")

    async with AsyncWebCrawler(
        headless=config.get("headless", True),
        verbose=False
    ) as crawler:

        for batch_start in range(0, total, batch_size):
            batch = urls[batch_start: batch_start + batch_size]
            tasks = [crawler.arun(url=u) for u in batch]
            t0    = time.time()

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - t0

            for url, result in zip(batch, batch_results):
                n_done  += 1
                now_iso  = datetime.now(timezone.utc).isoformat()

                if isinstance(result, Exception):
                    print(f"  [EXC] #{n_done:>4}/{total}  {url[:70]}")
                    failed.append({"url": url, "reason": str(result),
                                   "crawled_at": now_iso})
                    continue

                success  = getattr(result, "success", False)
                markdown = getattr(result, "markdown", "") or ""
                status   = "OK " if success else "ERR"

                print(f"  [{status}] #{n_done:>4}/{total}  "
                      f"md={len(markdown):>7}  ({elapsed:.1f}s)  {url[:60]}")

                if not success or len(markdown.strip()) < 10:
                    failed.append({"url": url,
                                   "reason": "failed_or_empty",
                                   "markdown_len": len(markdown),
                                   "crawled_at": now_iso})
                    continue

                results.append({
                    "url":        url,
                    "title":      _extract_title(markdown, url),
                    "markdown":   markdown,
                    "crawled_at": now_iso,
                })

    elapsed_total = time.time() - t_start

    # Write outputs
    OUT_CRAWL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CRAWL, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(OUT_FAILED, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    summary = {
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "total_urls_attempted":  total,
        "successful":            len(results),
        "failed":                len(failed),
        "success_rate_pct":      round(len(results) / total * 100, 1) if total else 0,
        "elapsed_seconds":       round(elapsed_total, 1),
        "avg_markdown_chars":    int(sum(len(r["markdown"]) for r in results)
                                     / len(results)) if results else 0,
        "output_file":           str(OUT_CRAWL.resolve()),
        "failed_urls":           [f["url"] for f in failed],
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"{'='*65}")
    print(f"  Attempted   : {total}")
    print(f"  Successful  : {len(results)}")
    print(f"  Failed      : {len(failed)}")
    print(f"  Success %   : {summary['success_rate_pct']}%")
    print(f"  Elapsed     : {elapsed_total:.1f}s")
    print(f"  Avg md size : {summary['avg_markdown_chars']:,} chars")
    print(f"\n  {OUT_CRAWL}")
    print(f"  {OUT_FAILED}")
    print(f"  {OUT_SUMMARY}")
    print(f"{'='*65}\n")

    if failed:
        print("  Failed URLs:")
        for f_ in failed[:20]:
            print(f"    {f_['reason'][:30]}  {f_['url'][:65]}")
        if len(failed) > 20:
            print(f"    ... +{len(failed)-20} more in failed_urls.json")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Crawl missing sitemap URLs from missing_urls.json."
    )
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Concurrent requests per batch (default: 5)")
    parser.add_argument("--input", default=str(INPUT_FILE),
                        help="Path to missing_urls.json")
    args = parser.parse_args()

    config = load_config()

    with open(args.input, encoding="utf-8") as f:
        entries = json.load(f)

    urls = [e["url"] for e in entries if e.get("url", "").startswith("http")]
    # deduplicate
    seen = set(); unique = []
    for u in urls:
        if u not in seen:
            seen.add(u); unique.append(u)

    print(f"Loaded {len(unique)} URLs from {args.input}")
    if not unique:
        print("Nothing to crawl.")
        return

    asyncio.run(crawl_missing(unique, config, args.batch_size))


if __name__ == "__main__":
    main()
