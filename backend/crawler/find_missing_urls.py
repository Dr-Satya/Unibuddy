"""
find_missing_urls.py
Computes: sitemap_urls - crawled_urls → missing_urls.json

Sources:
    crawler/sitemap.xml           (ground truth)
    data/raw/gdgu_crawl.json      (what we have)

Output:
    data/raw/missing_urls.json    (list of {url, lastmod, priority})

Does NOT crawl anything. Does NOT modify any existing file.
"""
import json
from pathlib import Path
from urllib.parse import urldefrag
from xml.etree import ElementTree as ET

BASE        = Path(__file__).parent.parent
SITEMAP     = Path(__file__).parent / "sitemap.xml"
CRAWL_FILE  = BASE / "data" / "raw" / "gdgu_crawl.json"
OUT_FILE    = BASE / "data" / "raw" / "missing_urls.json"

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

def norm(url: str) -> str:
    url, _ = urldefrag(url)
    return url.strip().rstrip("/")

# ── Parse sitemap ─────────────────────────────────────────────────────────
tree = ET.parse(str(SITEMAP))
root = tree.getroot()
sitemap_entries = []
for url_el in root.findall("sm:url", NS):
    loc = (url_el.findtext("sm:loc", "", NS) or "").strip()
    if loc:
        sitemap_entries.append({
            "url":      norm(loc),
            "raw_url":  loc,
            "lastmod":  url_el.findtext("sm:lastmod",    "", NS),
            "priority": url_el.findtext("sm:priority",   "", NS),
        })

sitemap_urls = {e["url"] for e in sitemap_entries}

# ── Read crawled URLs ─────────────────────────────────────────────────────
with open(CRAWL_FILE, encoding="utf-8") as f:
    crawl_data = json.load(f)

crawled_urls = {norm(p["url"]) for p in crawl_data}

# ── Compute difference ────────────────────────────────────────────────────
missing_normed = sitemap_urls - crawled_urls

# Build output list in sitemap order (preserves priority ordering)
missing_entries = [
    e for e in sitemap_entries
    if e["url"] in missing_normed
]

# ── Write output ──────────────────────────────────────────────────────────
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(missing_entries, f, ensure_ascii=False, indent=2)

# ── Print summary ─────────────────────────────────────────────────────────
print(f"Total sitemap URLs  : {len(sitemap_urls)}")
print(f"Total crawled URLs  : {len(crawled_urls)}")
print(f"Total missing URLs  : {len(missing_entries)}")
print(f"Output              : {OUT_FILE.resolve()}")
print()
print("Missing URLs:")
for e in missing_entries:
    print(f"  {e['url']}")
