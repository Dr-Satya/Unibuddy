"""
crawl_coverage_audit.py — GD Goenka Website Crawl Coverage Audit
=================================================================
Read-only. Does NOT modify any production file.

Phase 1: Discovers ALL internal URLs on the live website by
         crawling it with Crawl4AI (same library as the project crawler).
         Uses a high max_pages cap so we get a complete picture.

Phase 2: Reads our existing crawl output (data/raw/gdgu_crawl.json)
         and compares discovered URLs vs crawled URLs.

Phase 3: Analyses why pages were missed — max_pages cap, filter rules,
         JS navigation, etc.

Outputs:
    vectorstore/crawl_coverage_report.md
    vectorstore/crawl_coverage_report.html
    vectorstore/crawl_coverage_audit.json

Usage:
    python vectorstore/crawl_coverage_audit.py
    python vectorstore/crawl_coverage_audit.py --max-discover 2000
    python vectorstore/crawl_coverage_audit.py --skip-live    (offline: use existing crawl only)
"""

import argparse
import asyncio
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urldefrag, urlparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
BACKEND_DIR  = SCRIPT_DIR.parent
CRAWL_FILE   = BACKEND_DIR / "data" / "raw"       / "gdgu_crawl.json"
CLEAN_FILE   = BACKEND_DIR / "data" / "processed" / "gdgu_clean.json"
CONFIG_FILE  = BACKEND_DIR / "crawler" / "config" / "crawler_config.json"
OUT_MD       = SCRIPT_DIR  / "crawl_coverage_report.md"
OUT_HTML     = SCRIPT_DIR  / "crawl_coverage_report.html"
OUT_JSON     = SCRIPT_DIR  / "crawl_coverage_audit.json"

ROOT_URL     = "https://www.gdgoenkauniversity.com"
ALLOWED_DOM  = "gdgoenkauniversity.com"

# ---------------------------------------------------------------------------
# Load crawler config (to use same ignore rules as production crawler)
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"max_pages": 500, "ignore_extensions": [], "ignore_url_patterns": []}

# ---------------------------------------------------------------------------
# URL normalisation (mirrors crawler/crawl.py exactly)
# ---------------------------------------------------------------------------
def _norm(url: str) -> str:
    url, _ = urldefrag(url)
    # Strip common tracking params
    url = re.sub(r'[?&](utm_[^&]+|ref=[^&]+|fbclid=[^&]+)', '', url)
    url = re.sub(r'[?&]$', '', url)
    return url.rstrip("/")

def _is_internal(url: str) -> bool:
    try:
        return ALLOWED_DOM in urlparse(url).netloc
    except Exception:
        return False

def _is_allowed(url: str, ignore_exts: list, ignore_pats: list) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if not _is_internal(url):
        return False
    path_lower = parsed.path.lower()
    full_lower = url.lower()
    for ext in ignore_exts:
        if path_lower.endswith(ext.lower()):
            return False
    for pat in ignore_pats:
        if pat.lower() in full_lower:
            return False
    return True

# ---------------------------------------------------------------------------
# Link extractor (mirrors crawler/crawl.py _clean_href + _extract_links)
# ---------------------------------------------------------------------------
def _clean_href(href: str) -> str:
    href = href.strip()
    embedded = re.search(r'<(https?:/[^>]+)>', href)
    if embedded:
        href = embedded.group(1)
    href = href.strip('<>')
    href = re.sub(r'^(https?):/([^/])', r'\1://\2', href)
    return href

def _extract_links(markdown: str, base_url: str) -> list:
    raw = []
    for m in re.finditer(r'\[([^\]]*)\]\(([^)]+)\)', markdown):
        raw.append(m.group(2).strip())
    for m in re.finditer(r'href=["\']([^"\']+)["\']', markdown, re.IGNORECASE):
        raw.append(m.group(1).strip())
    result = []
    for r in raw:
        if r.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        href = _clean_href(r)
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        full = _norm(urljoin(base_url, href))
        result.append(full)
    return result

# ---------------------------------------------------------------------------
# Category classifier for URLs
# ---------------------------------------------------------------------------
_CATEGORY_RULES = [
    ("faculty",      ["/faculty", "/staff", "/people", "/dr-", "/professor",
                      "/lecturer", "/hod", "/director", "/school-of-engineering/",
                      "/deans", "/academic-team"]),
    ("courses",      ["/course/", "/programme", "/courses/", "/btech", "/mba",
                      "/bca", "/mca", "/phd", "/llb", "/diploma"]),
    ("admissions",   ["/admissions/", "/admission/", "/apply", "/eligibility",
                      "/fee-structure", "/hostel-transport", "/scholarship",
                      "/faqs", "/open-days", "/virtual-tour", "/international-students",
                      "/phd-announcement"]),
    ("placements",   ["/placement", "/corporate-resource-centre",
                      "/recruiter", "/employability"]),
    ("research",     ["/research", "/publication", "/innovation",
                      "/phd-scholar", "/ipr", "/funded", "/consultancy"]),
    ("schools",      ["/school/"]),
    ("hostel",       ["/hostel", "/stay-on-campus", "/accommodation"]),
    ("contact",      ["/contact", "/reach-us", "/location", "/address"]),
    ("campus_life",  ["/campus-life/", "/clubs", "/events", "/eat-on-campus",
                      "/safety-on-campus", "/culture"]),
    ("about",        ["/about-us", "/about-gd-goenka", "/chancellor",
                      "/governance", "/vision", "/mission", "/rankings",
                      "/accreditations", "/mandatory", "/organogram",
                      "/regulatory", "/deans-and-directors", "/core-values"]),
    ("internationalisation", ["/internationalisation", "/international"]),
    ("iqac_iiqa",    ["/iqac", "/iiqa", "/naac", "/nirf", "/obe-ranking",
                      "/qs-ranking", "/sustainability-ranking", "/the-impact"]),
    ("happenings",   ["/happenings", "/news", "/press", "/gallery",
                      "/conference", "/events-celebrations"]),
    ("programmes",   ["/programmes/"]),
    ("career",       ["/career"]),
    ("nss",          ["/national-service-scheme"]),
]

def _categorise(url: str) -> str:
    url_lower = url.lower()
    for cat, patterns in _CATEGORY_RULES:
        if any(p in url_lower for p in patterns):
            return cat
    return "other"

# ---------------------------------------------------------------------------
# Reason classifier — why was a URL likely missed?
# ---------------------------------------------------------------------------
def _miss_reason(url: str, config: dict, our_max: int,
                 total_discovered: int) -> str:
    """Classify why a URL was not in our existing crawl."""
    url_lower = url.lower()

    # Check ignore_url_patterns
    for pat in config.get("ignore_url_patterns", []):
        if pat.lower() in url_lower:
            return f"url_filter_pattern ({pat})"

    # Check ignore_extensions
    path = urlparse(url).path.lower()
    for ext in config.get("ignore_extensions", []):
        if path.endswith(ext.lower()):
            return f"url_filter_extension ({ext})"

    # If total discovered > max_pages, pages were cut by limit
    if total_discovered > our_max:
        return f"max_pages_cap ({our_max})"

    # Faculty profile URLs pattern — never existed in current crawl set
    if re.search(r'/school-of-[a-z]+/[a-z]', url_lower):
        return "individual_faculty_profile_url (not linked from new site nav)"

    # PDF links
    if url_lower.endswith(".pdf"):
        return "url_filter_extension (.pdf)"

    return "queue_not_reached (discovered after max_pages cap)"

# ---------------------------------------------------------------------------
# Phase 1: Live URL discovery
# ---------------------------------------------------------------------------
async def discover_live_urls(max_discover: int, config: dict) -> dict:
    """
    BFS crawl of the live site to discover all internal URLs.
    Returns {url: {"discovered_from": parent_url, "depth": int}}
    """
    from crawl4ai import AsyncWebCrawler

    ignore_exts = [e.lower() for e in config.get("ignore_extensions", [])]
    ignore_pats = config.get("ignore_url_patterns", [])

    start    = _norm(ROOT_URL)
    visited  = {}        # url → {"discovered_from": str, "depth": int}
    queue    = [(start, start, 0)]   # (url, parent, depth)
    n_crawled = 0

    print(f"  Starting live URL discovery (max={max_discover})...")
    t0 = time.time()

    async with AsyncWebCrawler(headless=True, verbose=False) as crawler:
        while queue and n_crawled < max_discover:
            batch_raw = []
            while queue and len(batch_raw) < 5:
                url, parent, depth = queue.pop(0)
                if url in visited:
                    continue
                if not _is_allowed(url, ignore_exts, ignore_pats):
                    continue
                visited[url] = {"discovered_from": parent, "depth": depth}
                batch_raw.append((url, parent, depth))

            if not batch_raw:
                continue

            tasks   = [crawler.arun(url=u) for u, _, _ in batch_raw]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (url, parent, depth), result in zip(batch_raw, results):
                n_crawled += 1
                if n_crawled % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"    Discovered {len(visited)} URLs, "
                          f"crawled {n_crawled}, queue {len(queue)}  ({elapsed:.0f}s)")

                if isinstance(result, Exception):
                    continue
                if not getattr(result, "success", False):
                    continue

                md = getattr(result, "markdown", "") or ""
                for link in _extract_links(md, url):
                    if link not in visited and not any(l == link for l, _, _ in queue):
                        queue.append((link, url, depth + 1))

    elapsed = time.time() - t0
    print(f"  Discovery complete: {len(visited)} unique URLs found in {elapsed:.0f}s")
    return visited

# ---------------------------------------------------------------------------
# Phase 1b: Offline — extract all links from existing raw crawl
# ---------------------------------------------------------------------------
def discover_from_existing_crawl(raw_pages: list, config: dict) -> dict:
    """
    When --skip-live is used, build the 'discovered' set from the links
    embedded in our already-crawled pages. This gives a lower-bound estimate.
    """
    ignore_exts = [e.lower() for e in config.get("ignore_extensions", [])]
    ignore_pats = config.get("ignore_url_patterns", [])

    visited = {}
    for page in raw_pages:
        url = _norm(page.get("url", ""))
        if url:
            visited[url] = {"discovered_from": "existing_crawl", "depth": 0}

    # Also extract all links from each page's markdown content
    for page in raw_pages:
        base = page.get("url", "")
        md   = page.get("markdown", "") or ""
        for link in _extract_links(md, base):
            if link not in visited and _is_allowed(link, ignore_exts, ignore_pats):
                visited[link] = {"discovered_from": base, "depth": 1}

    print(f"  Offline discovery: {len(visited)} URLs from existing crawl + its links")
    return visited

# ---------------------------------------------------------------------------
# Phase 2: Compare
# ---------------------------------------------------------------------------
def compare(discovered: dict, our_crawl_urls: set,
            config: dict, our_max: int) -> dict:

    disc_set   = set(discovered.keys())
    total_disc = len(disc_set)
    present    = disc_set & our_crawl_urls
    missing    = disc_set - our_crawl_urls
    extra      = our_crawl_urls - disc_set

    coverage_pct = round(len(present) / total_disc * 100, 1) if total_disc else 0

    # Group missing by category
    missing_by_cat = defaultdict(list)
    for url in sorted(missing):
        cat    = _categorise(url)
        parent = discovered[url]["discovered_from"]
        reason = _miss_reason(url, config, our_max, total_disc)
        missing_by_cat[cat].append({
            "url":    url,
            "parent": parent,
            "reason": reason,
            "depth":  discovered[url]["depth"],
        })

    # Group present by category
    present_by_cat = defaultdict(int)
    for url in present:
        present_by_cat[_categorise(url)] += 1

    # Reason summary
    reason_counts = defaultdict(int)
    for items in missing_by_cat.values():
        for item in items:
            key = item["reason"].split(" (")[0]
            reason_counts[key] += 1

    return {
        "total_discovered":    total_disc,
        "total_crawled":       len(our_crawl_urls),
        "total_present":       len(present),
        "total_missing":       len(missing),
        "total_extra":         len(extra),
        "coverage_pct":        coverage_pct,
        "missing_by_category": dict(missing_by_cat),
        "present_by_category": dict(present_by_cat),
        "extra_urls":          sorted(extra),
        "reason_summary":      dict(reason_counts),
    }

# ---------------------------------------------------------------------------
# Crawler limit analysis (static code scan)
# ---------------------------------------------------------------------------
def analyse_crawler_limits(config: dict, our_crawl_count: int) -> dict:
    crawl_py = BACKEND_DIR / "crawler" / "crawl.py"
    src = crawl_py.read_text(encoding="utf-8") if crawl_py.exists() else ""

    # Find all numeric limits in the source
    limits_found = re.findall(
        r'(max_pages|max_concurrent|page_limit|max_urls|depth_limit'
        r'|crawl_limit|url_limit|max_depth|queue_limit)\s*[=:]\s*(\d+)',
        src, re.IGNORECASE
    )
    config_limits = {
        "max_pages":      config.get("max_pages", "not set"),
        "max_concurrent": config.get("max_concurrent", "not set"),
    }

    # Did we actually hit the cap?
    hit_cap = our_crawl_count >= int(config.get("max_pages", 9999))

    return {
        "config_max_pages":     config.get("max_pages", "not set"),
        "config_max_concurrent":config.get("max_concurrent", "not set"),
        "our_crawl_count":      our_crawl_count,
        "hit_max_pages_cap":    hit_cap,
        "hardcoded_limits_in_crawl_py": limits_found,
        "ignore_extensions_count": len(config.get("ignore_extensions", [])),
        "ignore_patterns_count":   len(config.get("ignore_url_patterns", [])),
        "ignore_extensions":       config.get("ignore_extensions", []),
        "ignore_url_patterns":     config.get("ignore_url_patterns", []),
        "stop_reason": (
            "MAX_PAGES cap reached"
            if hit_cap
            else "Queue exhausted (all reachable links were crawled)"
        ),
    }

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def write_md(stats: dict, limits: dict, path: Path):
    c   = stats
    mis = c["missing_by_category"]
    pre = c["present_by_category"]
    all_cats = sorted(set(list(mis.keys()) + list(pre.keys())))

    lines = ["# Crawl Coverage Audit Report\n",
             f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
             "---\n",
             "## Summary\n",
             f"| Metric | Value |", f"|---|---|",
             f"| Total URLs discovered on website | {c['total_discovered']} |",
             f"| Total URLs in our crawl          | {c['total_crawled']} |",
             f"| URLs present in both             | {c['total_present']} |",
             f"| URLs missing from our crawl      | {c['total_missing']} |",
             f"| Extra URLs in our crawl          | {c['total_extra']} |",
             f"| **Crawl coverage**               | **{c['coverage_pct']}%** |",
             "\n## Crawler Limit Analysis\n",
             f"| Setting | Value |", f"|---|---|",
             f"| max_pages (config)      | {limits['config_max_pages']} |",
             f"| max_concurrent (config) | {limits['config_max_concurrent']} |",
             f"| Pages actually crawled  | {limits['our_crawl_count']} |",
             f"| Hit max_pages cap?      | {'**YES**' if limits['hit_max_pages_cap'] else 'No (queue exhausted)'} |",
             f"| Stop reason             | {limits['stop_reason']} |",
             f"| Ignored extensions      | {limits['ignore_extensions_count']} types |",
             f"| Ignored URL patterns    | {limits['ignore_patterns_count']} patterns |",
             "\n## Coverage by Category\n",
             f"| Category | Present | Missing | Coverage |",
             f"|---|---|---|---|"]

    for cat in all_cats:
        p = pre.get(cat, 0)
        m = len(mis.get(cat, []))
        total = p + m
        pct   = round(p / total * 100) if total else 0
        lines.append(f"| {cat} | {p} | {m} | {pct}% |")

    lines.append("\n## Why Are Pages Missing?\n")
    lines.append(f"| Reason | Count |")
    lines.append(f"|---|---|")
    for reason, count in sorted(c["reason_summary"].items(),
                                 key=lambda x: -x[1]):
        lines.append(f"| {reason} | {count} |")

    lines.append("\n## Missing Pages by Category\n")
    for cat in sorted(mis.keys()):
        items = mis[cat]
        lines.append(f"\n### {cat.upper()} — {len(items)} missing pages\n")
        lines.append(f"| URL | Reason | Depth |")
        lines.append(f"|---|---|---|")
        for item in items[:50]:
            lines.append(f"| {item['url'][:80]} | {item['reason'].split('(')[0].strip()} | {item['depth']} |")
        if len(items) > 50:
            lines.append(f"| *(+{len(items)-50} more)* | | |")

    if c["extra_urls"]:
        lines.append("\n## Extra URLs (in our crawl but not discovered live)\n")
        for u in c["extra_urls"][:20]:
            lines.append(f"- `{u}`")

    lines.append("\n## Final Conclusion\n")
    pct = c["coverage_pct"]
    if pct >= 90:
        conclusion = "The website is **substantially covered**."
    elif pct >= 60:
        conclusion = "The website is **partially covered** — significant gaps exist."
    else:
        conclusion = "The website is **poorly covered** — most pages are missing."

    lines.append(conclusion)
    lines.append(f"\n- Was the complete website crawled? **{'Yes' if pct >= 95 else 'No — ' + str(pct) + '% coverage'}**")
    lines.append(f"- Pages discovered vs crawled: **{c['total_discovered']} discovered / {c['total_crawled']} crawled**")
    lines.append(f"- Pages missing: **{c['total_missing']}**")
    lines.append(f"- Primary stop reason: **{limits['stop_reason']}**")
    miss_pct = round(c['total_missing'] / c['total_discovered'] * 100, 1) if c['total_discovered'] else 0
    lines.append(f"- Missing percentage: **{miss_pct}%**")

    if limits["hit_max_pages_cap"]:
        lines.append(
            f"\n> **Root cause:** The crawler was capped at `max_pages={limits['config_max_pages']}` "
            f"in `crawler/config/crawler_config.json`. The queue was not exhausted — "
            f"there were more pages to crawl when the limit was hit. "
            f"Increase `max_pages` to crawl the full site."
        )

    path.write_text("\n".join(lines), encoding="utf-8")

# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def write_html(stats: dict, limits: dict, path: Path):
    c   = stats
    mis = c["missing_by_category"]
    pre = c["present_by_category"]
    all_cats = sorted(set(list(mis.keys()) + list(pre.keys())))

    pct = c["coverage_pct"]
    bar_color = "#2ecc71" if pct >= 90 else "#f39c12" if pct >= 60 else "#e74c3c"

    cat_rows = ""
    for cat in all_cats:
        p = pre.get(cat, 0)
        m = len(mis.get(cat, []))
        total = p + m
        cp = round(p / total * 100) if total else 0
        bc = "#2ecc71" if cp >= 90 else "#f39c12" if cp >= 60 else "#e74c3c"
        cat_rows += (f"<tr><td>{cat}</td><td>{p}</td><td>{m}</td>"
                     f"<td><span style='background:{bc};color:white;padding:1px 6px;"
                     f"border-radius:3px'>{cp}%</span></td></tr>")

    reason_rows = ""
    for reason, count in sorted(c["reason_summary"].items(), key=lambda x: -x[1]):
        reason_rows += f"<tr><td>{reason}</td><td>{count}</td></tr>"

    miss_sections = ""
    for cat in sorted(mis.keys()):
        items = mis[cat]
        rows = ""
        for item in items[:60]:
            rows += (f"<tr><td style='font-size:.8em;word-break:break-all'>{item['url']}</td>"
                     f"<td style='font-size:.8em'>{item['parent'][:60]}</td>"
                     f"<td style='font-size:.8em'>{item['reason'].split('(')[0].strip()}</td>"
                     f"<td>{item['depth']}</td></tr>")
        if len(items) > 60:
            rows += f"<tr><td colspan='4'><em>+{len(items)-60} more…</em></td></tr>"
        miss_sections += f"""
<details><summary><strong>{cat.upper()}</strong> — {len(items)} missing</summary>
<table style='width:100%;font-size:.85em;border-collapse:collapse;margin-top:8px'>
<tr style='background:#34495e;color:white'><th>URL</th><th>Found On</th><th>Reason</th><th>Depth</th></tr>
{rows}</table></details>"""

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Crawl Coverage Audit</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f5f6fa;color:#333}}
.hdr{{background:#2c3e50;color:white;padding:20px 32px}}
.hdr h1{{margin:0 0 4px;font-size:1.5em}}
.cards{{display:flex;gap:16px;padding:20px 32px;flex-wrap:wrap}}
.card{{background:white;border-radius:8px;padding:14px 20px;box-shadow:0 1px 4px rgba(0,0,0,.08);min-width:130px}}
.card .v{{font-size:2em;font-weight:bold}}
.card .l{{font-size:.78em;color:#888;margin-top:2px}}
section{{padding:0 32px 24px}}
h2{{font-size:1.1em;border-bottom:2px solid #ecf0f1;padding-bottom:6px;color:#2c3e50}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:6px;
       overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.07);font-size:.88em}}
th{{background:#34495e;color:white;padding:8px 10px;text-align:left}}
td{{padding:6px 10px;border-bottom:1px solid #f0f0f0}}
details{{background:white;border-radius:6px;margin-bottom:8px;
         box-shadow:0 1px 3px rgba(0,0,0,.07);padding:10px 14px}}
summary{{cursor:pointer;font-size:.95em}}
.prog-wrap{{background:#ecf0f1;border-radius:20px;height:20px;margin:8px 0}}
.prog-bar{{background:{bar_color};height:20px;border-radius:20px;
           width:{pct}%;display:flex;align-items:center;justify-content:flex-end;
           padding-right:8px;color:white;font-size:.8em;font-weight:bold}}
</style></head><body>
<div class="hdr"><h1>Crawl Coverage Audit — GD Goenka University</h1>
<p style="margin:0;opacity:.7;font-size:.88em">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></div>

<div class="cards">
  <div class="card"><div class="v">{c['total_discovered']}</div><div class="l">URLs on Website</div></div>
  <div class="card"><div class="v">{c['total_crawled']}</div><div class="l">URLs We Crawled</div></div>
  <div class="card"><div class="v" style="color:#2ecc71">{c['total_present']}</div><div class="l">Present in KB</div></div>
  <div class="card"><div class="v" style="color:#e74c3c">{c['total_missing']}</div><div class="l">Missing from KB</div></div>
  <div class="card"><div class="v" style="color:{bar_color}">{pct}%</div><div class="l">Coverage</div></div>
  <div class="card"><div class="v" style="color:{'#e74c3c' if limits['hit_max_pages_cap'] else '#2ecc71'}">
    {'HIT' if limits['hit_max_pages_cap'] else 'OK'}</div>
    <div class="l">Max Pages Cap ({limits['config_max_pages']})</div></div>
</div>

<section>
<h2>Coverage Bar</h2>
<div class="prog-wrap"><div class="prog-bar">{pct}%</div></div>
<p style="font-size:.88em;color:#555">{limits['stop_reason']}</p>
</section>

<section>
<h2>Coverage by Category</h2>
<table><tr><th>Category</th><th>Present</th><th>Missing</th><th>Coverage</th></tr>
{cat_rows}</table>
</section>

<section>
<h2>Why Pages Are Missing</h2>
<table><tr><th>Reason</th><th>Count</th></tr>{reason_rows}</table>
</section>

<section>
<h2>Crawler Limits Detected</h2>
<table>
<tr><th>Setting</th><th>Value</th></tr>
<tr><td>max_pages (crawler_config.json)</td><td><strong>{limits['config_max_pages']}</strong></td></tr>
<tr><td>max_concurrent</td><td>{limits['config_max_concurrent']}</td></tr>
<tr><td>Pages actually crawled</td><td>{limits['our_crawl_count']}</td></tr>
<tr><td>Hit cap?</td><td style="color:{'red' if limits['hit_max_pages_cap'] else 'green'}">
  {'YES — crawler stopped before queue was empty' if limits['hit_max_pages_cap'] else 'No — queue exhausted naturally'}</td></tr>
<tr><td>Ignored extensions</td><td>{', '.join(limits['ignore_extensions'][:8])} …</td></tr>
<tr><td>Ignored URL patterns</td><td>{', '.join(limits['ignore_url_patterns'])}</td></tr>
</table>
</section>

<section>
<h2>Missing Pages by Category</h2>
{miss_sections}
</section>
</body></html>"""
    path.write_text(html, encoding="utf-8")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Crawl coverage audit tool")
    parser.add_argument("--max-discover", type=int, default=2000,
                        help="Max pages to fetch during live discovery (default: 2000)")
    parser.add_argument("--skip-live", action="store_true",
                        help="Skip live crawl; use only existing crawl + its links")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print("  Crawl Coverage Audit — GD Goenka University")
    print(f"{'='*65}\n")

    # Load config + existing crawl
    config = _load_config()
    with open(CRAWL_FILE, encoding="utf-8") as f:
        raw_pages = json.load(f)

    our_urls = {_norm(p["url"]) for p in raw_pages}
    print(f"  Existing crawl    : {len(our_urls)} URLs")
    print(f"  Config max_pages  : {config.get('max_pages', 'not set')}")
    print(f"  Hit cap?          : {len(our_urls) >= int(config.get('max_pages', 9999))}\n")

    # Phase 1: URL discovery
    if args.skip_live:
        print("[OFFLINE MODE] Building discovery set from existing crawl links...")
        discovered = discover_from_existing_crawl(raw_pages, config)
    else:
        print(f"[LIVE MODE] Discovering URLs on {ROOT_URL} ...")
        print(f"  max_discover = {args.max_discover}\n")
        discovered = asyncio.run(discover_live_urls(args.max_discover, config))

    # Phase 2: compare
    print("\nComparing discovered vs crawled...")
    stats  = compare(discovered, our_urls, config, int(config.get("max_pages", 500)))
    limits = analyse_crawler_limits(config, len(our_urls))

    # Write reports
    print("Writing reports...")
    write_md(stats, limits, OUT_MD)
    write_html(stats, limits, OUT_HTML)

    # JSON (strip per-item depth to keep file manageable)
    json_out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline" if args.skip_live else "live",
        "stats": {k: v for k, v in stats.items()
                  if k != "missing_by_category"},
        "missing_by_category_counts": {
            cat: len(items)
            for cat, items in stats["missing_by_category"].items()
        },
        "missing_urls": {
            cat: [{"url": i["url"], "reason": i["reason"], "parent": i["parent"]}
                  for i in items]
            for cat, items in stats["missing_by_category"].items()
        },
        "limits": limits,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    # Console summary
    print(f"\n{'='*65}")
    print(f"  AUDIT COMPLETE")
    print(f"{'='*65}")
    print(f"  URLs on website   : {stats['total_discovered']}")
    print(f"  URLs crawled      : {stats['total_crawled']}")
    print(f"  Present in KB     : {stats['total_present']}")
    print(f"  MISSING from KB   : {stats['total_missing']}")
    print(f"  Coverage          : {stats['coverage_pct']}%")
    print(f"  Stop reason       : {limits['stop_reason']}")
    print(f"\n  Missing by category:")
    for cat, items in sorted(stats["missing_by_category"].items(),
                              key=lambda x: -len(x[1])):
        print(f"    {cat:<22} {len(items):>4} missing")
    print(f"\n  Why pages were missed:")
    for reason, n in sorted(stats["reason_summary"].items(), key=lambda x: -x[1]):
        print(f"    {reason:<45} {n:>4}")
    print(f"\n  Output files:")
    print(f"    {OUT_MD}")
    print(f"    {OUT_HTML}")
    print(f"    {OUT_JSON}")
    print(f"{'='*65}\n")

    # Final verdict
    pct = stats["coverage_pct"]
    print("  VERDICT:")
    if pct >= 95:
        print("  Website is fully crawled.")
    elif limits["hit_max_pages_cap"]:
        print(f"  Website was TRUNCATED by max_pages={config.get('max_pages')} cap.")
        print(f"  Approx {stats['total_discovered']} pages exist, only {stats['total_crawled']} crawled.")
        print(f"  Fix: increase max_pages in crawler/config/crawler_config.json.")
    else:
        print(f"  Coverage is {pct}%. Some pages may be JS-only or behind auth.")
    print()

if __name__ == "__main__":
    main()
