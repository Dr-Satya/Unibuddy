"""
discover_urls.py — Production-grade URL discovery from sitemap(s)
=================================================================
Reads a sitemap.xml (standard urlset) or a sitemap index (sitemapindex).
Supports nested sitemap indexes (one level of recursion).
Filters to same domain only.
Removes duplicates, normalises trailing slashes.
Saves to crawler/discovered_urls.json.

Usage:
    # From a local sitemap file (already downloaded):
    python crawler/discover_urls.py --source crawler/sitemap.xml

    # From a live URL:
    python crawler/discover_urls.py --source https://www.gdgoenkauniversity.com/sitemap.xml

    # Custom output path:
    python crawler/discover_urls.py --source crawler/sitemap.xml --output crawler/discovered_urls.json

    # Filter to a specific domain (default: auto-detected from sitemap):
    python crawler/discover_urls.py --source crawler/sitemap.xml --domain gdgoenkauniversity.com
"""

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
DEFAULT_OUT  = SCRIPT_DIR / "discovered_urls.json"

# ---------------------------------------------------------------------------
# XML namespace helpers
# ---------------------------------------------------------------------------
_NS_SITEMAP  = "http://www.sitemaps.org/schemas/sitemap/0.9"
_NS_NEWS     = "http://www.google.com/schemas/sitemap-news/0.9"
_NS_IMAGE    = "http://www.google.com/schemas/sitemap-image/1.1"

def _tag(local: str) -> str:
    """Return a namespace-qualified tag for the standard sitemap NS."""
    return f"{{{_NS_SITEMAP}}}{local}"


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------
def _norm(url: str) -> str:
    """Strip fragment, normalise trailing slash, strip whitespace."""
    url = url.strip()
    # Remove fragment
    url = url.split("#")[0]
    # Remove common tracking params
    url = re.sub(r"[?&](utm_[^&]+|fbclid=[^&]+|ref=[^&]+)", "", url)
    url = re.sub(r"[?&]$", "", url)
    return url.rstrip("/")


def _is_same_domain(url: str, allowed_domain: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        # Allow exact match and subdomains
        return netloc == allowed_domain or netloc.endswith("." + allowed_domain)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fetch XML — supports both file paths and HTTP(S) URLs
# ---------------------------------------------------------------------------
def _fetch_xml(source: str) -> ET.Element:
    """
    Load XML from a local file path or a remote URL.
    Returns the root Element.
    """
    path = Path(source)
    if path.exists():
        tree = ET.parse(str(path))
        return tree.getroot()

    # Remote URL — bypass SSL verification for sites with expired/self-signed certs
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    req = urllib.request.Request(
        source,
        headers={"User-Agent": "Mozilla/5.0 (compatible; UnibudDY-sitemap-reader/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        content = resp.read()
    return ET.fromstring(content)


# ---------------------------------------------------------------------------
# Sitemap parsers
# ---------------------------------------------------------------------------
def _parse_urlset(root: ET.Element) -> list[dict]:
    """
    Parse a standard <urlset> sitemap.
    Returns list of {url, lastmod, changefreq, priority}.
    """
    entries = []
    for url_el in root.findall(_tag("url")):
        loc = url_el.findtext(_tag("loc"), "").strip()
        if not loc:
            continue
        entries.append({
            "url":        _norm(loc),
            "lastmod":    url_el.findtext(_tag("lastmod"),    ""),
            "changefreq": url_el.findtext(_tag("changefreq"), ""),
            "priority":   url_el.findtext(_tag("priority"),   ""),
        })
    return entries


def _parse_sitemapindex(root: ET.Element, allowed_domain: str) -> list[dict]:
    """
    Parse a <sitemapindex> and fetch each child sitemap.
    Returns combined list of URL entries from all children.
    """
    entries = []
    child_urls = []
    for sitemap_el in root.findall(_tag("sitemap")):
        loc = sitemap_el.findtext(_tag("loc"), "").strip()
        if loc:
            child_urls.append(loc)

    print(f"  Sitemap index: found {len(child_urls)} child sitemaps")
    for i, child_url in enumerate(child_urls, 1):
        print(f"  Fetching child sitemap {i}/{len(child_urls)}: {child_url}")
        try:
            child_root = _fetch_xml(child_url)
            child_tag  = child_root.tag.split("}")[-1] if "}" in child_root.tag else child_root.tag
            if child_tag == "urlset":
                entries.extend(_parse_urlset(child_root))
            elif child_tag == "sitemapindex":
                # One more level of recursion (rare but exists)
                entries.extend(_parse_sitemapindex(child_root, allowed_domain))
        except Exception as e:
            print(f"  WARNING: could not fetch {child_url}: {e}", file=sys.stderr)

    return entries


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------
def discover(source: str, allowed_domain: str = "", output: Path = DEFAULT_OUT) -> dict:
    """
    Discover all URLs from the sitemap at `source`.

    Returns a summary dict with keys:
        total_discovered, unique_urls, filtered_out,
        domain, source, generated_at, urls (list of entry dicts)
    """
    print(f"\n{'='*60}")
    print(f"  URL Discovery Module")
    print(f"  Source : {source}")
    print(f"{'='*60}\n")

    # Load the root XML
    print("Loading sitemap... ", end="", flush=True)
    root = _fetch_xml(source)
    print("OK")

    # Detect sitemap type
    root_local = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    print(f"Sitemap type: {root_local}")

    # Auto-detect domain from first URL if not provided
    raw_entries: list[dict] = []

    if root_local == "urlset":
        raw_entries = _parse_urlset(root)
    elif root_local == "sitemapindex":
        raw_entries = _parse_sitemapindex(root, allowed_domain)
    else:
        raise ValueError(f"Unrecognised sitemap root element: {root_local!r}")

    print(f"Raw entries parsed: {len(raw_entries)}")

    # Auto-detect domain from the first valid URL
    if not allowed_domain:
        for e in raw_entries:
            parsed = urlparse(e["url"])
            if parsed.netloc:
                # Take the registered domain (last two parts)
                parts = parsed.netloc.split(".")
                allowed_domain = ".".join(parts[-2:]) if len(parts) >= 2 else parsed.netloc
                print(f"Auto-detected domain: {allowed_domain}")
                break

    # Filter to same domain
    before_filter = len(raw_entries)
    entries = [e for e in raw_entries if _is_same_domain(e["url"], allowed_domain)]
    filtered_out = before_filter - len(entries)
    print(f"After domain filter ({allowed_domain}): {len(entries)}  (removed {filtered_out} external)")

    # Deduplicate by normalised URL — keep first occurrence (preserves lastmod order)
    seen: set[str] = set()
    unique: list[dict] = []
    for e in entries:
        if e["url"] not in seen and e["url"]:
            seen.add(e["url"])
            unique.append(e)

    dupes = len(entries) - len(unique)
    print(f"After deduplication: {len(unique)}  (removed {dupes} duplicates)")

    # Sort by priority desc, then url asc for deterministic output
    unique.sort(key=lambda e: (-float(e["priority"] or 0), e["url"]))

    # Build output
    summary = {
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "source":           source,
        "domain":           allowed_domain,
        "total_parsed":     before_filter,
        "total_discovered": len(unique),
        "duplicates_removed": dupes,
        "external_filtered":  filtered_out,
        "urls":             unique,
    }

    # Write JSON
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    size_kb = output.stat().st_size / 1024
    print(f"\n{'='*60}")
    print(f"  DISCOVERY COMPLETE")
    print(f"{'='*60}")
    print(f"  Total URLs discovered : {len(unique)}")
    print(f"  Duplicates removed    : {dupes}")
    print(f"  External filtered     : {filtered_out}")
    print(f"  Domain                : {allowed_domain}")
    print(f"  Output size           : {size_kb:.1f} KB")
    print(f"  Output path           : {output.resolve()}")
    print(f"{'='*60}\n")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Discover all URLs from a sitemap.xml or sitemap index."
    )
    parser.add_argument(
        "--source",
        default=str(SCRIPT_DIR / "sitemap.xml"),
        help="Local file path or HTTP(S) URL to the sitemap (default: crawler/sitemap.xml)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUT),
        help=f"Output JSON file path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--domain",
        default="",
        help="Allowed domain to filter URLs (default: auto-detected from sitemap)",
    )
    args = parser.parse_args()

    discover(
        source=args.source,
        allowed_domain=args.domain,
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
