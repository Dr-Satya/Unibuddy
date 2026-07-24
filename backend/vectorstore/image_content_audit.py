"""
image_content_audit.py — Image Content Audit for gdgu_clean.json
=================================================================
Read-only. Does NOT modify any pipeline file.

Answers:
  1. How many pages contain markdown images?
  2. How many pages have image-only sections?
  3. How many pages have little text but many images?
  4. Heading immediately followed by image (heading+image pattern).
  5. Aggregate counts: total images, pages with images, dominated pages,
     avg images/page.
  6. Category-wise statistics.
  7. Top 100 most affected pages.
  8. 20 sample snippets showing heading→image pattern.
  9. Estimate of knowledge locked inside images.
  10. Final recommendation.

Outputs:
    vectorstore/image_content_audit.json
    vectorstore/image_content_report.md

Usage:
    python vectorstore/image_content_audit.py
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
INPUT_FILE  = BACKEND_DIR / "data" / "processed" / "gdgu_clean.json"
OUT_JSON    = SCRIPT_DIR  / "image_content_audit.json"
OUT_MD      = SCRIPT_DIR  / "image_content_report.md"

# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------
_RE_IMAGE    = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')  # ![alt](url)
_RE_HEADING  = re.compile(r'^#{1,6}\s+(.+)$')
_RE_PROSE    = re.compile(r'[A-Za-z]{4,}')              # proxy for real text

# ---------------------------------------------------------------------------
# Category classifier
# ---------------------------------------------------------------------------
_CAT_RULES = [
    ("admissions",  ["/admissions/", "/admission/"]),
    ("fees",        ["/fee", "/scholarship", "/hostel-transport"]),
    ("placements",  ["/placement", "/corporate-resource-centre"]),
    ("faculty",     ["/faculty", "/department/", "/school/", "/testimonial/"]),
    ("research",    ["/research", "/publication", "/conferences/"]),
    ("hostel",      ["/hostel", "/stay-on-campus"]),
    ("courses",     ["/course/", "/programmes/"]),
    ("happenings",  ["/happenings/", "/blog/"]),
    ("about",       ["/about-us/", "/governance", "/rankings",
                     "/accreditations", "/chancellor", "/vision"]),
    ("miscellaneous",["/miscellaneous/", "/examination"]),
]

def _category(url: str) -> str:
    u = url.lower()
    for cat, patterns in _CAT_RULES:
        if any(p in u for p in patterns):
            return cat
    return "other"

# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------
def _severity(img_count: int, text_len: int, img_text_ratio: float,
              heading_img_pairs: int) -> str:
    if img_count == 0:
        return "NONE"
    if (img_text_ratio > 0.01 and img_count >= 5) or heading_img_pairs >= 3:
        return "HIGH"
    if (img_text_ratio > 0.005 and img_count >= 3) or heading_img_pairs >= 1:
        return "MEDIUM"
    return "LOW"

# ---------------------------------------------------------------------------
# Per-page analysis
# ---------------------------------------------------------------------------
def analyse_page(page: dict) -> dict:
    url   = page.get("url", "")
    title = page.get("title", "")
    text  = page.get("content", "") or ""
    lines = text.splitlines()

    # Count images
    images = _RE_IMAGE.findall(text)
    n_img  = len(images)

    # Count prose words (rough)
    words      = _RE_PROSE.findall(text)
    text_len   = len(text)
    prose_words= len(words)

    # Heading immediately followed by image (within 3 lines)
    heading_img_pairs = []
    for i, line in enumerate(lines):
        hm = _RE_HEADING.match(line.strip())
        if not hm:
            continue
        heading_text = hm.group(1).strip()
        # Look at next 5 non-blank lines for an image
        checked = 0
        for j in range(i + 1, min(i + 8, len(lines))):
            nxt = lines[j].strip()
            if not nxt:
                continue
            checked += 1
            if _RE_IMAGE.search(nxt):
                # Collect text lines after the image
                after_lines = []
                for k in range(j + 1, min(j + 6, len(lines))):
                    al = lines[k].strip()
                    if al and not _RE_IMAGE.search(al) and not _RE_HEADING.match(al):
                        after_lines.append(al)
                    elif _RE_HEADING.match(al) or (al and checked > 3):
                        break
                after_text = " ".join(after_lines)
                heading_img_pairs.append({
                    "heading":    heading_text[:80],
                    "image_line": nxt[:120],
                    "text_after": after_text[:120],
                    "line_no":    i,
                })
                break
            if checked >= 3:
                break

    # Image-only sections: heading followed ONLY by images (no prose)
    image_only_sections = [
        p for p in heading_img_pairs
        if len(_RE_PROSE.findall(p["text_after"])) < 5
    ]

    # Image/text ratio: images per 1000 chars
    img_text_ratio = (n_img / text_len * 1000) if text_len > 0 else 0

    # Dominated: images > 5 and ratio > 0.005 or almost no words
    dominated = (n_img >= 5 and img_text_ratio > 0.005) or \
                (n_img > 0 and prose_words < 50)

    sev = _severity(n_img, text_len, img_text_ratio, len(heading_img_pairs))

    return {
        "url":                   url,
        "title":                 title,
        "category":              _category(url),
        "image_count":           n_img,
        "text_length":           text_len,
        "prose_words":           prose_words,
        "img_text_ratio":        round(img_text_ratio, 4),
        "heading_img_pairs":     heading_img_pairs,
        "n_heading_img_pairs":   len(heading_img_pairs),
        "image_only_sections":   image_only_sections,
        "n_image_only_sections": len(image_only_sections),
        "image_dominated":       dominated,
        "severity":              sev,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)
    print(f"Loaded {len(pages)} pages.\n")

    results = [analyse_page(p) for p in pages]

    # ── Aggregates ────────────────────────────────────────────────────────────
    pages_with_images  = [r for r in results if r["image_count"] > 0]
    dominated_pages    = [r for r in results if r["image_dominated"]]
    image_only_pages   = [r for r in results if r["n_image_only_sections"] > 0]
    heading_img_pages  = [r for r in results if r["n_heading_img_pairs"] > 0]
    total_images       = sum(r["image_count"] for r in results)
    avg_img_per_page   = round(total_images / len(results), 2) if results else 0

    severity_dist = Counter(r["severity"] for r in results if r["image_count"] > 0)

    # Category breakdown
    cat_stats: dict = defaultdict(lambda: {
        "pages": 0, "pages_with_images": 0, "total_images": 0,
        "dominated": 0, "heading_img_pairs": 0
    })
    for r in results:
        c = r["category"]
        cat_stats[c]["pages"] += 1
        if r["image_count"] > 0:
            cat_stats[c]["pages_with_images"] += 1
            cat_stats[c]["total_images"]       += r["image_count"]
        if r["image_dominated"]:
            cat_stats[c]["dominated"] += 1
        cat_stats[c]["heading_img_pairs"] += r["n_heading_img_pairs"]

    # Top 100 worst
    top100 = sorted(
        [r for r in results if r["image_count"] > 0],
        key=lambda r: (-r["n_heading_img_pairs"], -r["image_count"], r["img_text_ratio"])
    )[:100]

    # 20 sample heading→image snippets
    snippets = []
    for r in results:
        for pair in r["heading_img_pairs"]:
            snippets.append({
                "url":        r["url"],
                "heading":    pair["heading"],
                "image_line": pair["image_line"],
                "text_after": pair["text_after"],
                "severity":   r["severity"],
            })
            if len(snippets) >= 20:
                break
        if len(snippets) >= 20:
            break

    # Knowledge estimation
    total_heading_img_pairs = sum(r["n_heading_img_pairs"] for r in results)
    total_image_only_sec    = sum(r["n_image_only_sections"] for r in results)
    high_pages = [r for r in results if r["severity"] == "HIGH"]
    est_missing_pct = round(
        min(100, (total_image_only_sec + len(dominated_pages)) / max(1, len(pages)) * 100), 1
    )

    # Recommendation
    if len(dominated_pages) > 100 or total_image_only_sec > 200:
        recommendation = "OCR strongly recommended"
        rec_reason = (
            f"{len(dominated_pages)} image-dominated pages and "
            f"{total_image_only_sec} heading+image-only sections detected. "
            "Key information (flowcharts, tables, process diagrams) likely locked in images."
        )
    elif len(dominated_pages) > 20 or total_image_only_sec > 50:
        recommendation = "OCR recommended for selected pages"
        rec_reason = (
            f"{len(dominated_pages)} image-dominated pages. "
            "Targeted OCR on HIGH-severity pages would recover meaningful content."
        )
    else:
        recommendation = "OCR not needed"
        rec_reason = "Image count is low and most headings are followed by text content."

    # ── Write JSON ────────────────────────────────────────────────────────────
    audit = {
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_pages":              len(pages),
            "pages_with_images":        len(pages_with_images),
            "pages_image_dominated":    len(dominated_pages),
            "pages_with_image_only_sections": len(image_only_pages),
            "pages_heading_img_pattern":len(heading_img_pages),
            "total_images":             total_images,
            "avg_images_per_page":      avg_img_per_page,
            "severity_distribution":    dict(severity_dist),
            "total_heading_img_pairs":  total_heading_img_pairs,
            "total_image_only_sections":total_image_only_sec,
            "estimated_missing_knowledge_pct": est_missing_pct,
            "recommendation":           recommendation,
            "recommendation_reason":    rec_reason,
        },
        "category_stats": {k: dict(v) for k, v in cat_stats.items()},
        "top100_affected_pages": [
            {
                "url":             r["url"],
                "title":           r["title"],
                "category":        r["category"],
                "image_count":     r["image_count"],
                "text_length":     r["text_length"],
                "img_text_ratio":  r["img_text_ratio"],
                "heading_img_pairs": r["n_heading_img_pairs"],
                "image_only_sections": r["n_image_only_sections"],
                "severity":        r["severity"],
            }
            for r in top100
        ],
        "sample_heading_image_snippets": snippets,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    # ── Write Markdown report ─────────────────────────────────────────────────
    s = audit["summary"]
    lines_md = [
        "# Image Content Audit Report\n",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n",
        "---\n",
        "## Summary\n",
        "| Metric | Value |", "|---|---|",
        f"| Total pages analysed | {s['total_pages']} |",
        f"| Pages containing images | {s['pages_with_images']} ({round(s['pages_with_images']/s['total_pages']*100)}%) |",
        f"| Image-dominated pages | {s['pages_image_dominated']} |",
        f"| Pages with image-only sections | {s['pages_with_image_only_sections']} |",
        f"| Pages with heading→image pattern | {s['pages_heading_img_pattern']} |",
        f"| Total images | {s['total_images']} |",
        f"| Avg images per page | {s['avg_images_per_page']} |",
        f"| Total heading+image pairs | {s['total_heading_img_pairs']} |",
        f"| Total image-only sections | {s['total_image_only_sections']} |",
        f"| Estimated missing knowledge | {s['estimated_missing_knowledge_pct']}% |",
        "\n## Severity Distribution\n",
        "| Severity | Count |", "|---|---|",
    ]
    for sev, n in sorted(s["severity_distribution"].items()):
        lines_md.append(f"| {sev} | {n} |")

    lines_md.append("\n## Category-wise Statistics\n")
    lines_md.append("| Category | Pages | With Images | Total Images | Dominated | Heading+Img Pairs |")
    lines_md.append("|---|---|---|---|---|---|")
    for cat, cs in sorted(audit["category_stats"].items()):
        lines_md.append(
            f"| {cat} | {cs['pages']} | {cs['pages_with_images']} | "
            f"{cs['total_images']} | {cs['dominated']} | {cs['heading_img_pairs']} |"
        )

    lines_md.append("\n## Top 50 Most Affected Pages\n")
    lines_md.append("| # | URL | Category | Images | Text Len | Ratio | Severity |")
    lines_md.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(top100[:50], 1):
        lines_md.append(
            f"| {i} | {r['url'][:70]} | {r['category']} | {r['image_count']} | "
            f"{r['text_length']} | {r['img_text_ratio']:.3f} | **{r['severity']}** |"
        )

    lines_md.append("\n## 20 Sample Heading → Image Snippets\n")
    for i, sn in enumerate(snippets[:20], 1):
        lines_md.append(f"\n### Snippet {i} — {sn['severity']}")
        lines_md.append(f"**URL:** {sn['url']}")
        lines_md.append(f"```")
        lines_md.append(f"#### {sn['heading']}")
        lines_md.append(f"{sn['image_line'][:100]}")
        if sn['text_after']:
            lines_md.append(f"{sn['text_after'][:100]}")
        else:
            lines_md.append(f"(no text follows)")
        lines_md.append(f"```")

    lines_md.append(f"\n## Knowledge Gap Estimate\n")
    lines_md.append(
        f"Approximately **{s['estimated_missing_knowledge_pct']}%** of pages have content "
        f"that may exist only inside images (image-dominated or image-only sections). "
        f"This represents {s['total_image_only_sections']} heading+image-only sections "
        f"across {s['pages_with_image_only_sections']} pages."
    )

    lines_md.append(f"\n## Recommendation\n")
    lines_md.append(f"### {s['recommendation']}\n")
    lines_md.append(s['recommendation_reason'])

    OUT_MD.write_text("\n".join(lines_md), encoding="utf-8")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  IMAGE CONTENT AUDIT COMPLETE")
    print(f"{'='*60}")
    print(f"  Total pages           : {s['total_pages']}")
    print(f"  Pages with images     : {s['pages_with_images']}")
    print(f"  Image-dominated pages : {s['pages_image_dominated']}")
    print(f"  Heading→image pairs   : {s['total_heading_img_pairs']}")
    print(f"  Image-only sections   : {s['total_image_only_sections']}")
    print(f"  Total images          : {s['total_images']}")
    print(f"  Severity HIGH pages   : {s['severity_distribution'].get('HIGH',0)}")
    print(f"  Est. missing knowledge: {s['estimated_missing_knowledge_pct']}%")
    print(f"\n  RECOMMENDATION: {s['recommendation']}")
    print(f"  {s['recommendation_reason'][:120]}")
    print(f"\n  Output: {OUT_JSON}")
    print(f"  Output: {OUT_MD}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
