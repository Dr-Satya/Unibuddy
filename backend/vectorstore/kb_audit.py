"""
kb_audit.py — Knowledge Base Cleanliness Audit Tool
=====================================================
Read-only. Does NOT modify any production files.

Inputs:
    data/processed/gdgu_clean.json          (new KB — cleaned pages)
    data/processed/gdgu_chunks_final.json   (new KB — final chunks)
    data/vectordb/metadata.json             (old KB — chunk metadata)

Outputs:
    vectorstore/audit_report.md
    vectorstore/audit_report.html
    vectorstore/audit_summary.json

Scoring:
    Each page is scored 0-100 for cleanliness.
    Deductions applied per detected issue type.
    Top 50 worst pages are ranked.

Usage:
    python vectorstore/kb_audit.py
"""

import json
import re
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
BACKEND_DIR  = SCRIPT_DIR.parent
NEW_CLEAN    = BACKEND_DIR / "data" / "processed" / "gdgu_clean.json"
NEW_CHUNKS   = BACKEND_DIR / "data" / "processed" / "gdgu_chunks_final.json"
OLD_META     = BACKEND_DIR / "data" / "vectordb"  / "metadata.json"
OUT_MD       = SCRIPT_DIR / "audit_report.md"
OUT_HTML     = SCRIPT_DIR / "audit_report.html"
OUT_JSON     = SCRIPT_DIR / "audit_summary.json"

# ---------------------------------------------------------------------------
# Artifact detectors
# ---------------------------------------------------------------------------

# Markdown structural artifacts that survive cleaning
_RE_MD_HEADING    = re.compile(r'^#{1,6}\s', re.MULTILINE)
_RE_MD_LINK       = re.compile(r'\[([^\]]*)\]\([^)]+\)')
_RE_MD_IMAGE      = re.compile(r'!\[([^\]]*)\]\([^)]+\)')
_RE_MD_TABLE_ROW  = re.compile(r'^\s*\|', re.MULTILINE)
_RE_MD_TABLE_SEP  = re.compile(r'^\s*\|?[-| ]+\|[-| ]+\|?\s*$', re.MULTILINE)
_RE_MD_HR         = re.compile(r'^---+\s*$', re.MULTILINE)
_RE_MD_BULLET     = re.compile(r'^\s*[\*\-]\s+', re.MULTILINE)

# Navigation / boilerplate leakage patterns
_RE_NAV_LINK      = re.compile(r'^\s*[\*\-]\s+\[.+\]\(.+\)\s*$', re.MULTILINE)
_RE_BREADCRUMB    = re.compile(r'\d+\.\s+\[Home\]|\[Home\].+\[Programs?\]', re.IGNORECASE)
_RE_COOKIE_NOTICE = re.compile(r'cookie|privacy policy|accept all', re.IGNORECASE)
_RE_FOOTER_SIG    = re.compile(
    r'© Copyright|All Rights Reserved|Follow Us|Quick Links|'
    r'GD Goenka Group|About Group|Our Ventures|Media Room|'
    r'Chat with me|chatcdn|refreshChatBotSession',
    re.IGNORECASE
)
_RE_SOCIAL        = re.compile(r'facebook|twitter|instagram|linkedin|youtube', re.IGNORECASE)

# Malformed URL patterns (angle-bracket wrapped URLs from Crawl4AI)
_RE_MALFORMED_URL = re.compile(r'<https?:/[^>]+>')

# Duplicate paragraph detection helper
def _para_fingerprint(text: str) -> str:
    return hashlib.md5(re.sub(r'\s+', ' ', text.lower()).strip().encode()).hexdigest()

# ---------------------------------------------------------------------------
# Per-page analysis
# ---------------------------------------------------------------------------

def analyse_page(url: str, title: str, text: str, chunks: list) -> dict:
    """
    Returns a dict of all metrics and a cleanliness score 0-100.
    chunks = list of chunk texts belonging to this page.
    """
    issues   = []
    score    = 100

    length   = len(text)
    lines    = text.splitlines()
    n_lines  = len(lines)

    # ── Chunk stats ───────────────────────────────────────────────────────────
    n_chunks  = len(chunks)
    avg_chunk = int(sum(len(c) for c in chunks) / n_chunks) if n_chunks else 0

    # ── Markdown artifact counts ──────────────────────────────────────────────
    n_headings  = len(_RE_MD_HEADING.findall(text))
    n_links     = len(_RE_MD_LINK.findall(text))
    n_images    = len(_RE_MD_IMAGE.findall(text))
    n_table_rows= len(_RE_MD_TABLE_ROW.findall(text))
    n_bullets   = len(_RE_MD_BULLET.findall(text))
    n_hr        = len(_RE_MD_HR.findall(text))

    # ── Navigation leakage ────────────────────────────────────────────────────
    nav_lines    = _RE_NAV_LINK.findall(text)
    n_nav        = len(nav_lines)
    nav_ratio    = n_nav / n_lines if n_lines else 0

    breadcrumb   = bool(_RE_BREADCRUMB.search(text))
    footer_leak  = bool(_RE_FOOTER_SIG.search(text))
    cookie_notice= bool(_RE_COOKIE_NOTICE.search(text))
    social_links = bool(_RE_SOCIAL.search(text))

    # ── Malformed URLs ────────────────────────────────────────────────────────
    malformed_urls = _RE_MALFORMED_URL.findall(text)
    n_malformed    = len(malformed_urls)

    # ── Duplicate paragraphs within page ─────────────────────────────────────
    paras     = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    fps       = [_para_fingerprint(p) for p in paras]
    dup_count = len(fps) - len(set(fps))

    # ── Content density ───────────────────────────────────────────────────────
    word_count = len(re.findall(r'\b[a-zA-Z]{3,}\b', text))

    # ── Scoring deductions ────────────────────────────────────────────────────
    # Navigation leakage
    if nav_ratio > 0.4:
        deduct = min(30, int(nav_ratio * 50))
        score -= deduct
        issues.append(f"HIGH nav leakage: {n_nav} nav-link lines ({nav_ratio:.0%} of page)")
    elif nav_ratio > 0.2:
        deduct = min(15, int(nav_ratio * 30))
        score -= deduct
        issues.append(f"MODERATE nav leakage: {n_nav} nav-link lines ({nav_ratio:.0%})")

    # Malformed URLs
    if n_malformed > 10:
        score -= 20
        issues.append(f"MANY malformed URLs: {n_malformed}")
    elif n_malformed > 3:
        score -= 10
        issues.append(f"SOME malformed URLs: {n_malformed}")
    elif n_malformed > 0:
        score -= 5
        issues.append(f"FEW malformed URLs: {n_malformed}")

    # Footer / boilerplate leakage
    if footer_leak:
        score -= 15
        issues.append("FOOTER boilerplate detected")

    # Breadcrumb nav
    if breadcrumb:
        score -= 5
        issues.append("Breadcrumb navigation in content")

    # Cookie notice
    if cookie_notice:
        score -= 5
        issues.append("Cookie/privacy notice text present")

    # Duplicate paragraphs
    if dup_count > 5:
        score -= 15
        issues.append(f"MANY duplicate paragraphs: {dup_count}")
    elif dup_count > 2:
        score -= 8
        issues.append(f"SOME duplicate paragraphs: {dup_count}")
    elif dup_count > 0:
        score -= 3
        issues.append(f"FEW duplicate paragraphs: {dup_count}")

    # Very low content (likely navigation-only page)
    if word_count < 50 and length > 200:
        score -= 20
        issues.append(f"LOW content density: {word_count} words in {length} chars")

    # Too many images survived (shouldn't be there)
    if n_images > 5:
        score -= 10
        issues.append(f"IMAGE markdown tags present: {n_images}")
    elif n_images > 0:
        score -= 3
        issues.append(f"Some image tags present: {n_images}")

    # Social media / footer links
    if social_links:
        score -= 5
        issues.append("Social media links present")

    score = max(0, score)

    return {
        "url":          url,
        "title":        title,
        "score":        score,
        "length":       length,
        "word_count":   word_count,
        "n_chunks":     n_chunks,
        "avg_chunk":    avg_chunk,
        "n_headings":   n_headings,
        "n_links":      n_links,
        "n_images":     n_images,
        "n_table_rows": n_table_rows,
        "n_bullets":    n_bullets,
        "n_hr":         n_hr,
        "n_nav_lines":  n_nav,
        "nav_ratio":    round(nav_ratio, 3),
        "malformed_urls": n_malformed,
        "dup_paragraphs": dup_count,
        "footer_leak":  footer_leak,
        "breadcrumb":   breadcrumb,
        "cookie_notice": cookie_notice,
        "issues":       issues,
        "sample_issues": issues[:3],
    }


# ---------------------------------------------------------------------------
# Old KB summary
# ---------------------------------------------------------------------------

def summarise_old_kb(old_meta: dict) -> dict:
    ct_counts   = Counter(v.get("content_type","") for v in old_meta.values())
    src_counts  = Counter(v.get("source","") for v in old_meta.values())
    contents    = [v.get("content","") or "" for v in old_meta.values()]
    sizes       = [len(c) for c in contents]
    return {
        "total_chunks":    len(old_meta),
        "content_types":   dict(ct_counts.most_common()),
        "unique_sources":  len(src_counts),
        "avg_chunk_size":  int(sum(sizes)/len(sizes)) if sizes else 0,
        "min_chunk_size":  min(sizes) if sizes else 0,
        "max_chunk_size":  max(sizes) if sizes else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading new KB (gdgu_clean.json)...")
    with open(NEW_CLEAN, "r", encoding="utf-8") as f:
        new_pages = json.load(f)

    print("Loading new KB chunks (gdgu_chunks_final.json)...")
    with open(NEW_CHUNKS, "r", encoding="utf-8") as f:
        new_chunks_all = json.load(f)

    print("Loading old KB (metadata.json)...")
    with open(OLD_META, "r", encoding="utf-8") as f:
        old_meta = json.load(f)

    # Group chunks by page URL
    chunks_by_url = defaultdict(list)
    for chunk in new_chunks_all:
        chunks_by_url[chunk.get("page_url","")].append(chunk.get("text",""))

    # Analyse every new KB page
    print(f"Analysing {len(new_pages)} pages...")
    results = []
    for page in new_pages:
        url    = page.get("url",     "")
        title  = page.get("title",   "")
        text   = page.get("content", "") or ""
        chunks = chunks_by_url.get(url, [])
        result = analyse_page(url, title, text, chunks)
        results.append(result)

    results.sort(key=lambda r: r["score"])

    # Old KB summary
    old_summary = summarise_old_kb(old_meta)

    # New KB aggregate stats
    scores     = [r["score"] for r in results]
    new_summary = {
        "total_pages":      len(results),
        "total_chunks":     len(new_chunks_all),
        "avg_cleanliness":  round(sum(scores)/len(scores), 1),
        "min_score":        min(scores),
        "max_score":        max(scores),
        "pages_below_50":   sum(1 for s in scores if s < 50),
        "pages_below_70":   sum(1 for s in scores if s < 70),
        "pages_above_90":   sum(1 for s in scores if s >= 90),
        "total_malformed_urls":   sum(r["malformed_urls"] for r in results),
        "total_dup_paragraphs":   sum(r["dup_paragraphs"] for r in results),
        "total_footer_leaks":     sum(1 for r in results if r["footer_leak"]),
        "total_nav_heavy_pages":  sum(1 for r in results if r["nav_ratio"] > 0.3),
        "avg_chunk_size":   int(sum(r["avg_chunk"] for r in results)/len(results)),
    }

    top50_worst = results[:50]

    # ── Write JSON summary ────────────────────────────────────────────────────
    summary_obj = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "old_kb_summary":  old_summary,
        "new_kb_summary":  new_summary,
        "top50_worst_pages": [
            {
                "rank":    i+1,
                "score":   r["score"],
                "url":     r["url"],
                "title":   r["title"],
                "issues":  r["issues"],
                "n_chunks":r["n_chunks"],
                "avg_chunk":r["avg_chunk"],
                "malformed_urls": r["malformed_urls"],
                "nav_ratio": r["nav_ratio"],
                "dup_paragraphs": r["dup_paragraphs"],
            }
            for i, r in enumerate(top50_worst)
        ],
        "all_pages": [
            {k: v for k, v in r.items() if k != "issues"}
            for r in results
        ],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    # ── Write Markdown report ─────────────────────────────────────────────────
    md_lines = []
    md_lines.append("# Knowledge Base Cleanliness Audit Report\n")
    md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    md_lines.append("\n## Old KB Summary\n")
    md_lines.append(f"| Metric | Value |")
    md_lines.append(f"|---|---|")
    md_lines.append(f"| Total chunks | {old_summary['total_chunks']} |")
    md_lines.append(f"| Unique source URLs | {old_summary['unique_sources']} |")
    md_lines.append(f"| Avg chunk size | {old_summary['avg_chunk_size']} chars |")
    md_lines.append(f"| Content types | {', '.join(f'{k}:{v}' for k,v in old_summary['content_types'].items())} |")

    md_lines.append("\n## New KB Summary\n")
    md_lines.append(f"| Metric | Value |")
    md_lines.append(f"|---|---|")
    for k, v in new_summary.items():
        md_lines.append(f"| {k.replace('_',' ').title()} | {v} |")

    md_lines.append("\n## Score Distribution\n")
    bands = [
        ("90–100 (Excellent)", sum(1 for s in scores if s >= 90)),
        ("70–89  (Good)",      sum(1 for s in scores if 70 <= s < 90)),
        ("50–69  (Fair)",      sum(1 for s in scores if 50 <= s < 70)),
        ("0–49   (Poor)",      sum(1 for s in scores if s < 50)),
    ]
    for label, count in bands:
        pct = count / len(scores) * 100
        bar = "█" * (count // 3)
        md_lines.append(f"- **{label}**: {count} pages ({pct:.0f}%)  {bar}")

    md_lines.append("\n## Top 50 Worst Pages\n")
    md_lines.append("| Rank | Score | Title | Issues |")
    md_lines.append("|---|---|---|---|")
    for i, r in enumerate(top50_worst, 1):
        title   = (r["title"] or r["url"].split("/")[-1] or "—")[:40]
        issues  = "; ".join(r["sample_issues"][:2]) or "—"
        md_lines.append(f"| {i} | {r['score']} | {title} | {issues} |")

    md_lines.append("\n## Detailed Page Analysis (Top 50 Worst)\n")
    for i, r in enumerate(top50_worst, 1):
        md_lines.append(f"\n### {i}. Score {r['score']}/100 — {r['title'] or '(no title)'}")
        md_lines.append(f"**URL:** {r['url']}")
        md_lines.append(f"**Metrics:** {r['n_chunks']} chunks | avg {r['avg_chunk']} chars | "
                        f"{r['word_count']} words | {r['malformed_urls']} malformed URLs | "
                        f"nav ratio {r['nav_ratio']:.0%} | {r['dup_paragraphs']} dup paras")
        if r["issues"]:
            md_lines.append("**Issues:**")
            for issue in r["issues"]:
                md_lines.append(f"- {issue}")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # ── Write HTML report ─────────────────────────────────────────────────────
    def score_color(s):
        if s >= 90: return "#2ecc71"
        if s >= 70: return "#f39c12"
        if s >= 50: return "#e67e22"
        return "#e74c3c"

    html_rows = ""
    for i, r in enumerate(top50_worst, 1):
        color  = score_color(r["score"])
        title  = (r["title"] or "—")[:50]
        issues = "<br>".join(r["issues"][:3]) or "—"
        url_short = r["url"][:70]
        html_rows += f"""
        <tr>
            <td>{i}</td>
            <td><span style="background:{color};color:white;padding:2px 8px;
                border-radius:4px;font-weight:bold">{r['score']}</span></td>
            <td title="{r['url']}">{title}</td>
            <td>{r['n_chunks']}</td>
            <td>{r['avg_chunk']}</td>
            <td>{r['malformed_urls']}</td>
            <td>{r['nav_ratio']:.0%}</td>
            <td>{r['dup_paragraphs']}</td>
            <td style="font-size:0.85em;color:#555">{issues}</td>
        </tr>"""

    # Summary cards
    avg_score = new_summary["avg_cleanliness"]
    avg_color = score_color(avg_score)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KB Cleanliness Audit Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; background: #f5f6fa; color: #333; }}
  .header {{ background: #2c3e50; color: white; padding: 24px 32px; }}
  .header h1 {{ margin: 0 0 4px; font-size: 1.6em; }}
  .header p  {{ margin: 0; opacity: 0.7; font-size: 0.9em; }}
  .cards {{ display: flex; gap: 16px; padding: 24px 32px; flex-wrap: wrap; }}
  .card {{ background: white; border-radius: 8px; padding: 16px 24px;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); min-width: 140px; }}
  .card .val {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
  .card .lbl {{ font-size: 0.8em; color: #888; margin-top: 4px; }}
  section {{ padding: 0 32px 32px; }}
  h2 {{ font-size: 1.2em; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; color:#2c3e50; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 8px; overflow: hidden;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); font-size:0.9em; }}
  th {{ background: #34495e; color: white; padding: 10px 12px; text-align:left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #ecf0f1; vertical-align:top; }}
  tr:hover td {{ background: #fafafa; }}
  .old-table {{ max-width: 600px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Knowledge Base Cleanliness Audit</h1>
  <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
     New KB: {new_summary['total_pages']} pages / {new_summary['total_chunks']} chunks</p>
</div>

<div class="cards">
  <div class="card">
    <div class="val" style="color:{avg_color}">{avg_score}</div>
    <div class="lbl">Avg Cleanliness Score</div>
  </div>
  <div class="card">
    <div class="val">{new_summary['pages_below_50']}</div>
    <div class="lbl">Pages Below 50 (Poor)</div>
  </div>
  <div class="card">
    <div class="val">{new_summary['total_malformed_urls']}</div>
    <div class="lbl">Malformed URLs</div>
  </div>
  <div class="card">
    <div class="val">{new_summary['total_dup_paragraphs']}</div>
    <div class="lbl">Duplicate Paragraphs</div>
  </div>
  <div class="card">
    <div class="val">{new_summary['total_footer_leaks']}</div>
    <div class="lbl">Footer Leak Pages</div>
  </div>
  <div class="card">
    <div class="val">{new_summary['total_nav_heavy_pages']}</div>
    <div class="lbl">Nav-Heavy Pages</div>
  </div>
  <div class="card">
    <div class="val">{new_summary['avg_chunk_size']}</div>
    <div class="lbl">Avg Chunk Size (chars)</div>
  </div>
  <div class="card">
    <div class="val">{new_summary['pages_above_90']}</div>
    <div class="lbl">Excellent Pages (90+)</div>
  </div>
</div>

<section>
  <h2>Old KB vs New KB Comparison</h2>
  <table class="old-table">
    <tr><th>Metric</th><th>Old KB</th><th>New KB</th></tr>
    <tr><td>Total chunks</td>
        <td>{old_summary['total_chunks']}</td>
        <td>{new_summary['total_chunks']}</td></tr>
    <tr><td>Unique source URLs</td>
        <td>{old_summary['unique_sources']}</td>
        <td>{new_summary['total_pages']}</td></tr>
    <tr><td>Avg chunk size</td>
        <td>{old_summary['avg_chunk_size']} chars</td>
        <td>{new_summary['avg_chunk_size']} chars</td></tr>
    <tr><td>Content types</td>
        <td>{'<br>'.join(f"{k}: {v}" for k,v in old_summary['content_types'].items())}</td>
        <td>13 URL-derived categories</td></tr>
  </table>
</section>

<section>
  <h2>Top 50 Worst Pages (by Cleanliness Score)</h2>
  <table>
    <tr>
      <th>#</th><th>Score</th><th>Title</th>
      <th>Chunks</th><th>Avg Chunk</th>
      <th>Bad URLs</th><th>Nav%</th>
      <th>Dup Paras</th><th>Top Issues</th>
    </tr>
    {html_rows}
  </table>
</section>

</body>
</html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  AUDIT COMPLETE")
    print(f"{'='*65}")
    print(f"  Pages analysed        : {len(results)}")
    print(f"  Avg cleanliness score : {new_summary['avg_cleanliness']}/100")
    print(f"  Pages below 50        : {new_summary['pages_below_50']}")
    print(f"  Pages above 90        : {new_summary['pages_above_90']}")
    print(f"  Total malformed URLs  : {new_summary['total_malformed_urls']}")
    print(f"  Total dup paragraphs  : {new_summary['total_dup_paragraphs']}")
    print(f"  Footer leaks          : {new_summary['total_footer_leaks']}")
    print(f"  Nav-heavy pages       : {new_summary['total_nav_heavy_pages']}")
    print(f"\n  Worst 10 pages:")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i:>2}. [{r['score']:>3}]  {r['url'][:65]}")
    print(f"\n  Output files:")
    print(f"    {OUT_MD}")
    print(f"    {OUT_HTML}")
    print(f"    {OUT_JSON}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
