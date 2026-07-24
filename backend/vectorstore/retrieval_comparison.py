"""
retrieval_comparison.py — Old KB vs New KB retrieval audit
===========================================================
Read-only. Does NOT modify any production file.

Loads both indexes directly and runs 8 diagnostic queries through each,
applying the same threshold/category/dedup logic as threaded_rag.py.

Outputs:
    vectorstore/retrieval_comparison.md
    vectorstore/retrieval_comparison.html
    vectorstore/retrieval_summary.json

Usage:
    python vectorstore/retrieval_comparison.py
"""

import json
import os
import re
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent

OLD_INDEX = BACKEND_DIR / "data" / "vectordb"  / "index.faiss"
OLD_META  = BACKEND_DIR / "data" / "vectordb"  / "metadata.json"
NEW_INDEX = SCRIPT_DIR  / "gdgu.index"
NEW_META  = SCRIPT_DIR  / "retrieval_metadata.json"

OUT_MD    = SCRIPT_DIR  / "retrieval_comparison.md"
OUT_HTML  = SCRIPT_DIR  / "retrieval_comparison.html"
OUT_JSON  = SCRIPT_DIR  / "retrieval_summary.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RAW  = 50   # raw FAISS fetch before filters
TOP_K_SHOW = 10   # chunks to display in report

# ---------------------------------------------------------------------------
# Category routing (mirrors threaded_rag.py classify_query exactly)
# ---------------------------------------------------------------------------
_FACULTY_KW   = ['who is','about dr','about prof','faculty','professor','teacher',
                 'lecturer','staff','dean','hod','head of department','instructor',
                 'dr.','dr ','phd','research by','taught by','who teaches',
                 'faculty member','associate professor','assistant professor']
_FEE_KW       = ['fee','fees','tuition','cost','charges','hostel fee','transport fee',
                 'annual fee','semester fee','admission fee','how much','price',
                 'payment','scholarship','stipend','financial','rupees','lakh']
_ADMISSION_KW = ['admission','admissions','eligibility','apply','application',
                 'entrance','cutoff','merit','selection','registration','enroll',
                 'enrolment','how to join','criteria','requirement','document',
                 'last date','deadline']
_UNIVERSITY_KW= ['about university','about gdgu','about gd goenka','recognition',
                 'ranking','ugc','accreditation','naac','nirf','affiliation',
                 'campus','facilities','infrastructure','hostel','library',
                 'sports','placement','located','established','history of',
                 'vision','mission','chancellor','convocation']
_COURSE_KW    = ['btech','b.tech','mba','bca','mca','m.tech','bsc','msc',
                 'course','program','programme','curriculum','syllabus',
                 'specialization','branch','stream','degree','duration',
                 'diploma','llb','ba ','ma ']

def classify(q: str) -> str:
    ql = q.lower()
    if any(kw in ql for kw in _FACULTY_KW):    return 'faculty'
    if any(kw in ql for kw in _FEE_KW):        return 'fee'
    if any(kw in ql for kw in _ADMISSION_KW):  return 'admission'
    if any(kw in ql for kw in _UNIVERSITY_KW): return 'university_info'
    if any(kw in ql for kw in _COURSE_KW):     return 'course'
    return 'general'

# ---------------------------------------------------------------------------
# OLD KB category filter (original logic: content_type + source URL)
# ---------------------------------------------------------------------------
_OLD_CAT_TYPES = {
    'faculty':       {'faculty_profile', 'faculty'},
    'fee':           {'fee_structure'},
    'admission':     {'admission', 'courses'},
    'university_info': {'about', 'facilities', 'admission'},
    'course':        {'courses', 'faculty_profile'},
    'general':       set(),
}

def old_matches_category(meta: dict, category: str) -> bool:
    if category == 'general':
        return True
    ct  = meta.get('content_type', '')
    src = meta.get('source', '')
    allowed = _OLD_CAT_TYPES.get(category, set())
    if category == 'faculty':
        return ct in allowed and '/school-of-engineering/' in src and '/course/' not in src
    if category == 'course':
        if ct == 'faculty_profile':
            return '/course/' in src
        return ct == 'courses'
    if not allowed:
        return True
    return ct in allowed

# ---------------------------------------------------------------------------
# NEW KB category filter (current threaded_rag.py logic)
# ---------------------------------------------------------------------------
_NEW_CAT_TYPES = {
    'faculty':       {'about', 'course', 'research'},
    'fee':           {'fee', 'scholarship'},
    'admission':     {'admission', 'faq', 'course'},
    'university_info': {'about', 'facilities', 'campus', 'placement', 'research'},
    'course':        {'course', 'about'},
    'general':       set(),
}
_FACULTY_URL_SIGS = ('/school/', '/deans', '/research', '/director',
                     '/faculty', '/staff', '/people', '/academic')

def new_matches_category(meta: dict, category: str) -> bool:
    if category == 'general':
        return True
    ct  = meta.get('content_type', '')
    src = meta.get('source', '')
    allowed = _NEW_CAT_TYPES.get(category, set())
    if category == 'faculty':
        if ct in allowed:
            return True
        return any(sig in src for sig in _FACULTY_URL_SIGS)
    if not allowed:
        return True
    return ct in allowed

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
OLD_THRESHOLD_STRICT  = 1.05   # L2  (lower = better)
OLD_THRESHOLD_RELAXED = 1.10
NEW_THRESHOLD_STRICT  = float(os.environ.get('RAG_L2_THRESHOLD',         '0.45'))  # IP (higher = better)
NEW_THRESHOLD_RELAXED = float(os.environ.get('RAG_L2_THRESHOLD_RELAXED', '0.40'))

# ---------------------------------------------------------------------------
# Dedup key (mirrors threaded_rag.py exactly)
# ---------------------------------------------------------------------------
def dedup_key(meta: dict, content: str) -> tuple:
    doc_id      = meta.get('doc_id') or meta.get('document_id') or meta.get('chunk_id')
    chunk_index = meta.get('chunk_index')
    if doc_id is not None and chunk_index is not None:
        return (str(doc_id), int(chunk_index))
    norm = ' '.join(content.split()).lower()
    return ('content', hashlib.md5(norm.encode()).hexdigest())

# ---------------------------------------------------------------------------
# Retrieve from one index
# ---------------------------------------------------------------------------
def retrieve(query_vec: np.ndarray, index, meta_map: dict,
             category: str, is_old: bool, top_k: int = TOP_K_RAW) -> list:
    """
    Returns list of dicts: {rank, raw_score, norm_score, content, source,
    title, content_type, chunk_len, passed_threshold, passed_category,
    passed_dedup, final_rank}
    """
    fetch_k = min(top_k, index.ntotal)
    D, I    = index.search(query_vec, fetch_k)

    thr_strict  = OLD_THRESHOLD_STRICT  if is_old else NEW_THRESHOLD_STRICT
    thr_relaxed = OLD_THRESHOLD_RELAXED if is_old else NEW_THRESHOLD_RELAXED
    threshold   = thr_strict if category == 'faculty' else thr_relaxed
    cat_fn      = old_matches_category if is_old else new_matches_category

    results = []
    seen_dedup = set()
    final_rank = 0

    for raw_rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        if idx == -1:
            continue
        meta    = meta_map.get(str(idx), {})
        content = meta.get('content', '') or ''

        # For display: normalise score so both KBs show 0-1 cosine-like value
        if is_old:
            # L2: cosine ≈ 1 - (L2² / 2)  for unit vectors
            norm_score = round(max(0.0, 1.0 - (float(score) ** 2) / 2.0), 4)
        else:
            norm_score = round(float(score), 4)

        # threshold
        if is_old:
            passed_thr = float(score) <= threshold
        else:
            passed_thr = float(score) >= threshold

        # category
        passed_cat = cat_fn(meta, category) if passed_thr else False

        # dedup
        dk = dedup_key(meta, content)
        if dk in seen_dedup:
            passed_dedup = False
        else:
            passed_dedup = True
            if passed_thr and passed_cat:
                seen_dedup.add(dk)
                final_rank += 1

        results.append({
            'raw_rank':        raw_rank,
            'raw_score':       round(float(score), 4),
            'norm_score':      norm_score,
            'content':         content,
            'source':          meta.get('source', ''),
            'title':           meta.get('title', '') or '',
            'content_type':    meta.get('content_type', ''),
            'chunk_len':       len(content),
            'passed_threshold': passed_thr,
            'passed_category':  passed_cat,
            'passed_dedup':     passed_dedup,
            'final_rank':       final_rank if (passed_thr and passed_cat and passed_dedup) else 0,
        })

    return results

# ---------------------------------------------------------------------------
# Build context string (mirrors api_adapter.py sanitize+assemble logic)
# ---------------------------------------------------------------------------
def build_context(results: list) -> str:
    final = [r for r in results if r['final_rank'] > 0]
    final.sort(key=lambda r: r['final_rank'])
    parts = []
    for r in final[:TOP_K_SHOW]:
        title = r['title'] or r['source']
        parts.append(f"[Source: {title}]\n{r['content'][:500]}")
    return '\n\n---\n\n'.join(parts)

# ---------------------------------------------------------------------------
# Compare OLD vs NEW for one query
# ---------------------------------------------------------------------------
def compare_query(query: str, model, old_index, old_map, new_index, new_map) -> dict:
    category  = classify(query)
    q_vec     = model.encode([query], convert_to_numpy=True,
                              normalize_embeddings=True).astype(np.float32)

    old_results = retrieve(q_vec, old_index, old_map, category, is_old=True)
    new_results = retrieve(q_vec, new_index, new_map, category, is_old=False)

    old_final = [r for r in old_results if r['final_rank'] > 0]
    new_final = [r for r in new_results if r['final_rank'] > 0]

    old_context = build_context(old_results)
    new_context = build_context(new_results)

    # Ranking differences: compare source URLs
    old_sources = [r['source'] for r in old_final[:TOP_K_SHOW]]
    new_sources = [r['source'] for r in new_final[:TOP_K_SHOW]]
    old_set     = set(old_sources)
    new_set     = set(new_sources)
    missing_in_new = old_set - new_set
    added_in_new   = new_set - old_set

    # OLD threshold/category filter stats
    old_thr  = sum(1 for r in old_results if r['passed_threshold'])
    old_cat  = sum(1 for r in old_results if r['passed_category'])
    new_thr  = sum(1 for r in new_results if r['passed_threshold'])
    new_cat  = sum(1 for r in new_results if r['passed_category'])

    # Verdict
    def verdict(final_results, kb_name):
        n = len(final_results)
        if n == 0:
            return f"BROKEN — zero chunks passed all filters in {kb_name}"
        if n < 3:
            return f"WEAK — only {n} chunk(s) passed in {kb_name}"
        avg = sum(r['norm_score'] for r in final_results[:5]) / min(5, n)
        if avg >= 0.55:
            return f"GOOD — {n} chunks, avg cosine {avg:.3f} in {kb_name}"
        if avg >= 0.40:
            return f"FAIR — {n} chunks, avg cosine {avg:.3f} in {kb_name}"
        return f"POOR — {n} chunks but avg cosine only {avg:.3f} in {kb_name}"

    old_verdict = verdict(old_final, "OLD KB")
    new_verdict = verdict(new_final, "NEW KB")

    return {
        'query':          query,
        'category':       category,
        'old': {
            'chunk_count':    len(old_final),
            'passed_threshold': old_thr,
            'passed_category':  old_cat,
            'top10':          old_final[:TOP_K_SHOW],
            'context':        old_context,
            'avg_chunk_size': int(sum(r['chunk_len'] for r in old_final[:10]) / max(1,min(10,len(old_final)))),
            'total_context_len': len(old_context),
            'verdict':        old_verdict,
        },
        'new': {
            'chunk_count':    len(new_final),
            'passed_threshold': new_thr,
            'passed_category':  new_cat,
            'top10':          new_final[:TOP_K_SHOW],
            'context':        new_context,
            'avg_chunk_size': int(sum(r['chunk_len'] for r in new_final[:10]) / max(1,min(10,len(new_final)))),
            'total_context_len': len(new_context),
            'verdict':        new_verdict,
        },
        'missing_in_new':  sorted(missing_in_new),
        'added_in_new':    sorted(added_in_new),
    }

# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------
def write_md(comparisons: list, path: Path):
    lines = []
    lines.append("# Retrieval Comparison Report: Old KB vs New KB\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")

    for c in comparisons:
        q   = c['query']
        cat = c['category']
        o   = c['old']
        n   = c['new']
        lines.append(f"\n## Query: \"{q}\"  |  Category: `{cat}`\n")
        lines.append(f"| Metric | OLD KB | NEW KB |")
        lines.append(f"|---|---|---|")
        lines.append(f"| Chunks retrieved | {o['chunk_count']} | {n['chunk_count']} |")
        lines.append(f"| Passed threshold | {o['passed_threshold']} | {n['passed_threshold']} |")
        lines.append(f"| Passed category  | {o['passed_category']} | {n['passed_category']} |")
        lines.append(f"| Avg chunk size   | {o['avg_chunk_size']} chars | {n['avg_chunk_size']} chars |")
        lines.append(f"| Total context    | {o['total_context_len']} chars | {n['total_context_len']} chars |")
        lines.append(f"| Verdict | {o['verdict']} | {n['verdict']} |")

        lines.append(f"\n### OLD KB — Top 10 Chunks")
        lines.append(f"| Rank | Cosine | Content Type | Title | Source | Len |")
        lines.append(f"|---|---|---|---|---|---|")
        for r in o['top10']:
            t = (r['title'] or '—')[:35]
            s = r['source'][:50]
            lines.append(f"| {r['final_rank']} | {r['norm_score']:.4f} | {r['content_type']} | {t} | {s} | {r['chunk_len']} |")

        lines.append(f"\n### NEW KB — Top 10 Chunks")
        lines.append(f"| Rank | Cosine | Content Type | Title | Source | Len |")
        lines.append(f"|---|---|---|---|---|---|")
        for r in n['top10']:
            t = (r['title'] or '—')[:35]
            s = r['source'][:50]
            lines.append(f"| {r['final_rank']} | {r['norm_score']:.4f} | {r['content_type']} | {t} | {s} | {r['chunk_len']} |")

        if c['missing_in_new']:
            lines.append(f"\n### Sources in OLD but NOT in NEW (top-10 window)")
            for u in c['missing_in_new']:
                lines.append(f"- `{u}`")

        if c['added_in_new']:
            lines.append(f"\n### Sources in NEW but NOT in OLD (top-10 window)")
            for u in c['added_in_new']:
                lines.append(f"- `{u}`")

        lines.append(f"\n### OLD Context ({o['total_context_len']} chars)")
        lines.append("```")
        lines.append(o['context'][:1500] + ("..." if len(o['context']) > 1500 else ""))
        lines.append("```")

        lines.append(f"\n### NEW Context ({n['total_context_len']} chars)")
        lines.append("```")
        lines.append(n['context'][:1500] + ("..." if len(n['context']) > 1500 else ""))
        lines.append("```")
        lines.append("\n---")

    path.write_text("\n".join(lines), encoding="utf-8")

# ---------------------------------------------------------------------------
# HTML report writer
# ---------------------------------------------------------------------------
def write_html(comparisons: list, path: Path):
    def score_color(s):
        if s >= 0.55: return "#2ecc71"
        if s >= 0.40: return "#f39c12"
        return "#e74c3c"

    def verdict_color(v):
        if v.startswith("GOOD"):   return "#2ecc71"
        if v.startswith("FAIR"):   return "#f39c12"
        if v.startswith("BROKEN"): return "#e74c3c"
        return "#e67e22"

    def chunk_rows(top10):
        rows = ""
        for r in top10:
            c = score_color(r['norm_score'])
            t = (r['title'] or '—')[:40]
            s = r['source'][:55]
            rows += (f"<tr><td>{r['final_rank']}</td>"
                     f"<td><span style='background:{c};color:white;padding:1px 6px;"
                     f"border-radius:3px'>{r['norm_score']:.4f}</span></td>"
                     f"<td>{r['content_type']}</td>"
                     f"<td title='{r['source']}'>{t}</td>"
                     f"<td style='font-size:.8em'>{s}</td>"
                     f"<td>{r['chunk_len']}</td></tr>")
        return rows or "<tr><td colspan='6'>(no results)</td></tr>"

    sections = ""
    for c in comparisons:
        q, cat = c['query'], c['category']
        o, n   = c['old'], c['new']
        ov_col = verdict_color(o['verdict'])
        nv_col = verdict_color(n['verdict'])
        missing_html = "".join(f"<li><code>{u}</code></li>" for u in c['missing_in_new'])
        added_html   = "".join(f"<li><code>{u}</code></li>" for u in c['added_in_new'])
        old_ctx = o['context'][:1500].replace('<','&lt;').replace('>','&gt;')
        new_ctx = n['context'][:1500].replace('<','&lt;').replace('>','&gt;')

        sections += f"""
<div class="query-block">
  <h2>"{q}" <span class="cat">{cat}</span></h2>
  <div class="summary-row">
    <div class="kb-card old">
      <div class="kb-label">OLD KB</div>
      <div class="verdict" style="color:{ov_col}">{o['verdict']}</div>
      <table class="mini"><tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Chunks retrieved</td><td>{o['chunk_count']}</td></tr>
        <tr><td>Passed threshold</td><td>{o['passed_threshold']}</td></tr>
        <tr><td>Passed category</td><td>{o['passed_category']}</td></tr>
        <tr><td>Avg chunk size</td><td>{o['avg_chunk_size']} chars</td></tr>
        <tr><td>Total context</td><td>{o['total_context_len']} chars</td></tr>
      </table>
    </div>
    <div class="kb-card new">
      <div class="kb-label">NEW KB</div>
      <div class="verdict" style="color:{nv_col}">{n['verdict']}</div>
      <table class="mini"><tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Chunks retrieved</td><td>{n['chunk_count']}</td></tr>
        <tr><td>Passed threshold</td><td>{n['passed_threshold']}</td></tr>
        <tr><td>Passed category</td><td>{n['passed_category']}</td></tr>
        <tr><td>Avg chunk size</td><td>{n['avg_chunk_size']} chars</td></tr>
        <tr><td>Total context</td><td>{n['total_context_len']} chars</td></tr>
      </table>
    </div>
  </div>
  <div class="chunks-row">
    <div class="chunk-table">
      <h3>OLD KB Top 10</h3>
      <table><tr><th>#</th><th>Cosine</th><th>Type</th><th>Title</th><th>Source</th><th>Len</th></tr>
        {chunk_rows(o['top10'])}</table>
    </div>
    <div class="chunk-table">
      <h3>NEW KB Top 10</h3>
      <table><tr><th>#</th><th>Cosine</th><th>Type</th><th>Title</th><th>Source</th><th>Len</th></tr>
        {chunk_rows(n['top10'])}</table>
    </div>
  </div>
  {'<div class="diff-block"><strong>Sources missing in NEW:</strong><ul>' + missing_html + '</ul></div>' if missing_html else ''}
  {'<div class="diff-block"><strong>Sources added in NEW:</strong><ul>' + added_html + '</ul></div>' if added_html else ''}
  <div class="context-row">
    <div class="ctx-block"><h4>OLD Context</h4><pre>{old_ctx}</pre></div>
    <div class="ctx-block"><h4>NEW Context</h4><pre>{new_ctx}</pre></div>
  </div>
</div>"""

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Retrieval Comparison</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f5f6fa;color:#333}}
.header{{background:#2c3e50;color:white;padding:20px 32px}}
.header h1{{margin:0 0 4px;font-size:1.5em}}
.query-block{{background:white;margin:24px 32px;border-radius:8px;padding:24px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
h2{{margin:0 0 16px;font-size:1.2em;color:#2c3e50}}
.cat{{background:#3498db;color:white;font-size:.75em;padding:2px 8px;border-radius:10px;margin-left:8px;font-weight:normal}}
.summary-row,.chunks-row,.context-row{{display:flex;gap:16px;margin-bottom:16px}}
.kb-card{{flex:1;border-radius:6px;padding:12px 16px;border:2px solid #ecf0f1}}
.kb-card.old{{border-color:#e74c3c22}}
.kb-card.new{{border-color:#2ecc7122}}
.kb-label{{font-weight:bold;font-size:.85em;color:#888;margin-bottom:4px}}
.verdict{{font-weight:bold;font-size:.9em;margin-bottom:8px}}
.mini{{width:100%;border-collapse:collapse;font-size:.85em}}
.mini th{{background:#f8f9fa;padding:4px 8px;text-align:left}}
.mini td{{padding:4px 8px;border-bottom:1px solid #f0f0f0}}
.chunk-table{{flex:1;overflow-x:auto}}
h3{{font-size:.95em;color:#555;margin:0 0 6px}}
h4{{font-size:.85em;color:#666;margin:0 0 4px}}
table{{width:100%;border-collapse:collapse;font-size:.82em}}
th{{background:#34495e;color:white;padding:6px 8px;text-align:left}}
td{{padding:5px 8px;border-bottom:1px solid #f0f0f0}}
.diff-block{{background:#fff8e1;border-left:3px solid #f39c12;padding:8px 12px;margin-bottom:12px;font-size:.85em}}
.diff-block ul{{margin:4px 0;padding-left:20px}}
.ctx-block{{flex:1}}
pre{{background:#f8f9fa;padding:12px;border-radius:4px;font-size:.78em;overflow-x:auto;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow-y:auto}}
</style></head><body>
<div class="header"><h1>Retrieval Comparison: OLD KB vs NEW KB</h1>
<p style="margin:0;opacity:.7;font-size:.9em">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
OLD: IndexFlatL2 864 vectors &nbsp;|&nbsp; NEW: IndexFlatIP 1880 vectors</p></div>
{sections}
</body></html>"""
    path.write_text(html, encoding="utf-8")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
QUERIES = [
    "MBA fee",
    "Placement",
    "Hostel",
    "Dean of School of Management",
    "Contact",
    "B.Tech CSE",
    "Scholarship",
    "Admission process",
]

def main():
    print("\n" + "="*65)
    print("  Retrieval Comparison Tool — OLD KB vs NEW KB")
    print("="*65)

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading OLD index + metadata...")
    old_index = faiss.read_index(str(OLD_INDEX))
    with open(OLD_META, "r", encoding="utf-8") as f:
        old_map = json.load(f)

    print("Loading NEW index + metadata...")
    new_index = faiss.read_index(str(NEW_INDEX))
    with open(NEW_META, "r", encoding="utf-8") as f:
        new_map = json.load(f)

    print(f"OLD: {type(old_index).__name__} ntotal={old_index.ntotal}")
    print(f"NEW: {type(new_index).__name__} ntotal={new_index.ntotal}")
    print(f"\nRunning {len(QUERIES)} queries...\n")

    comparisons = []
    for q in QUERIES:
        print(f"  → {q!r}")
        comparisons.append(compare_query(q, model, old_index, old_map,
                                          new_index, new_map))

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_md(comparisons, OUT_MD)
    write_html(comparisons, OUT_HTML)

    # JSON summary (strip content text to keep file small)
    json_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "old_index": {"type": type(old_index).__name__, "ntotal": old_index.ntotal},
        "new_index": {"type": type(new_index).__name__, "ntotal": new_index.ntotal},
        "thresholds": {
            "old_strict": OLD_THRESHOLD_STRICT, "old_relaxed": OLD_THRESHOLD_RELAXED,
            "new_strict": NEW_THRESHOLD_STRICT, "new_relaxed": NEW_THRESHOLD_RELAXED,
        },
        "queries": [
            {
                "query":    c["query"],
                "category": c["category"],
                "old_chunks": c["old"]["chunk_count"],
                "new_chunks": c["new"]["chunk_count"],
                "old_passed_threshold": c["old"]["passed_threshold"],
                "new_passed_threshold": c["new"]["passed_threshold"],
                "old_passed_category":  c["old"]["passed_category"],
                "new_passed_category":  c["new"]["passed_category"],
                "old_avg_chunk_size":   c["old"]["avg_chunk_size"],
                "new_avg_chunk_size":   c["new"]["avg_chunk_size"],
                "old_context_len":      c["old"]["total_context_len"],
                "new_context_len":      c["new"]["total_context_len"],
                "old_verdict": c["old"]["verdict"],
                "new_verdict": c["new"]["verdict"],
                "missing_in_new": c["missing_in_new"],
                "added_in_new":   c["added_in_new"],
                "old_top5": [
                    {"rank": r["final_rank"], "cosine": r["norm_score"],
                     "type": r["content_type"], "title": r["title"][:60],
                     "source": r["source"], "len": r["chunk_len"]}
                    for r in c["old"]["top10"][:5]
                ],
                "new_top5": [
                    {"rank": r["final_rank"], "cosine": r["norm_score"],
                     "type": r["content_type"], "title": r["title"][:60],
                     "source": r["source"], "len": r["chunk_len"]}
                    for r in c["new"]["top10"][:5]
                ],
            }
            for c in comparisons
        ],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Query':<35} {'OLD':>5}  {'NEW':>5}  {'Change'}")
    print(f"  {'-'*35} {'-'*5}  {'-'*5}  {'-'*20}")
    for c in comparisons:
        o_n = c['old']['chunk_count']
        n_n = c['new']['chunk_count']
        delta = n_n - o_n
        arrow = f"{'↑' if delta>0 else '↓' if delta<0 else '='}{abs(delta)}"
        print(f"  {c['query']:<35} {o_n:>5}  {n_n:>5}  {arrow}")

    print(f"\n  Output files:")
    print(f"    {OUT_MD}")
    print(f"    {OUT_HTML}")
    print(f"    {OUT_JSON}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
