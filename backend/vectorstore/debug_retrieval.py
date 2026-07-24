"""
debug_retrieval.py  —  read-only retrieval pipeline debugger
============================================================
Loads the new knowledge base directly (no api_adapter, no LLM) and traces
every filter step for 5 diagnostic queries.

Does NOT import or modify any production source files.

Usage:
    python vectorstore/debug_retrieval.py
"""

import json
import os
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths  (resolved relative to this file so cwd doesn't matter)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
BACKEND_DIR  = SCRIPT_DIR.parent
INDEX_FILE   = SCRIPT_DIR / "gdgu.index"
META_FILE    = SCRIPT_DIR / "retrieval_metadata.json"

# ---------------------------------------------------------------------------
# Thresholds from threaded_rag.py (current values after migration)
# ---------------------------------------------------------------------------
IP_THRESHOLD_STRICT  = float(os.environ.get("RAG_L2_THRESHOLD",          "0.45"))
IP_THRESHOLD_RELAXED = float(os.environ.get("RAG_L2_THRESHOLD_RELAXED",  "0.40"))

# ---------------------------------------------------------------------------
# Category routing  (mirrors threaded_rag.py classify_query)
# ---------------------------------------------------------------------------
_FACULTY_KW   = ['who is','about dr','about prof','faculty','professor','teacher',
                 'lecturer','staff','dean','hod','head of department','instructor',
                 'dr.','dr ','phd','research by','publications of','taught by',
                 'who teaches','faculty member','associate professor','assistant professor']
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
                 'specialization','specialisation','branch','stream','degree',
                 'duration','semester','subjects in','what is taught',
                 'diploma','phd','doctorate','llb','ba ','ma ']

def classify(q: str) -> str:
    q = q.lower()
    if any(kw in q for kw in _FACULTY_KW):   return 'faculty'
    if any(kw in q for kw in _FEE_KW):       return 'fee'
    if any(kw in q for kw in _ADMISSION_KW): return 'admission'
    if any(kw in q for kw in _UNIVERSITY_KW):return 'university_info'
    if any(kw in q for kw in _COURSE_KW):    return 'course'
    return 'general'

# ---------------------------------------------------------------------------
# Category content-type filter  (mirrors threaded_rag.py _matches_category)
# ---------------------------------------------------------------------------
_CAT_TYPES = {
    'faculty':       {'about', 'course', 'research'},
    'fee':           {'fee', 'scholarship'},
    'admission':     {'admission', 'faq', 'course'},
    'university_info': {'about', 'facilities', 'campus', 'placement', 'research'},
    'course':        {'course', 'about'},
    'general':       set(),
}
_FACULTY_URL_SIGS = ('/school/', '/deans', '/research', '/director',
                     '/faculty', '/staff', '/people', '/academic')

def matches_category(meta: dict, category: str) -> bool:
    if category == 'general':
        return True
    ct  = meta.get('content_type', '')
    src = meta.get('source', '')
    allowed = _CAT_TYPES.get(category, set())
    if category == 'faculty':
        if ct in allowed:
            return True
        return any(sig in src for sig in _FACULTY_URL_SIGS)
    if not allowed:
        return True
    return ct in allowed

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading index...")
index = faiss.read_index(str(INDEX_FILE))

print("Loading metadata...")
with open(META_FILE, "r", encoding="utf-8") as f:
    meta_map: dict = json.load(f)

print(f"Ready. Index ntotal={index.ntotal}  Metadata keys={len(meta_map)}\n")

# ---------------------------------------------------------------------------
# Diagnostic queries
# ---------------------------------------------------------------------------
QUERIES = [
    "MBA fee",
    "Placement",
    "Contact",
    "Dean of School of Management",
    "B.Tech CSE",
]

TOP_K    = 20   # raw FAISS fetch
SEP      = "=" * 75
THIN_SEP = "-" * 75

for query in QUERIES:
    category = classify(query)
    threshold = IP_THRESHOLD_STRICT if category == 'faculty' else IP_THRESHOLD_RELAXED

    print(SEP)
    print(f"  QUERY    : {query!r}")
    print(f"  CATEGORY : {category}   (threshold >= {threshold})")
    print(SEP)

    # 1. Encode
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    print(f"  Encoded query vector norm : {float(np.linalg.norm(q_vec[0])):.4f}")
    print()

    # 2. Raw FAISS top-20
    D, I = index.search(q_vec, TOP_K)
    scores = D[0]
    indices = I[0]

    print(f"  ── RAW FAISS top-{TOP_K} (before any filter) ──")
    print(f"  {'Rank':<5} {'Score':>7}  {'Cat':<12} {'Title':<35} {'URL':<45}")
    print(f"  {'-'*4:<5} {'-'*7:>7}  {'-'*12:<12} {'-'*35:<35} {'-'*45:<45}")
    for rank, (sc, ix) in enumerate(zip(scores, indices), 1):
        if ix == -1:
            print(f"  {rank:<5} {'—':>7}  (FAISS returned -1)")
            continue
        m  = meta_map.get(str(ix), {})
        ct = m.get("content_type", "?")
        ti = (m.get("title", "") or "")[:35]
        ur = (m.get("source", "") or "")[:45]
        print(f"  {rank:<5} {sc:>7.4f}  {ct:<12} {ti:<35} {ur:<45}")

    print()

    # 3. Step-by-step filter trace
    passed_threshold   = []
    removed_threshold  = []
    passed_category    = []
    removed_category   = []

    for sc, ix in zip(scores, indices):
        if ix == -1:
            continue
        m = meta_map.get(str(ix), {})

        # threshold filter
        if sc < threshold:
            removed_threshold.append((sc, ix, m))
        else:
            passed_threshold.append((sc, ix, m))

    for sc, ix, m in passed_threshold:
        if matches_category(m, category):
            passed_category.append((sc, ix, m))
        else:
            removed_category.append((sc, ix, m))

    print(f"  ── FILTER 1: similarity threshold (score >= {threshold}) ──")
    print(f"  Passed : {len(passed_threshold)}   Removed : {len(removed_threshold)}")
    if removed_threshold:
        print(f"  Removed (scores too low):")
        for sc, ix, m in removed_threshold[:5]:
            ti = (m.get("title","") or "")[:40]
            print(f"    score={sc:.4f}  [{m.get('content_type','?')}]  {ti}")
    print()

    print(f"  ── FILTER 2: category filter (category={category!r}) ──")
    print(f"  Passed : {len(passed_category)}   Removed : {len(removed_category)}")
    if removed_category:
        print(f"  Removed (wrong content_type):")
        for sc, ix, m in removed_category[:5]:
            ti = (m.get("title","") or "")[:40]
            print(f"    score={sc:.4f}  [{m.get('content_type','?')}]  {ti}")
    print()

    # 4. Deduplication (content hash)
    seen_keys = set()
    deduped   = []
    removed_dedup = []
    for sc, ix, m in passed_category:
        content = m.get("content", "") or ""
        key = ("content", " ".join(content.split()).lower()[:200])
        if key in seen_keys:
            removed_dedup.append((sc, ix, m))
        else:
            seen_keys.add(key)
            deduped.append((sc, ix, m))

    print(f"  ── FILTER 3: deduplication ──")
    print(f"  Passed : {len(deduped)}   Removed : {len(removed_dedup)}")
    print()

    # 5. Final context sent to LLM
    print(f"  ── FINAL CONTEXT (top-{min(len(deduped),5)} chunks) ──")
    if not deduped:
        print("  *** NO CHUNKS PASSED ALL FILTERS — LLM will say 'not found' ***")
    for rank, (sc, ix, m) in enumerate(deduped[:5], 1):
        title   = m.get("title","")   or "(no title)"
        url     = m.get("source","")  or "(no url)"
        ct      = m.get("content_type","")
        content = (m.get("content","") or "")[:150].replace("\n"," ")
        print(f"  [{rank}] score={sc:.4f}  [{ct}]")
        print(f"       title : {title[:60]}")
        print(f"       url   : {url[:70]}")
        print(f"       text  : {content}...")
        print()

    # 6. Root-cause diagnosis
    print(f"  ── DIAGNOSIS ──")
    if not passed_threshold:
        print(f"  ROOT CAUSE: ALL {TOP_K} results scored below threshold {threshold}.")
        print(f"  Max score was {max(scores):.4f}. Threshold is too high for this query.")
        print(f"  FIX: Lower RAG_L2_THRESHOLD_RELAXED or check embedding alignment.")
    elif not passed_category:
        print(f"  ROOT CAUSE: {len(passed_threshold)} results passed threshold but ALL")
        print(f"  were removed by category filter (category={category!r}).")
        ct_present = {meta_map.get(str(ix),{}).get('content_type','?')
                      for _, ix, _ in passed_threshold}
        print(f"  Content types that scored well: {ct_present}")
        print(f"  Allowed for {category!r}: {_CAT_TYPES.get(category,set())}")
        print(f"  FIX: Add missing content types to _CAT_TYPES[{category!r}].")
    elif not deduped:
        print(f"  ROOT CAUSE: All candidates removed by deduplication.")
    else:
        # Check if the expected content type is present in top results
        top_cts = [m.get('content_type','?') for _, _, m in deduped[:5]]
        print(f"  Top-5 content types in final context: {top_cts}")
        if len(deduped) >= 3:
            print(f"  Retrieval appears functional. If LLM says 'not found',")
            print(f"  the issue is in the LLM prompt or the chunk text quality.")
        else:
            print(f"  Only {len(deduped)} chunk(s) passed. Consider widening threshold.")

    print()

print(SEP)
print("  DEBUG COMPLETE")
print(SEP)
