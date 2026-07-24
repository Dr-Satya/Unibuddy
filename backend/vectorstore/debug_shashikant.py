"""
Debug retrieval for: "Who is Dr. Shashikant Gupta?"
Read-only. No code changes.
"""
import os, sys
os.environ["DEBUG_RAG"] = "false"   # suppress verbose pipeline noise
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, faiss, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

QUERY     = "Who is Dr. Shashikant Gupta?"
META_FILE = Path("vectorstore/retrieval_metadata.json")
IDX_FILE  = Path("vectorstore/gdgu.index")

# ── Load ──────────────────────────────────────────────────────────────────
with open(META_FILE, encoding="utf-8") as f:
    meta = json.load(f)
index = faiss.read_index(str(IDX_FILE))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── Query variants (mirrors api_adapter.py logic) ─────────────────────────
# category = faculty → suffixes used in api_adapter.py
base      = QUERY
variants  = [
    base,
    base + " profile",
    base + " research",
    base + " assistant professor GD Goenka",
]

print(f"Query          : {QUERY}")
print(f"Category       : faculty")
print(f"\nRewritten queries:")
for i, v in enumerate(variants, 1):
    print(f"  {i}. {v}")

# ── Retrieve top 10 per variant, collect all ──────────────────────────────
seen   = {}   # url → best (score, meta_entry)
SEP    = "-" * 65

for v in variants:
    vec = model.encode([v], convert_to_numpy=True,
                       normalize_embeddings=True).astype(np.float32)
    D, I = index.search(vec, 20)
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        m    = meta.get(str(idx), {})
        url  = m.get("source", "")
        if url not in seen or score > seen[url][0]:
            seen[url] = (float(score), m, v)

# Sort by score descending, take top 10
top10 = sorted(seen.values(), key=lambda x: -x[0])[:10]

print(f"\nTop 10 retrieved chunks (across all variants):")
print(f"  {'#':<3} {'Score':>7}  {'Title':<45}  URL")
print(f"  {SEP}")
for i, (score, m, from_q) in enumerate(top10, 1):
    title = (m.get("title","") or "")[:45]
    url   = m.get("source","")[:70]
    print(f"  {i:<3} {score:>7.4f}  {title:<45}  {url}")

# ── Final chunks sent to LLM (category filter applied) ───────────────────
ALLOWED_TYPES = {"about", "course", "research"}
FACULTY_SIGS  = ("/school/", "/deans", "/research", "/director",
                 "/faculty", "/staff", "/people", "/academic")
THRESHOLD     = 0.30   # current faculty threshold in threaded_rag.py

def passes_category(m):
    ct  = m.get("content_type", "")
    src = m.get("source", "")
    if ct in ALLOWED_TYPES:
        return True
    return any(sig in src for sig in FACULTY_SIGS)

final = [(s, m) for s, m, _ in top10
         if s >= THRESHOLD and passes_category(m)]

print(f"\nFinal chunks sent to LLM ({len(final)} after threshold={THRESHOLD} + category filter):")
if not final:
    print("  *** ZERO chunks passed — LLM will say 'not found' ***")
for i, (score, m) in enumerate(final, 1):
    title   = (m.get("title","") or "")[:45]
    url     = m.get("source","")[:70]
    content = (m.get("content","") or "")[:120].replace("\n"," ")
    print(f"\n  [{i}] score={score:.4f}  {title}")
    print(f"       url     : {url}")
    print(f"       content : {content}...")
