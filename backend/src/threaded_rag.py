import os, json, threading, hashlib, re
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, DEBUG_RAG
from src.rag_debug import warn
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, DEBUG_RAG

# IndexFlatIP threshold — minimum cosine similarity to accept a result.
# All embeddings are L2-normalised so IP score == cosine similarity.
# Values are env-var overridable; defaults tuned for the new clean KB.
#   0.45 → only chunks with cosine >= 0.45 pass (faculty: strict)
#   0.40 → relaxed floor for other categories (fee, course, admission, etc.)
_L2_THRESHOLD        = float(os.environ.get('RAG_L2_THRESHOLD',         '0.30'))
_L2_THRESHOLD_RELAXED = float(os.environ.get('RAG_L2_THRESHOLD_RELAXED', '0.25'))


def _normalize_text(text: Any) -> str:
    return re.sub(r'\s+', ' ', str(text or '').lower()).strip()


def _exact_metadata_match_score(query: str, meta: Dict[str, Any]) -> int:
    query_norm = _normalize_text(query)
    if not query_norm:
        return 0

    fields = {
        'title': _normalize_text(meta.get('title', '')),
        'source': _normalize_text(meta.get('source', '')),
        'name': _normalize_text(meta.get('name', '')),
        'designation': _normalize_text(meta.get('designation', '')),
        'department': _normalize_text(meta.get('department', '')),
        'content_type': _normalize_text(meta.get('content_type', '')),
    }

    query_terms = set(re.findall(r"\b[a-z0-9]{3,}\b", query_norm))
    if not query_terms:
        return 0

    # Strong exact match: query text appears as a substring of a key metadata field
    # or the metadata field appears in the query text.
    for field in ('title', 'source', 'name', 'designation', 'department'):
        value = fields[field]
        if not value:
            continue
        if query_norm == value:
            return 4
        if query_norm in value or value in query_norm:
            return 3

    score = 0
    for field in ('title', 'source', 'name', 'designation', 'department'):
        value = fields[field]
        if not value:
            continue
        field_terms = set(re.findall(r"\b[a-z0-9]{3,}\b", value))
        if query_terms and query_terms <= field_terms:
            return 3
        if query_terms & field_terms:
            score = max(score, 2)

    if fields['content_type'] and query_norm in fields['content_type']:
        score = max(score, 2)

    return score


# ---------------------------------------------------------------------------
# Query classifier
# ---------------------------------------------------------------------------
# Maps a free-text query to one of six category strings so that retrieval
# can be restricted to the relevant subset of the FAISS index.
# All matching is case-insensitive on the lower-cased query.

_FACULTY_KEYWORDS = [
    'who is', 'about dr', 'about prof', 'faculty', 'professor', 'teacher',
    'lecturer', 'staff', 'dean', 'hod', 'head of department', 'instructor',
    'dr.', 'dr ', 'phd', 'research by', 'publications of', 'taught by',
    'who teaches', 'faculty member', 'associate professor', 'assistant professor',
]
_FEE_KEYWORDS = [
    'fee', 'fees', 'tuition', 'cost', 'charges', 'hostel fee', 'transport fee',
    'annual fee', 'semester fee', 'admission fee', 'how much', 'price',
    'payment', 'scholarship', 'stipend', 'financial', 'rupees', 'lakh',
]
_ADMISSION_KEYWORDS = [
    'admission', 'admissions', 'eligibility', 'apply', 'application',
    'entrance', 'cutoff', 'merit', 'selection', 'registration', 'enroll',
    'enrolment', 'how to join', 'criteria', 'requirement', 'document',
    'last date', 'deadline',
]
_UNIVERSITY_KEYWORDS = [
    'about university', 'about gdgu', 'about gd goenka', 'recognition',
    'ranking', 'ugc', 'accreditation', 'naac', 'nirf', 'affiliation',
    'campus', 'facilities', 'infrastructure', 'hostel', 'library',
    'sports', 'placement', 'located', 'established', 'history of',
    'vision', 'mission', 'chancellor', 'convocation',
]
_COURSE_KEYWORDS = [
    'btech', 'b.tech', 'mba', 'bca', 'mca', 'm.tech', 'bsc', 'msc',
    'course', 'program', 'programme', 'curriculum', 'syllabus',
    'specialization', 'specialisation', 'branch', 'stream', 'degree',
    'duration', 'semester', 'subjects in', 'what is taught',
    'diploma', 'phd', 'doctorate', 'llb', 'ba ', 'ma ',
]


def classify_query(query: str) -> str:
    """
    Classify a user query into one of six retrieval categories:
      faculty | fee | admission | university_info | course | general

    Matching is done against lowercased query text using keyword lists.
    Faculty keywords are checked first because queries like
    "Dr. X fee structure" should stay in faculty category.
    Returns 'general' if no keywords match.
    """
    q = query.lower()

    # Faculty takes priority — "who is dr X" must not fall into course
    if any(kw in q for kw in _FACULTY_KEYWORDS):
        return 'faculty'
    if any(kw in q for kw in _FEE_KEYWORDS):
        return 'fee'
    if any(kw in q for kw in _ADMISSION_KEYWORDS):
        return 'admission'
    if any(kw in q for kw in _UNIVERSITY_KEYWORDS):
        return 'university_info'
    if any(kw in q for kw in _COURSE_KEYWORDS):
        return 'course'
    return 'general'


# Content-type sets per query category.
# retrieval_metadata.json exposes content_type mapped from the new KB's
# category field: course, fee, about, admission, facilities, campus,
# research, placement, scholarship, faq, event, general, contact.
_CATEGORY_CONTENT_TYPES: Dict[str, set] = {
    'faculty':       {'about', 'course', 'research'},   # faculty info lives in school/course/research pages
    'fee':           {'fee', 'scholarship'},
    'admission':     {'admission', 'faq', 'course'},
    'university_info': {'about', 'facilities', 'campus', 'placement', 'research'},
    'course':        {'course', 'about'},
    'general':       set(),   # empty → no filter applied
}


def _matches_category(meta: Dict[str, Any], category: str) -> bool:
    """
    Return True if a metadata entry should be included for the given category.
    Called inside get_candidates() on every raw FAISS result.
    Uses content_type field from retrieval_metadata.json (mapped from new KB
    category values: course, fee, about, admission, facilities, etc.).
    """
    if category == 'general':
        return True  # no filtering

    ct  = meta.get('content_type', '')
    src = meta.get('source', '')

    allowed_types = _CATEGORY_CONTENT_TYPES.get(category, set())

    # Faculty: any chunk mentioning faculty/school/research qualifies —
    # new KB doesn't have a dedicated 'faculty' content_type; use broad filter
    # plus URL signal (school/ or research/ or course/ pages carry faculty info).
    if category == 'faculty':
        if not allowed_types:
            return True
        if ct in allowed_types:
            return True
        # Also accept chunks from school, deans, or research URLs
        faculty_url_signals = ('/school/', '/deans', '/research', '/director',
                               '/faculty', '/staff', '/people', '/academic')
        return any(sig in src for sig in faculty_url_signals)

    # All other categories — simple content_type membership check.
    # Empty allowed_types means no filter (general).
    if not allowed_types:
        return True
    return ct in allowed_types

class ThreadedRAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.vector_db = None
        self.index_to_doc_map = {}
        self.lock = threading.Lock()
        self._initialize()

    def _initialize(self):
        if self.embedding_model:
            return
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f'[RAG] embedding init error: {e}')

        # ---------------------------------------------------------------------------
        # Load new knowledge base:
        #   vectorstore/gdgu.index               — IndexFlatIP, 1880 vectors
        #   vectorstore/retrieval_metadata.json  — str(row) → {content, source,
        #                                          title, content_type, chunk_id}
        # Paths are resolved relative to this file so they work regardless of cwd.
        # ---------------------------------------------------------------------------
        import pathlib
        _src_dir  = pathlib.Path(__file__).parent.parent   # backend/
        _idx_path = _src_dir / 'vectorstore' / 'gdgu.index'
        _meta_path= _src_dir / 'vectorstore' / 'retrieval_metadata.json'

        if _idx_path.exists() and _meta_path.exists():
            try:
                self.vector_db = faiss.read_index(str(_idx_path))
                with open(_meta_path, 'r', encoding='utf-8') as f:
                    self.index_to_doc_map = json.load(f)
                print(f'[RAG] Loaded index (IndexFlatIP) with '
                      f'{self.vector_db.ntotal} vectors')
            except Exception as e:
                print(f'[RAG] failed to load index: {e}')
                self._create_new_index()
        else:
            print(f'[RAG] index or metadata not found at {_idx_path}')
            self._create_new_index()

    def _create_new_index(self):
        dim = 384  # all-MiniLM-L6-v2 dimension
        # Use IndexFlatIP to match the new vectorstore (L2-normalised embeddings,
        # dot-product == cosine similarity).  Legacy fallback also uses IP so
        # both paths behave identically when no index file is present.
        self.vector_db = faiss.IndexFlatIP(dim)
        self.index_to_doc_map = {}
        print('[RAG] Created new empty FAISS index (IndexFlatIP)')

    def get_candidates(self, query: str, top_k: int = None,
                       category: str = 'general') -> List[Dict[str,Any]]:
        top_k = top_k or TOP_K_RESULTS
        self._initialize()
        if not self.vector_db or self.vector_db.ntotal == 0:
            return []

        q_emb = self.embedding_model.encode([query], convert_to_numpy=True)[0]

        # Over-fetch so that after category filter + dedup we still fill top_k.
        # Category filtering can remove a large fraction of results, so we
        # fetch more aggressively when a category is set.
        multiplier = 10 if category != 'general' else 5
        fetch_k = min(top_k * multiplier, int(self.vector_db.ntotal))
        D, I = self.vector_db.search(q_emb.reshape(1, -1), fetch_k)
        
        # ── DEBUG: FAISS raw results ──────────────────────────────────────
        print(f"\n[DEBUG] ── STAGE 1: FAISS RAW RESULTS ──────────────────────")
        print(f"[DEBUG]  query={query[:70]}  category={category}")
        print(f"[DEBUG]  fetch_k={fetch_k}  threshold={_L2_THRESHOLD if category=='faculty' else _L2_THRESHOLD_RELAXED}")
        for _rank, (_sc, _ix) in enumerate(zip(D[0][:10], I[0][:10]), 1):
            if _ix == -1: continue
            _m = self.index_to_doc_map.get(str(_ix), {})
            print(f"[DEBUG]    #{_rank:>2}  score={float(_sc):>7.4f}  "
                  f"type={_m.get('content_type','?'):<12}  "
                  f"title={(_m.get('title','') or '')[:40]}")
        print(f"[DEBUG]  (showing top 10 of {fetch_k} fetched)")


        # --- Step 1: threshold filter + category filter ---
        # IndexFlatIP returns inner-product scores on L2-normalised vectors,
        # which equals cosine similarity.  Higher score = more similar.
        # Thresholds are expressed as minimum acceptable cosine similarity:
        #   IP >= 0.50  →  meaningful relevance  (was L2 <= 1.0)
        # Faculty uses the strict floor (0.50), all other categories use the
        # relaxed floor (0.45) to recover short/lightly-worded queries.
        # Env-var names are kept for backward compat; values now mean
        # minimum cosine similarity instead of maximum L2 distance.
        effective_threshold = _L2_THRESHOLD if category == 'faculty' else _L2_THRESHOLD_RELAXED
        raw = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            # IP: keep only scores ABOVE threshold (higher = more similar)
            if score < effective_threshold:
                continue
            meta = self.index_to_doc_map.get(str(idx), {})
            if not _matches_category(meta, category):
                continue

            exact_score = _exact_metadata_match_score(query, meta)
            # For IndexFlatIP: boost by adding to score (higher = more similar).
            if exact_score >= 3:
                adjusted_score = float(score) + 0.2
            elif exact_score == 2:
                adjusted_score = float(score) + 0.1
            else:
                adjusted_score = float(score)

            # Discard weak semantic matches that are not reinforced by metadata.
            # For IndexFlatIP: low score = weak match (opposite of L2).
            if exact_score == 0 and adjusted_score < 0.05:
                continue

            raw.append({
                'score':    adjusted_score,
                'index':    int(idx),
                'content':  meta.get('content', ''),
                'metadata': meta,
                'exact_match': exact_score,
            })
            
        # ── DEBUG: after threshold + category filter ──────────────────────
        print(f"\n[DEBUG] ── STAGE 2: AFTER THRESHOLD + CATEGORY FILTER ──────")
        print(f"[DEBUG]  passed={len(raw)}  rejected={fetch_k - len(raw)}")
        for _i, _r in enumerate(raw[:10], 1):
            _m = _r['metadata']
            print(f"[DEBUG]    #{_i:>2}  score={_r['score']:>7.4f}  "
                  f"exact={_r['exact_match']}  "
                  f"type={_m.get('content_type','?'):<12}  "
                  f"title={(_m.get('title','') or '')[:40]}")

        # --- Step 2: metadata-based duplicate chunk removal ---
        # Keep the highest-ranked occurrence of each chunk and preserve
        # the original retrieval order from FAISS.
        deduped: List[Dict[str, Any]] = []
        seen_keys = set()
        for item in raw:
            meta = item['metadata']
            doc_id = meta.get('doc_id') or meta.get('document_id')
            chunk_index = meta.get('chunk_index')
            if doc_id is not None and chunk_index is not None:
                key = (str(doc_id), int(chunk_index))
            else:
                norm = ' '.join(item['content'].split()).lower()
                key = ('content', hashlib.md5(norm.encode('utf-8')).hexdigest())

            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)
            if len(deduped) >= top_k:
                break

                # ── DEBUG: after deduplication ────────────────────────────────────
        print(f"\n[DEBUG] ── STAGE 3: AFTER DEDUPLICATION ────────────────────")
        print(f"[DEBUG]  before={len(raw)}  after={len(deduped)}  dupes_removed={len(raw)-len(deduped)}")
        for _i, _r in enumerate(deduped[:10], 1):
            _m = _r['metadata']
            print(f"[DEBUG]    #{_i:>2}  score={_r['score']:>7.4f}  "
                  f"type={_m.get('content_type','?'):<12}  "
                  f"url={(_m.get('source','') or '')[:60]}")


        if DEBUG_RAG:
            _sep  = "=" * 65
            _line = "-" * 65
            rejected = fetch_k - len(raw)
            print(f"\n{_sep}\n  [RAG] Dense Candidate Fetch Detail\n{_sep}")
            print(f"  Query       : {query[:80]}")
            print(f"  Category    : {category}")
            print(f"  Fetched     : {len(D[0])}")
            print(f"  L2 Threshold: {effective_threshold} (category={category})")
            print(f"  Passed      : {len(raw)} (rejected: {rejected})")
            print(f"  After Dedup : {len(deduped)}")
            print(_sep)

        return deduped

    def _profile_key(self, candidate: Dict[str, Any]) -> str:
        meta = candidate['metadata']
        doc_id = meta.get('doc_id') or meta.get('document_id')
        if doc_id:
            return str(doc_id)
        source = meta.get('source', '')
        title = meta.get('title', '')
        return f"{source}|{title}"

    def _entity_key(self, metadata: Dict[str, Any]) -> str:
        # Prefer an explicit faculty name if available; fall back to page identity.
        name = _normalize_text(metadata.get('name') or '')
        if name:
            return name
        return self._profile_key({'metadata': metadata})

    def get_context_for_query(self, query: str, top_k: int = None,
                              category: str = 'general') -> str:
        cands = self.get_candidates(query, top_k=top_k, category=category)

        # If an exact faculty metadata match exists, keep only that faculty profile.
        exact_faculty = [
            cand for cand in cands
            if cand.get('exact_match', 0) >= 2
            and _matches_category(cand['metadata'], 'faculty')
        ]
        if exact_faculty:
            primary = max(exact_faculty, key=lambda cand: cand['score'])  # IP: higher=better
            primary_key = self._entity_key(primary['metadata'])
            cands = [cand for cand in cands if self._entity_key(cand['metadata']) == primary_key]

        profile_blocks: Dict[str, Dict[str, Any]] = {}
        ordered_keys: List[str] = []

        for candidate in cands:
            key = self._profile_key(candidate)
            if key not in profile_blocks:
                profile_blocks[key] = {
                    'metadata': candidate['metadata'],
                    'contents': [],
                    'first_score': candidate['score'],
                }
                ordered_keys.append(key)

            profile_blocks[key]['contents'].append(candidate['content'])
            if candidate['score'] < profile_blocks[key]['first_score']:
                profile_blocks[key]['first_score'] = candidate['score']

        parts = []
        for key in ordered_keys:
            block = profile_blocks[key]
            meta = block['metadata']
            src = meta.get('source', 'Unknown')
            title = meta.get('title', '')
            body = '\n\n'.join(block['contents']).strip()
            source_label = title or src
            url_line = f"URL: {src}" if isinstance(src, str) and src.startswith(('http://', 'https://')) else ''
            parts.append(f"[Source: {source_label}]\n{url_line}\n{body}" if url_line else f"[Source: {source_label}]\n{body}")

        context = '\n\n---\n\n'.join(parts) if parts else ''
        
                # ── DEBUG: final context sent to LLM ─────────────────────────────
        print(f"\n[DEBUG] ── STAGE 4: FINAL CONTEXT SENT TO LLM ─────────────")
        print(f"[DEBUG]  blocks={len(parts)}  total_chars={len(context)}")
        for _i, _key in enumerate(ordered_keys[:5], 1):
            _blk = profile_blocks[_key]
            _m   = _blk['metadata']
            _src = _m.get('source', '')
            _ttl = (_m.get('title', '') or '')[:50]
            _txt = (' '.join(_blk['contents'][0].split()))[:100] if _blk['contents'] else ''
            print(f"[DEBUG]    block #{_i}  score={_blk['first_score']:.4f}  "
                  f"title={_ttl}")
            print(f"[DEBUG]            url={_src[:70]}")
            print(f"[DEBUG]            content_preview={_txt[:100]}...")
        if not parts:
            print("[DEBUG]    *** EMPTY CONTEXT — LLM will say not found ***")
        print(f"[DEBUG] ─────────────────────────────────────────────────────")
        # ─────────────────────────────────────────────────────────────────


        return context
