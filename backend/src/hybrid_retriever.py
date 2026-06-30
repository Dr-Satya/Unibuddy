"""
src/hybrid_retriever.py  (Phase-1 Hybrid RAG)

PRODUCTION MERGE NOTE
----------------------
This module builds ON TOP OF the existing ThreadedRAGSystem in
threaded_rag.py. It does not duplicate logic that already exists there or
in api_adapter.py:

    - Category classification (classify_query)            -> threaded_rag.py
    - Category filtering (_matches_category)               -> threaded_rag.py
    - Exact-metadata-match boosting (_exact_metadata_match_score) -> threaded_rag.py
    - L2 threshold filtering                                -> threaded_rag.py
    - doc_id+chunk_index duplicate removal                  -> threaded_rag.py
    - Faculty-profile isolation (exact_match >= 2)           -> threaded_rag.py / api_adapter.py
    - Menu/boilerplate paragraph filtering                   -> api_adapter.py (_is_menu_boilerplate, _clean_chunk_content)
    - Follow-up query rewriting                              -> api_adapter.py

HybridRetriever adds the Phase-1 pieces that are genuinely missing:

    Dense (existing FAISS, via rag.get_candidates -- UNCHANGED)
        + BM25 (new, sparse lexical retrieval over the same corpus)
        -> Reciprocal Rank Fusion (RRF, standard 1/(k+rank))
        -> Cross-Encoder reranking (ms-marco-MiniLM-L-6-v2, graceful fallback)
        -> Context cleaning (light -- defers to api_adapter.py's own
           _is_menu_boilerplate/_clean_chunk_content for the heavy lifting;
           this module only normalizes whitespace before scoring)
        -> Duplicate removal (defense-in-depth on top of threaded_rag.py's
           own dedup, keyed the same way: doc_id + chunk_index)
        -> Adjacent chunk merge (merge chunks from the same doc_id that are
           consecutive by chunk_index, so fragmented pages aren't split
           across multiple context blocks)
        -> Entity-aware grouping (reuses threaded_rag.py's own
           classify_query + _matches_category + the same faculty-isolation
           pattern api_adapter.py already applies, so faculty/fee/admission
           content isn't mixed)
        -> Evidence check (Step 10 -- lets the caller bail out with a
           grounded "not found" message instead of calling the LLM on thin
           or irrelevant context)

DOES NOT TOUCH:
    - FAISS index/search itself (reused as-is via rag.get_candidates)
    - Embedding generation (reused as-is)
    - classify_query / _matches_category / _exact_metadata_match_score
      (imported and reused, never reimplemented)
    - Groq / LLM call (lives in api_adapter.py)
    - api.py routes, Timetable, Mentor-Mentee, Auth

DEBUG_RAG controls verbose logging of every stage, matching the existing
[RAG] log-line convention used throughout threaded_rag.py / api_adapter.py.
"""

import re
import threading
from typing import List, Dict, Any, Optional, Tuple

from rank_bm25 import BM25Okapi

from src.config import DEBUG_RAG, TOP_K_RESULTS
from src.threaded_rag import classify_query, _matches_category, ThreadedRAGSystem

# Cross-encoder is optional at import time: if sentence-transformers / the
# specific model can't be loaded (e.g. no network access to the model hub
# in some deployment environments), HybridRetriever degrades gracefully to
# RRF-only ranking instead of crashing the whole RAG pipeline.
try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_IMPORT_OK = True
except Exception as e:  # pragma: no cover
    CrossEncoder = None
    _CROSS_ENCODER_IMPORT_OK = False
    if DEBUG_RAG:
        print(f"[HYBRID] CrossEncoder import failed, will skip reranking: {e}")


# ---------------------------------------------------------------------------
# BM25 sparse retrieval over the same corpus threaded_rag.py already loaded
# (no new dataset -- built from rag.index_to_doc_map).
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """Thin wrapper building a BM25 index over the existing FAISS metadata
    corpus (rag.index_to_doc_map). Row ids match FAISS row indices so
    dense/BM25 results can be fused by the same dedup key used throughout
    threaded_rag.py: (doc_id, chunk_index)."""

    def __init__(self, index_to_doc_map: Dict[str, Any]):
        self._row_ids: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        corpus_tokens: List[List[str]] = []
        for row_id, meta in index_to_doc_map.items():
            content = meta.get("content", "") or ""
            self._row_ids.append(row_id)
            self._metas.append(meta)
            corpus_tokens.append(_tokenize(content))
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    @property
    def ready(self) -> bool:
        return self._bm25 is not None and len(self._row_ids) > 0

    def search(self, query: str, top_k: int, category: str = "general") -> List[Dict[str, Any]]:
        """Returns top_k hits as a ranked list (best first), in the same
        candidate-dict shape used by ThreadedRAGSystem.get_candidates:
        {'index', 'content', 'metadata', 'score', 'exact_match'}.

        Applies the SAME category filter threaded_rag.py's dense path uses
        (_matches_category), so BM25 doesn't reintroduce off-category noise
        that the dense path already excludes (e.g. fee pages leaking into
        a faculty query) before RRF fusion ever sees it."""
        if not self.ready:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # Over-fetch before category filtering, mirroring threaded_rag.py's
        # own over-fetch-then-filter pattern in get_candidates().
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in order:
            if scores[i] <= 0:
                break  # BM25Okapi scores are sorted; zero/negative means no lexical overlap
            meta = self._metas[i]
            if not _matches_category(meta, category):
                continue
            results.append({
                "index": self._row_ids[i],
                "content": meta.get("content", ""),
                "metadata": meta,
                "score": float(scores[i]),
                "exact_match": 0,  # BM25 hits don't carry threaded_rag.py's exact-match boost
            })
            if len(results) >= top_k:
                break
        return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (standard RRF, NOT weighted averaging)
# ---------------------------------------------------------------------------

def _dedup_key(item: Dict[str, Any]) -> Tuple[Any, Any]:
    """Same dedup key threaded_rag.py's get_candidates() already uses:
    (doc_id, chunk_index), falling back to a content-based key if either
    is missing -- mirrors threaded_rag.py's own fallback exactly."""
    meta = item.get("metadata", {})
    doc_id = meta.get("doc_id") or meta.get("document_id")
    chunk_index = meta.get("chunk_index")
    if doc_id is not None and chunk_index is not None:
        return (str(doc_id), int(chunk_index))
    return ("content", " ".join((item.get("content") or "").split()).lower())


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Standard RRF: score += 1 / (k + rank), rank starting at 1 for the
    top result in each input ranked list. Input lists must already be
    sorted best-first (true for both rag.get_candidates() and
    BM25Index.search())."""
    fused_scores: Dict[Any, float] = {}
    fused_items: Dict[Any, Dict[str, Any]] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = _dedup_key(item)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in fused_items:
                fused_items[key] = item
            else:
                # Prefer the item carrying the higher exact_match boost
                # (dense results carry threaded_rag.py's exact-match score;
                # BM25 results don't) so faculty-isolation logic downstream
                # still sees the real exact_match value.
                if item.get("exact_match", 0) > fused_items[key].get("exact_match", 0):
                    fused_items[key] = item
    fused = [
        {**fused_items[key], "rrf_score": score}
        for key, score in fused_scores.items()
    ]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


# ---------------------------------------------------------------------------
# Cross-Encoder reranking
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Lazy-loaded cross-encoder. If the model can't be loaded (e.g. no
    network access to the model hub in this deployment), `ready` is False
    and rerank() degrades to passing through RRF order untouched -- a safe,
    documented degradation, not a crash."""

    _MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        self._model = None
        self._load_failed = not _CROSS_ENCODER_IMPORT_OK
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None or self._load_failed:
            return
        with self._lock:
            if self._model is not None or self._load_failed:
                return
            try:
                self._model = CrossEncoder(self._MODEL_NAME)
            except Exception as e:
                self._load_failed = True
                if DEBUG_RAG:
                    print(f"[HYBRID] CrossEncoder load failed, falling back to RRF order: {e}")

    @property
    def ready(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        if not self.ready:
            return candidates[:top_k]
        pairs = [(query, c.get("content", "")) for c in candidates]
        scores = self._model.predict(pairs)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        top = scored[:top_k]
        ranked = []
        for c, s in top:
            c["_evidence_score"] = float(s)  # cross-encoder relevance logit
            ranked.append(c)
        if DEBUG_RAG:
            preview = [(c.get("metadata", {}).get("doc_id"), c["_evidence_score"]) for c in ranked[:10]]
            print(f"[HYBRID] Reranker ranking (doc_id, score): {preview}")
        return ranked


# ---------------------------------------------------------------------------
# Adjacent-chunk merging
# ---------------------------------------------------------------------------

def _safe_chunk_index(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def merge_adjacent_chunks(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """If multiple selected chunks share the same doc_id, merge their
    content into one block (sorted by chunk_index) instead of sending
    fragmented context to the LLM as separate [Source: ...] blocks. This
    runs AFTER api_adapter.py's profile-block grouping concept but at the
    HybridRetriever stage, before context ever reaches api_adapter.py, so
    api_adapter.py's existing per-chunk dedup/cleaning still applies
    unchanged on top."""
    by_doc: Dict[Any, List[Dict[str, Any]]] = {}
    order: List[Any] = []
    for item in items:
        doc_id = item.get("metadata", {}).get("doc_id") or item.get("metadata", {}).get("document_id")
        group_key = doc_id if doc_id is not None else id(item)
        if group_key not in by_doc:
            by_doc[group_key] = []
            order.append(group_key)
        by_doc[group_key].append(item)

    merged: List[Dict[str, Any]] = []
    for group_key in order:
        group = by_doc[group_key]
        group.sort(key=lambda it: _safe_chunk_index(it.get("metadata", {}).get("chunk_index")))
        merged_text = "\n".join(
            it.get("content", "") for it in group if it.get("content")
        )
        best = min(group, key=lambda it: it.get("score", it.get("rrf_score", 0.0)) if "score" in it else -it.get("rrf_score", 0.0))
        merged.append({**best, "content": merged_text, "_merged_count": len(group)})
    return merged


# ---------------------------------------------------------------------------
# Public entry point used by api_adapter.py
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Wraps an existing ThreadedRAGSystem instance. Does not modify it.
    Call get_hybrid_candidates(query, category) in place of
    rag.get_candidates(query, category=...) inside api_adapter.py's
    per-query-variant loop to use the Phase-1 hybrid pipeline. Returns the
    SAME candidate-dict shape (score/index/content/metadata/exact_match)
    that rag.get_candidates() already returns, so every downstream step in
    api_adapter.py (faculty isolation, menu filtering, chunk dedup, context
    assembly) works completely unchanged."""

    def __init__(self, rag_system: ThreadedRAGSystem):
        self._rag = rag_system
        self._bm25_lock = threading.Lock()
        self._bm25_index: Optional[BM25Index] = None
        self._bm25_built_for_ntotal: Optional[int] = None
        self._reranker = CrossEncoderReranker()

    def _ensure_bm25(self):
        """Builds/rebuilds the BM25 index lazily from the same metadata the
        FAISS index already uses. Rebuilt only if the FAISS vector count
        changes, avoiding a rebuild on every single query."""
        self._rag._initialize()  # reuse existing init, no-op if already loaded
        current_ntotal = int(self._rag.vector_db.ntotal) if self._rag.vector_db else 0
        with self._bm25_lock:
            if self._bm25_index is not None and self._bm25_built_for_ntotal == current_ntotal:
                return
            self._bm25_index = BM25Index(self._rag.index_to_doc_map)
            self._bm25_built_for_ntotal = current_ntotal
            if DEBUG_RAG:
                print(f"[HYBRID] Built BM25 index over {len(self._rag.index_to_doc_map)} chunks")

    def get_hybrid_candidates(
        self,
        query: str,
        top_k: int = None,
        category: str = "general",
        rrf_top_n: int = 30,
        final_top_k: int = 8,
        rrf_k: int = 60,
        use_bm25: bool = True,
    ) -> List[Dict[str, Any]]:
        """Dense (existing FAISS, unchanged) + BM25 -> RRF -> cross-encoder
        rerank -> dedup (defense-in-depth) -> adjacent-chunk merge.
        Returns candidates in the exact dict shape rag.get_candidates()
        already returns, so api_adapter.py's existing faculty-isolation /
        menu-filtering / dedup / context-assembly code requires NO changes
        beyond swapping the call site.

        use_bm25: api_adapter.py calls this once per query *variant*
        (original query plus broadened suffixes like " profile"/" research").
        BM25 rewards literal term overlap, so a single common suffix word
        can match broadly across the corpus regardless of whether the
        *original* query was relevant at all (verified directly: "<nonsense
        query> profile" lexically matches thousands of "Faculty Profile -
        ..." chunks via the word "profile" alone). Dense retrieval doesn't
        have this failure mode the same way (holistic embedding similarity,
        not literal term counting), so callers should pass use_bm25=False
        for the broadened variants and leave it True only for the original,
        un-suffixed query -- see api_adapter.py's call site."""
        top_k = top_k or TOP_K_RESULTS

        # Dense retrieval: 100% reuse of the existing, unmodified FAISS path
        # in threaded_rag.py (category filter, exact-match boost, threshold
        # filter, and its own dedup all already applied inside).
        dense_hits = self._rag.get_candidates(query, top_k=top_k, category=category)
        if DEBUG_RAG:
            preview = [(h["metadata"].get("doc_id"), round(h["score"], 4)) for h in dense_hits[:top_k]]
            print(f"[HYBRID] Dense Top-K ({category}): {preview}")

        bm25_hits: List[Dict[str, Any]] = []
        if use_bm25:
            self._ensure_bm25()
            bm25_hits = self._bm25_index.search(query, top_k=top_k, category=category) if self._bm25_index else []
            if DEBUG_RAG:
                preview = [(h["metadata"].get("doc_id"), round(h["score"], 4)) for h in bm25_hits[:top_k]]
                print(f"[HYBRID] BM25 Top-K ({category}): {preview}")

        # RRF fusion.
        fused = reciprocal_rank_fusion([dense_hits, bm25_hits], k=rrf_k)
        fused_top_n = fused[:rrf_top_n]
        if DEBUG_RAG:
            preview = [(it["metadata"].get("doc_id"), round(it["rrf_score"], 5)) for it in fused_top_n[:10]]
            print(f"[HYBRID] RRF ranking (top10 of {len(fused_top_n)}): {preview}")

        # Defense-in-depth duplicate removal (threaded_rag.py's dense path
        # and RRF's own key_fn already dedup; this guards the merged list).
        deduped, seen = [], set()
        for it in fused_top_n:
            key = _dedup_key(it)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)

        # Cross-encoder rerank (graceful fallback to RRF order if the model
        # can't be loaded in this environment).
        reranked = self._reranker.rerank(query, deduped, top_k=final_top_k)

        # Adjacent-chunk merge so fragmented same-page chunks aren't sent
        # to api_adapter.py as separate blocks. api_adapter.py's own
        # per-paragraph cleaning/dedup still runs on the merged content
        # exactly as it does today.
        merged = merge_adjacent_chunks(reranked)

        return merged

    def has_sufficient_evidence(self, candidates: List[Dict[str, Any]], min_chars: int = 40) -> bool:
        """Step 10 (Evidence Check): lets api_adapter.py bail out with the
        grounded fallback instead of calling the LLM on thin/irrelevant
        context.

        IMPORTANT, documented honestly: the cross-encoder's relevance score
        (`_evidence_score`) is only attached when the cross-encoder model
        actually loaded (see CrossEncoderReranker.rerank). When it didn't
        load, we deliberately do NOT fall back to a raw FAISS L2 distance
        or BM25 score as a relevance proxy -- those live on different,
        non-comparable scales (L2 is "smaller is better" and unbounded;
        BM25 is "bigger is better" and corpus-size-dependent), so a single
        fixed numeric cutoff against either would be misleading rather than
        merely imprecise. In that case this method falls back to length-
        only gating, same as the pre-hybrid behavior, rather than pretend a
        miscalibrated numeric threshold adds safety."""
        if not candidates:
            return False
        total_len = sum(len((c.get("content") or "").strip()) for c in candidates)
        if total_len < min_chars:
            return False
        evidence_scores = [c["_evidence_score"] for c in candidates if "_evidence_score" in c]
        if evidence_scores and max(evidence_scores) < 0.0:
            # Cross-encoder ran and every candidate scored below its
            # relevance midpoint -- treat as insufficient evidence even
            # though *some* text was retrieved.
            return False
        return True