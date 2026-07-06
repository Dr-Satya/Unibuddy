"""
src/rag_debug.py  —  Professional RAG pipeline debug logging.

ALL output is gated behind DEBUG_RAG=True (imported from config).
When DEBUG_RAG is False every function in this module is a no-op,
so production performance is completely unaffected.

Usage:
    from src.rag_debug import RagDebugger
    dbg = RagDebugger()          # one instance per request, in get_reply()
    dbg.step_intent(...)
    dbg.step_followup(...)
    ...
    dbg.summary()                # prints the timing table at the end

PATCH INSTRUCTIONS  (api_adapter.py, hybrid_retriever.py):
  1. Add:   from src.rag_debug import RagDebugger
  2. Replace each existing if DEBUG_RAG: print(...) block with the
     corresponding dbg.step_*() call shown in the docstring for each method.
  See bottom of this file for the exact replacement diff.
"""

import os
import re
import time
import threading
from typing import Any, Dict, List, Optional

from src.config import DEBUG_RAG

# ── optional resource monitoring ──────────────────────────────────────────────
try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _psutil = None
    _PSUTIL = False

_SEP  = "=" * 65
_WARN = "⚠  WARNING"
_LINE = "-" * 65


def _now_ms() -> float:
    """Monotonic millisecond timestamp."""
    return time.monotonic() * 1000.0


def _thread_name() -> str:
    return threading.current_thread().name


def _ram_mb() -> str:
    if not _PSUTIL:
        return "psutil not installed"
    try:
        proc = _psutil.Process(os.getpid())
        return f"{proc.memory_info().rss / 1024 / 1024:.1f} MB"
    except Exception:
        return "unavailable"


def _cpu_pct() -> str:
    if not _PSUTIL:
        return "psutil not installed"
    try:
        return f"{_psutil.cpu_percent(interval=None):.1f} %"
    except Exception:
        return "unavailable"


def _preview(text: str, max_chars: int = 150) -> str:
    if not text:
        return "(empty)"
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars] + "…"


def _token_estimate(text: str) -> int:
    """Rough token count: ~4 chars per token (GPT/Llama typical)."""
    return max(1, len(text) // 4)


def _print_section(step_num: int, title: str, lines: List[str],
                    elapsed_ms: Optional[float] = None,
                    warnings: Optional[List[str]] = None):
    """Core printer — all debug output goes through here."""
    if not DEBUG_RAG:
        return
    print(f"\n{_SEP}")
    print(f"  STEP {step_num} : {title}")
    print(_SEP)
    print(f"  Thread : {_thread_name()}")
    if elapsed_ms is not None:
        print(f"  Time   : {elapsed_ms:.2f} ms")
    print(_LINE)
    for line in lines:
        print(f"  {line}")
    if warnings:
        print(_LINE)
        for w in warnings:
            print(f"  {_WARN}: {w}")
    print(_SEP)


# ──────────────────────────────────────────────────────────────────────────────
# Public class — one instance per request in get_reply()
# ──────────────────────────────────────────────────────────────────────────────

class RagDebugger:
    """
    Collects per-stage timing and prints structured debug sections.

    Instantiate once at the start of get_reply() and call the appropriate
    step_*() method at each pipeline stage. Call summary() at the end.
    """

    def __init__(self, query: str = ""):
        self._query  = query
        self._t0     = _now_ms()
        self._timings: Dict[str, float] = {}

    # ── internal timing helpers ───────────────────────────────────────────────

    def _record(self, stage: str, elapsed_ms: float):
        self._timings[stage] = elapsed_ms

    # ── step 1  Intent / small-talk / guardrail detection ────────────────────

    def step_intent(self, query: str, intent: str,
                     elapsed_ms: float = 0.0, confidence: str = "n/a"):
        """
        PATCH — replace in api_adapter.py:
            _debug_section(1, "Intent Detection", ...)
        with:
            dbg.step_intent(user_message, intent_label, elapsed_ms=t)
        """
        self._record("Intent", elapsed_ms)
        warnings = []
        if intent == "University Knowledge Base Query" and not query.strip():
            warnings.append("Empty query passed to intent detection.")
        _print_section(1, "Intent Detection", [
            f"Input Query  : {_preview(query)}",
            f"Detected     : {intent}",
            f"Confidence   : {confidence}",
        ], elapsed_ms=elapsed_ms, warnings=warnings or None)

    # ── step 2  Follow-up detection / query rewriting ─────────────────────────

    def step_followup(self, original: str, rewritten: str,
                       focus_field: Optional[str],
                       history_turns: int,
                       elapsed_ms: float = 0.0):
        """
        PATCH — replace in api_adapter.py:
            _debug_section(2, "Follow-up Detection", ...)
        with:
            dbg.step_followup(user_message, rewritten, focus_field, len(history), t)
        """
        self._record("Followup", elapsed_ms)
        rewritten_flag = "(rewritten)" if original != rewritten else "(unchanged)"
        _print_section(2, "Follow-up Detection", [
            f"Original Query  : {_preview(original)}",
            f"Rewritten Query : {_preview(rewritten)} {rewritten_flag}",
            f"Focus Field     : {focus_field or '(none — full answer)'}",
            f"History Turns   : {history_turns}",
        ], elapsed_ms=elapsed_ms)

    # ── step 3  Category classification ──────────────────────────────────────

    def step_category(self, query: str, category: str,
                       is_list_query: bool,
                       corrected: bool = False,
                       elapsed_ms: float = 0.0):
        """
        PATCH — replace in api_adapter.py:
            _debug_section(3, "Category Classification", ...)
        with:
            dbg.step_category(user_message, category, is_list_query, corrected, t)
        """
        self._record("Category", elapsed_ms)
        note = "(auto-corrected from 'general' via focus_field)" if corrected else ""
        _print_section(3, "Category Classification", [
            f"Input Query    : {_preview(query)}",
            f"Category       : {category} {note}",
            f"List / Aggregate Query : {'Yes — history suppressed in prompt' if is_list_query else 'No'}",
        ], elapsed_ms=elapsed_ms)

    # ── step 4  Dense retrieval ───────────────────────────────────────────────

    def step_dense(self, query: str, category: str,
                    candidates: List[Dict[str, Any]],
                    rejected_count: int,
                    fetch_k: int,
                    embed_time_ms: float,
                    search_time_ms: float,
                    threshold: float,
                    model_name: str = "all-MiniLM-L6-v2",
                    embed_dim: int = 384):
        """
        PATCH — add immediately after rag.get_candidates() returns in
        HybridRetriever.get_hybrid_candidates(), replacing the existing
        [HYBRID] Dense Top-K print with:
            dbg.step_dense(query, category, dense_hits, rejected, fetch_k,
                           embed_ms, search_ms, threshold)
        embed_time_ms / search_time_ms: wrap the encode() and search() calls
        with time.monotonic() before/after.
        """
        elapsed_ms = embed_time_ms + search_time_ms
        self._record("Dense", elapsed_ms)
        warnings = []
        if not candidates:
            warnings.append("Dense retrieval returned zero chunks.")
        elif len(candidates) < 3:
            warnings.append(f"Very few dense results ({len(candidates)}). Low confidence retrieval.")

        chunk_lines = []
        for i, c in enumerate(candidates[:10], 1):
            meta  = c.get("metadata", {})
            score = c.get("score", 0.0)
            src   = meta.get("source", "unknown")
            ctype = meta.get("content_type", "?")
            clen  = len(c.get("content", ""))
            prev  = _preview(c.get("content", ""), 150)
            chunk_lines.append(f"  [{i:02d}] score={score:.4f} | type={ctype} | len={clen}ch")
            chunk_lines.append(f"       src   : {src}")
            chunk_lines.append(f"       preview: {prev}")

        lines = [
            f"Embedding Model  : {model_name}",
            f"Embedding Dim    : {embed_dim}",
            f"Embed Time       : {embed_time_ms:.2f} ms",
            f"Vector Search    : {search_time_ms:.2f} ms",
            f"L2 Threshold     : {threshold}",
            f"Vectors Searched : {fetch_k}",
            f"Passed Threshold : {len(candidates)}",
            f"Rejected         : {rejected_count}",
            f"Category Filter  : {category}",
            _LINE,
            "Top Chunks:",
        ] + chunk_lines

        _print_section(4, "Dense Retrieval", lines,
                        elapsed_ms=elapsed_ms, warnings=warnings or None)

    # ── step 5  BM25 retrieval ────────────────────────────────────────────────

    def step_bm25(self, query: str, tokenized: List[str],
                   candidates: List[Dict[str, Any]],
                   category: str,
                   elapsed_ms: float = 0.0):
        """
        PATCH — in hybrid_retriever.py BM25Index.search(), replace the
        existing [HYBRID] BM25 Top-K print with:
            dbg.step_bm25(query, q_tokens, results, category, t)
        """
        self._record("BM25", elapsed_ms)
        warnings = []
        if not candidates:
            warnings.append("BM25 returned zero results — no lexical overlap with corpus.")
        chunk_lines = []
        for i, c in enumerate(candidates[:10], 1):
            score = c.get("score", 0.0)
            prev  = _preview(c.get("content", ""), 150)
            chunk_lines.append(f"  [{i:02d}] score={score:.4f} | preview: {prev}")

        lines = [
            f"Query           : {_preview(query)}",
            f"Tokenized Query : {tokenized[:20]} … ({len(tokenized)} tokens)",
            f"Category Filter : {category}",
            f"Results         : {len(candidates)}",
            _LINE,
            "Top BM25 Chunks:",
        ] + chunk_lines

        _print_section(5, "BM25 Retrieval", lines,
                        elapsed_ms=elapsed_ms, warnings=warnings or None)

    # ── step 6  RRF fusion + hybrid merge ─────────────────────────────────────

    def step_rrf(self, dense_hits: int, bm25_hits: int,
                  fused: List[Dict[str, Any]],
                  deduped_count: int,
                  rrf_k: int = 60,
                  elapsed_ms: float = 0.0):
        """
        PATCH — in hybrid_retriever.py, replace the [HYBRID] RRF ranking
        print with:
            dbg.step_rrf(len(dense_hits), len(bm25_hits), fused_top_n,
                         deduped_count, rrf_k, t)
        """
        self._record("RRF", elapsed_ms)
        warnings = []
        if deduped_count > 0:
            warnings.append(f"{deduped_count} duplicate(s) removed after RRF fusion.")

        rank_lines = []
        for i, c in enumerate(fused[:10], 1):
            rrf   = c.get("rrf_score", 0.0)
            doc   = c.get("metadata", {}).get("doc_id", "?")
            prev  = _preview(c.get("content", ""), 100)
            rank_lines.append(f"  [{i:02d}] rrf={rrf:.5f} | doc={str(doc)[:24]} | {prev}")

        lines = [
            f"Dense Input    : {dense_hits} chunks",
            f"BM25  Input    : {bm25_hits} chunks",
            f"RRF k          : {rrf_k}",
            f"Fused Total    : {len(fused)}",
            f"Duplicates Removed : {deduped_count}",
            _LINE,
            "Combined Ranking (top 10):",
        ] + rank_lines

        _print_section(6, "Hybrid RRF Fusion", lines,
                        elapsed_ms=elapsed_ms, warnings=warnings or None)

    # ── step 7  Cross-encoder reranking ──────────────────────────────────────

    def step_rerank(self, before: List[Dict[str, Any]],
                     after: List[Dict[str, Any]],
                     reranker_available: bool,
                     elapsed_ms: float = 0.0):
        """
        PATCH — in CrossEncoderReranker.rerank(), replace the existing
        [HYBRID] Reranker ranking print with:
            dbg.step_rerank(candidates, ranked, self.ready, t)
        """
        self._record("Rerank", elapsed_ms)
        warnings = []
        if not reranker_available:
            warnings.append("Cross-encoder NOT loaded — using RRF order (graceful degradation).")

        removed = len(before) - len(after)
        before_lines, after_lines = [], []
        for i, c in enumerate(before[:10], 1):
            s = c.get("rrf_score", c.get("score", 0.0))
            p = _preview(c.get("content", ""), 100)
            before_lines.append(f"  [{i:02d}] pre_score={s:.4f} | {p}")
        for i, c in enumerate(after[:10], 1):
            s = c.get("_evidence_score", c.get("rrf_score", 0.0))
            p = _preview(c.get("content", ""), 100)
            after_lines.append(f"  [{i:02d}] ce_score={s:.4f} | {p}")

        lines = (
            [f"Reranker        : {'cross-encoder/ms-marco-MiniLM-L-6-v2' if reranker_available else 'UNAVAILABLE — RRF passthrough'}",
             f"Candidates In   : {len(before)}",
             f"Candidates Out  : {len(after)}",
             f"Removed         : {removed} (below keep threshold)",
             _LINE,
             "Before Rerank:"]
            + before_lines
            + [_LINE, "After Rerank:"]
            + after_lines
        )
        _print_section(7, "Cross-Encoder Reranking", lines,
                        elapsed_ms=elapsed_ms, warnings=warnings or None)

    # ── step 8  Prompt building ───────────────────────────────────────────────

    def step_prompt(self, prompt: str, template_name: str,
                     full_context: str, history: str,
                     user_question: str,
                     elapsed_ms: float = 0.0):
        """
        PATCH — in api_adapter.py, after `prompt = template.format(...)`,
        replace _debug_section(10, "Prompt", ...) with:
            dbg.step_prompt(prompt, "FACULTY" or "GENERAL", full_context,
                            history, user_message, t)
        """
        self._record("Prompt", elapsed_ms)
        prompt_len     = len(prompt)
        tokens_est     = _token_estimate(prompt)
        context_tokens = _token_estimate(full_context)
        history_tokens = _token_estimate(history)
        q_tokens       = _token_estimate(user_question)
        warnings       = []
        if tokens_est > 3500:
            warnings.append(f"Prompt is large ({tokens_est} est. tokens) — may hit context limit.")
        _print_section(8, "Prompt Building", [
            f"Template        : {template_name}",
            f"Prompt Length   : {prompt_len} chars",
            f"Est. Tokens     : {tokens_est}",
            f"  Context Tokens: {context_tokens}",
            f"  History Tokens: {history_tokens}",
            f"  Question Tokens: {q_tokens}",
            _LINE,
            "Prompt (first 800 chars):",
            prompt[:800] + ("…[truncated]" if len(prompt) > 800 else ""),
        ], elapsed_ms=elapsed_ms, warnings=warnings or None)

    # ── step 9  LLM generation ───────────────────────────────────────────────

    def step_llm(self, model_name: str, max_tokens: int,
                  temperature: float, streaming: bool,
                  elapsed_ms: float = 0.0):
        """
        PATCH — in api_adapter.py, after the model.generate() call, replace
        _debug_section(11, "Raw LLM Output", ...) with two calls:
            dbg.step_llm(model_name, 1200, 0.1, False, gen_ms)
            dbg.step_llm_output(raw_text)
        """
        self._record("Generation", elapsed_ms)
        _print_section(9, "LLM Generation", [
            f"Model           : {model_name}",
            f"Max Tokens      : {max_tokens}",
            f"Temperature     : {temperature}",
            f"Streaming       : {'Yes' if streaming else 'No'}",
            f"Latency         : {elapsed_ms:.2f} ms",
        ], elapsed_ms=elapsed_ms)

    def step_llm_output(self, raw_text: str):
        """Print the raw LLM output (called right after step_llm)."""
        if not DEBUG_RAG:
            return
        print(f"\n{_LINE}")
        print("  RAW LLM OUTPUT:")
        print(_LINE)
        print(raw_text[:2000] + ("…[truncated]" if len(raw_text) > 2000 else ""))
        print(_LINE)

    # ── step 10  Post-processing ──────────────────────────────────────────────

    def step_postprocess(self, structured: Dict[str, Any],
                          html_reply: str,
                          elapsed_ms: float = 0.0):
        """
        PATCH — in _finish_reply(), replace _debug_section(12, ...) with:
            dbg.step_postprocess(structured, html_reply, t)
        """
        self._record("PostProcess", elapsed_ms)
        html_len   = len(html_reply)
        warnings   = []
        missing    = [k for k, v in structured.items()
                      if isinstance(v, list) and not v and k not in ("links", "sources")]
        if missing:
            warnings.append(f"Missing structured fields (possible hallucination risk): {missing}")

        lines = [
            "Parsed Fields:",
        ]
        for k, v in structured.items():
            if v:
                lines.append(f"  {k:14s} : {_preview(str(v), 120)}")
        lines += [
            _LINE,
            f"HTML Size       : {html_len} chars",
            "HTML Preview    :",
            _preview(re.sub(r"<[^>]+>", "", html_reply), 300),
        ]
        _print_section(10, "Post Processing", lines,
                        elapsed_ms=elapsed_ms, warnings=warnings or None)

    # ── step 11  Source extraction ────────────────────────────────────────────

    def step_sources(self, urls: List[str],
                      duplicates_removed: int = 0,
                      saved_path: Optional[str] = None,
                      elapsed_ms: float = 0.0):
        """
        PATCH — in _finish_reply(), replace _debug_section(13, "Sources", ...)
        with:
            dbg.step_sources(context_source_urls, dups, path, t)
        """
        self._record("Sources", elapsed_ms)
        warnings = []
        if not urls:
            warnings.append("No URLs extracted from retrieved context.")
        lines = [f"URLs Found      : {len(urls)}",
                 f"Duplicates Removed: {duplicates_removed}",
                 f"Saved To        : {saved_path or '(not saved this turn)'}",
                 _LINE,
                 "Extracted URLs:"]
        for u in urls[:10]:
            lines.append(f"  {u}")
        _print_section(11, "Source Extraction", lines,
                        elapsed_ms=elapsed_ms, warnings=warnings or None)

    # ── step 12  Resource usage snapshot ─────────────────────────────────────

    def step_resources(self):
        """
        PATCH — call dbg.step_resources() once at the end of get_reply()
        (just before summary()) to capture system state at response time.
        """
        if not DEBUG_RAG:
            return
        lines = [
            f"RAM Usage       : {_ram_mb()}",
            f"CPU Usage       : {_cpu_pct()}",
            f"Thread          : {_thread_name()}",
        ]
        _print_section(12, "Resource Usage", lines)

    # ── step 13  Evidence check ───────────────────────────────────────────────

    def step_evidence(self, chunk_count: int, decision: str,
                       elapsed_ms: float = 0.0):
        """
        PATCH — in api_adapter.py evidence check block, replace
        _debug_section(9, "Evidence Check", ...) with:
            dbg.step_evidence(len(top_chunks), "Proceed" or "Fallback", t)
        """
        self._record("Evidence", elapsed_ms)
        warnings = []
        if chunk_count == 0:
            warnings.append("Zero evidence chunks — LLM call skipped, fallback response sent.")
        elif chunk_count < 3:
            warnings.append(f"Only {chunk_count} chunk(s) retrieved — possible low-confidence answer.")
        _print_section(9, "Evidence Check", [
            f"Chunks Available : {chunk_count}",
            f"Decision         : {decision}",
        ], elapsed_ms=elapsed_ms, warnings=warnings or None)

    # ── step 14  Performance summary ─────────────────────────────────────────

    def summary(self):
        """
        PATCH — call dbg.summary() as the LAST thing before the final
        return statement in get_reply().
        """
        if not DEBUG_RAG:
            return
        total_ms = _now_ms() - self._t0
        print(f"\n{_SEP}")
        print("  PERFORMANCE SUMMARY")
        print(_SEP)
        stage_order = [
            "Intent", "Followup", "Category", "Dense", "BM25",
            "RRF", "Rerank", "Evidence", "Prompt", "Generation",
            "PostProcess", "Sources",
        ]
        for stage in stage_order:
            ms = self._timings.get(stage)
            if ms is not None:
                bar = "█" * min(40, int(ms / 10))
                print(f"  {stage:<14s} : {ms:>8.2f} ms  {bar}")
        print(_LINE)
        # stages not yet individually timed appear as unlisted
        unlisted = {k: v for k, v in self._timings.items() if k not in stage_order}
        for stage, ms in unlisted.items():
            print(f"  {stage:<14s} : {ms:>8.2f} ms")
        print(_LINE)
        print(f"  {'TOTAL':<14s} : {total_ms:>8.2f} ms")
        print(f"\n  RAM      : {_ram_mb()}")
        print(f"  CPU      : {_cpu_pct()}")
        print(f"  Thread   : {_thread_name()}")
        print(_SEP)


# ──────────────────────────────────────────────────────────────────────────────
# Minimal timing context manager (convenience, used in patched code below)
# ──────────────────────────────────────────────────────────────────────────────

class _Timer:
    """Context manager: `with _Timer() as t: ...; elapsed = t.ms`"""
    def __enter__(self):
        self._start = _now_ms()
        return self
    def __exit__(self, *_):
        self.ms = _now_ms() - self._start

timer = _Timer  # exported alias so call sites can do: with timer() as t:


# ──────────────────────────────────────────────────────────────────────────────
# Standalone warning printer (used by hybrid_retriever.py where no RagDebugger
# instance is available — e.g. BM25 index build, cross-encoder load failure)
# ──────────────────────────────────────────────────────────────────────────────

def warn(message: str):
    """Print a standalone warning line. Always gated by DEBUG_RAG."""
    if not DEBUG_RAG:
        return
    print(f"\n  {_WARN}: {message}")