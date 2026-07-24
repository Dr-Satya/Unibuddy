"""
src/rag_debug.py — lightweight RAG debug logging helpers.

This module is intentionally small and import-safe. All output is gated by
DEBUG_RAG so production behavior stays unchanged when debugging is off.
"""

import os
import re
import time
import threading
from typing import Any, Dict, List, Optional

from src.config import DEBUG_RAG

try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:  # pragma: no cover
    _psutil = None
    _PSUTIL = False

_SEP = "=" * 65
_LINE = "-" * 65
_WARN = "WARNING"


def _now_ms() -> float:
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
    return clean if len(clean) <= max_chars else clean[:max_chars] + "…"


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def _print_section(
    step_num: int,
    title: str,
    lines: List[str],
    elapsed_ms: Optional[float] = None,
    warnings: Optional[List[str]] = None,
):
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
        for warning in warnings:
            print(f"  {_WARN}: {warning}")
    print(_SEP)


class RagDebugger:
    def __init__(self, query: str = ""):
        self._query = query
        self._t0 = _now_ms()
        self._timings: Dict[str, float] = {}

    def _record(self, stage: str, elapsed_ms: float):
        self._timings[stage] = elapsed_ms

    def step_intent(self, user_query: str, detected: str, elapsed_ms: float = 0.0):
        self._record("Intent", elapsed_ms)
        _print_section(1, "Intent Detection", [
            f"Query           : {_preview(user_query)}",
            f"Detected Intent : {detected}",
        ], elapsed_ms=elapsed_ms)

    def step_followup(self, original: str, rewritten: str, focus_field: Optional[str], elapsed_ms: float = 0.0):
        self._record("Followup", elapsed_ms)
        _print_section(2, "Follow-up Detection", [
            f"Original Query  : {_preview(original)}",
            f"Rewritten Query : {_preview(rewritten)}",
            f"Focus Field     : {focus_field or '(none)'}",
        ], elapsed_ms=elapsed_ms)

    def step_category(self, category: str, is_list_query: bool = False, elapsed_ms: float = 0.0):
        self._record("Category", elapsed_ms)
        _print_section(3, "Category Classification", [
            f"Category        : {category}",
            f"List/Aggregate   : {is_list_query}",
        ], elapsed_ms=elapsed_ms)

    def step_dense(self, query: str, candidates: List[Dict[str, Any]], category: str, elapsed_ms: float = 0.0):
        self._record("Dense", elapsed_ms)
        warnings = []
        if not candidates:
            warnings.append("FAISS returned zero candidates.")
        lines = [
            f"Query           : {_preview(query)}",
            f"Category Filter : {category}",
            f"Results         : {len(candidates)}",
            _LINE,
            "Top Chunks:",
        ]
        for i, c in enumerate(candidates[:10], 1):
            meta = c.get("metadata", {}) or {}
            lines.append(
                f"  [{i:02d}] score={c.get('score', 0.0):.4f} | exact={c.get('exact_match', 0)} | "
                f"title={(meta.get('title') or '')[:40]} | source={(meta.get('source') or '')[:80]}"
            )
        _print_section(4, "Dense Retrieval", lines, elapsed_ms=elapsed_ms, warnings=warnings or None)

    def step_bm25(self, query: str, tokenized: List[str], candidates: List[Dict[str, Any]], category: str, elapsed_ms: float = 0.0):
        self._record("BM25", elapsed_ms)
        warnings = []
        if not candidates:
            warnings.append("BM25 returned zero results — no lexical overlap with corpus.")
        lines = [
            f"Query           : {_preview(query)}",
            f"Tokenized Query : {tokenized[:20]} … ({len(tokenized)} tokens)",
            f"Category Filter : {category}",
            f"Results         : {len(candidates)}",
            _LINE,
            "Top BM25 Chunks:",
        ]
        for i, c in enumerate(candidates[:10], 1):
            lines.append(f"  [{i:02d}] score={c.get('score', 0.0):.4f} | preview: {_preview(c.get('content', ''), 150)}")
        _print_section(5, "BM25 Retrieval", lines, elapsed_ms=elapsed_ms, warnings=warnings or None)

    def step_rrf(self, dense_hits: int, bm25_hits: int, fused: List[Dict[str, Any]], deduped_count: int, rrf_k: int = 60, elapsed_ms: float = 0.0):
        self._record("RRF", elapsed_ms)
        warnings = []
        if deduped_count > 0:
            warnings.append(f"{deduped_count} duplicate(s) removed after RRF fusion.")
        lines = [
            f"Dense Input    : {dense_hits} chunks",
            f"BM25 Input     : {bm25_hits} chunks",
            f"RRF k          : {rrf_k}",
            f"Fused Total    : {len(fused)}",
            f"Duplicates Removed : {deduped_count}",
            _LINE,
            "Combined Ranking (top 10):",
        ]
        for i, c in enumerate(fused[:10], 1):
            lines.append(f"  [{i:02d}] rrf={c.get('rrf_score', 0.0):.5f} | doc={str(c.get('metadata', {}).get('doc_id', '?'))[:24]} | {_preview(c.get('content', ''), 100)}")
        _print_section(6, "Hybrid RRF Fusion", lines, elapsed_ms=elapsed_ms, warnings=warnings or None)

    def step_rerank(self, before: List[Dict[str, Any]], after: List[Dict[str, Any]], reranker_available: bool, elapsed_ms: float = 0.0):
        self._record("Rerank", elapsed_ms)
        warnings = []
        if not reranker_available:
            warnings.append("Cross-encoder NOT loaded — using RRF order (graceful degradation).")
        lines = [
            f"Reranker        : {'cross-encoder/ms-marco-MiniLM-L-6-v2' if reranker_available else 'UNAVAILABLE — RRF passthrough'}",
            f"Candidates In   : {len(before)}",
            f"Candidates Out  : {len(after)}",
            f"Removed         : {len(before) - len(after)} (below keep threshold)",
            _LINE,
            "Before Rerank:",
        ]
        for i, c in enumerate(before[:10], 1):
            lines.append(f"  [{i:02d}] pre_score={c.get('rrf_score', c.get('score', 0.0)):.4f} | {_preview(c.get('content', ''), 100)}")
        lines += [_LINE, "After Rerank:"]
        for i, c in enumerate(after[:10], 1):
            lines.append(f"  [{i:02d}] ce_score={c.get('_evidence_score', c.get('rrf_score', 0.0)):.4f} | {_preview(c.get('content', ''), 100)}")
        _print_section(7, "Cross-Encoder Reranking", lines, elapsed_ms=elapsed_ms, warnings=warnings or None)

    def step_prompt(self, prompt: str, template_name: str, full_context: str, history: str, user_question: str, elapsed_ms: float = 0.0):
        self._record("Prompt", elapsed_ms)
        tokens_est = _token_estimate(prompt)
        warnings = []
        if tokens_est > 3500:
            warnings.append(f"Prompt is large ({tokens_est} est. tokens) — may hit context limit.")
        _print_section(8, "Prompt Building", [
            f"Template        : {template_name}",
            f"Prompt Length   : {len(prompt)} chars",
            f"Est. Tokens     : {tokens_est}",
            f"  Context Tokens: {_token_estimate(full_context)}",
            f"  History Tokens: {_token_estimate(history)}",
            f"  Query Tokens   : {_token_estimate(user_question)}",
        ], elapsed_ms=elapsed_ms, warnings=warnings or None)

    def step_generation(self, raw_text: str, elapsed_ms: float = 0.0):
        self._record("Generation", elapsed_ms)
        _print_section(9, "Generation", [
            f"Raw Output Preview: {_preview(raw_text, 200)}",
        ], elapsed_ms=elapsed_ms)

    def step_postprocess(self, raw_text: str, structured: Dict[str, Any], html_reply: str, elapsed_ms: float = 0.0):
        self._record("PostProcess", elapsed_ms)
        _print_section(10, "Post Processing", [
            f"Structured Parsing: {structured}",
            f"Final HTML Preview : {_preview(html_reply, 400)}",
        ], elapsed_ms=elapsed_ms)

    def step_sources(self, urls: List[str], elapsed_ms: float = 0.0):
        self._record("Sources", elapsed_ms)
        _print_section(11, "Sources", [f"Extracted URLs: {urls}"], elapsed_ms=elapsed_ms)

    def summary(self):
        total_ms = _now_ms() - self._t0
        if not DEBUG_RAG:
            return
        print(f"\n{_SEP}")
        print("  DEBUG SUMMARY")
        print(_SEP)
        for stage, ms in self._timings.items():
            print(f"  {stage:<14s} : {ms:>8.2f} ms")
        print(_LINE)
        print(f"  {'TOTAL':<14s} : {total_ms:>8.2f} ms")
        print(f"  RAM      : {_ram_mb()}")
        print(f"  CPU      : {_cpu_pct()}")
        print(f"  Thread   : {_thread_name()}")
        print(_SEP)


class _Timer:
    def __enter__(self):
        self._start = _now_ms()
        return self

    def __exit__(self, *_):
        self.ms = _now_ms() - self._start


timer = _Timer


def warn(message: str):
    if not DEBUG_RAG:
        return
    print(f"\n  {_WARN}: {message}")

