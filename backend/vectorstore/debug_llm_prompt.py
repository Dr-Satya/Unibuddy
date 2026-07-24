"""
debug_llm_prompt.py — temporary LLM prompt debugger
=====================================================
Intercepts the prompt sent to the LLM for the query "MBA fee"
and prints every diagnostic without modifying any production file.

Technique: monkey-patches api_adapter.models (the model manager)
to capture the exact prompt before it reaches the LLM, then lets
the real LLM call proceed normally so the response is also printed.

Does NOT modify api_adapter.py, threaded_rag.py, or any other file.

Usage (run from backend/ directory):
    python vectorstore/debug_llm_prompt.py
"""

import os
import sys
import time

# ---------------------------------------------------------------------------
# Set DEBUG_RAG so existing _debug_section() calls also fire
# ---------------------------------------------------------------------------
os.environ["DEBUG_RAG"] = "true"

# Ensure imports resolve from backend/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEP  = "=" * 75
THIN = "-" * 75

print(SEP)
print("  LLM PROMPT DEBUGGER  —  query: 'MBA fee'")
print(SEP)

# ---------------------------------------------------------------------------
# Import the production module (this triggers RAG init + index load)
# ---------------------------------------------------------------------------
print("\n[1/4] Importing api_adapter (triggers RAG + model init)...")
import src.api_adapter as adapter

# ---------------------------------------------------------------------------
# Monkey-patch: intercept model.generate() to capture prompt + response
# ---------------------------------------------------------------------------
_intercepted = {}

_model_manager = adapter.models

# Find the live model object (Groq / whatever is loaded)
_live_model = (
    _model_manager.models.get("groq-llama")
    or next(iter(_model_manager.models.values()), None)
)

if _live_model is None:
    print("ERROR: No model loaded in models.models — check .env / model init.")
    sys.exit(1)

_original_generate = _live_model.generate

def _intercepting_generate(prompt, **kwargs):
    """Wraps the real generate() to capture prompt + response + timing."""
    _intercepted["prompt"]  = prompt
    _intercepted["kwargs"]  = kwargs
    _intercepted["t_start"] = time.time()
    result = _original_generate(prompt, **kwargs)
    _intercepted["t_end"]   = time.time()
    _intercepted["result"]  = result
    return result

_live_model.generate = _intercepting_generate
print(f"  Model intercepted: {type(_live_model).__name__}")

# ---------------------------------------------------------------------------
# Also intercept ThreadedRAGSystem.get_candidates to capture chunk scores
# ---------------------------------------------------------------------------
_rag_system = adapter.rag
_original_get_candidates = _rag_system.get_candidates

def _intercepting_get_candidates(query, top_k=None, category="general"):
    candidates = _original_get_candidates(query, top_k=top_k, category=category)
    _intercepted["candidates"] = candidates
    _intercepted["query"]      = query
    _intercepted["category"]   = category
    return candidates

_rag_system.get_candidates = _intercepting_get_candidates
print("  RAG get_candidates intercepted.")

# Also intercept HybridRetriever to capture top_chunks after merge
_hybrid = adapter.hybrid
_original_get_hybrid = _hybrid.get_hybrid_candidates

def _intercepting_get_hybrid(*args, **kwargs):
    result = _original_get_hybrid(*args, **kwargs)
    _intercepted["hybrid_result"] = result
    return result

_hybrid.get_hybrid_candidates = _intercepting_get_hybrid
print("  HybridRetriever intercepted.\n")

# ---------------------------------------------------------------------------
# Run the query
# ---------------------------------------------------------------------------
QUERY = "MBA fee"
print(f"[2/4] Calling get_reply({QUERY!r})...\n")
t0 = time.time()
reply = adapter.get_reply(QUERY, session_id="debug_session_001")
total_time = time.time() - t0

# ---------------------------------------------------------------------------
# Print diagnostic report
# ---------------------------------------------------------------------------
print("\n" + SEP)
print("  DIAGNOSTIC REPORT")
print(SEP)

# 1. Retrieved chunks
cands = _intercepted.get("candidates", [])
hybrid_result = _intercepted.get("hybrid_result", [])
print(f"\n[1] RETRIEVED CHUNKS")
print(THIN)
print(f"  Dense candidates (get_candidates): {len(cands)}")
print(f"  Hybrid merged candidates          : {len(hybrid_result)}")
if cands:
    print(f"  Chunk scores (dense, top-10):")
    for i, c in enumerate(cands[:10], 1):
        title = (c.get("metadata", {}).get("title", "") or "")[:45]
        ct    = c.get("metadata", {}).get("content_type", "?")
        print(f"    [{i:02d}] score={c.get('score', 0):.4f}  [{ct:<12}]  {title}")

# 2. Full context string
prompt = _intercepted.get("prompt", "")
ctx_start = prompt.find("Retrieved Context:\n")
ctx_end   = prompt.find("\nConversation History:")
if ctx_start != -1 and ctx_end != -1:
    full_context = prompt[ctx_start + len("Retrieved Context:\n"):ctx_end]
else:
    full_context = "(could not extract context block)"

print(f"\n[2] FULL CONTEXT STRING PASSED TO LLM")
print(THIN)
print(f"  Context length: {len(full_context)} chars")
print(f"  Context:\n")
print(full_context[:4000])
if len(full_context) > 4000:
    print(f"  ... [truncated — total {len(full_context)} chars]")

# 3. Full prompt
print(f"\n[3] FULL PROMPT SENT TO LLM")
print(THIN)
print(f"  Prompt length : {len(prompt)} chars")
# Rough token estimate (GPT-style: ~4 chars/token)
est_tokens = len(prompt) // 4
print(f"  Est. tokens   : ~{est_tokens}")
print(f"  Prompt:\n")
print(prompt[:5000])
if len(prompt) > 5000:
    print(f"\n  ... [prompt truncated at 5000 chars — total {len(prompt)} chars]")

# 4. Truncation check
MAX_TOKENS = 1200  # from api_adapter get_reply call
print(f"\n[4] TRUNCATION CHECK")
print(THIN)
print(f"  LLM max_tokens setting: {_intercepted.get('kwargs', {}).get('max_tokens', 'unknown')}")
if len(prompt) > 12000:
    print(f"  WARNING: prompt is {len(prompt)} chars (~{len(prompt)//4} tokens) — may exceed context window")
else:
    print(f"  Prompt size OK ({len(prompt)} chars)")

# 5. LLM response
result = _intercepted.get("result")
raw_text = ""
if result is not None:
    raw_text = getattr(result, "content", None) or getattr(result, "text", "") or str(result)
gen_time = 0.0
if "t_start" in _intercepted and "t_end" in _intercepted:
    gen_time = _intercepted["t_end"] - _intercepted["t_start"]

print(f"\n[5] LLM RESPONSE")
print(THIN)
print(f"  Generation time : {gen_time:.2f}s")
print(f"  Response length : {len(raw_text)} chars")
print(f"  Raw LLM output  :\n")
print(raw_text[:3000] if raw_text else "(empty — model may not have been called)")
if len(raw_text) > 3000:
    print(f"  ... [truncated — total {len(raw_text)} chars]")

# 6. Final reply returned by get_reply()
print(f"\n[6] FINAL REPLY RETURNED TO FRONTEND")
print(THIN)
final_reply = reply.get("reply", "") if isinstance(reply, dict) else str(reply)
print(f"  Reply length : {len(final_reply)} chars")
print(f"  Sources      : {reply.get('sources', []) if isinstance(reply, dict) else '?'}")
print(f"  Final reply  :\n{final_reply[:2000]}")
if len(final_reply) > 2000:
    print(f"  ... [truncated — total {len(final_reply)} chars]")

print(f"\n[7] TIMING")
print(THIN)
print(f"  Total get_reply() time : {total_time:.2f}s")
print(f"  LLM generation time    : {gen_time:.2f}s")
print(f"  Retrieval time         : {total_time - gen_time:.2f}s")

print(f"\n{SEP}")
print("  DEBUG COMPLETE")
print(SEP)
