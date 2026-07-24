"""
inspect_sync.py — Phase 7c
Reads vectorstore/sync_report.json and prints a human-readable report.

Usage:
    python vectorstore/inspect_sync.py
    python vectorstore/inspect_sync.py --report path/to/sync_report.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
REPORT_FILE = SCRIPT_DIR / "sync_report.json"


def main():
    parser = argparse.ArgumentParser(description="Inspect sync_report.json")
    parser.add_argument("--report", type=str, default=str(REPORT_FILE))
    args   = parser.parse_args()
    path   = Path(args.report)

    if not path.exists():
        print(f"ERROR: {path} not found. Run compare_embeddings.py first.")
        raise SystemExit(1)

    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    s = report["summary"]

    print(f"\n{'='*65}")
    print("  Phase 7 — Sync Inspection Report")
    print(f"{'='*65}")
    print(f"\n  Generated at  : {report['generated_at']}")
    print(f"  Old source    : {report['old_source']}")
    print(f"  New source    : {report['new_source']}")

    # ── High-level summary ────────────────────────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  SUMMARY")
    print(f"  {'─'*55}")
    print(f"  Total old chunks    : {s['total_old']:>6}")
    print(f"  Total new chunks    : {s['total_new']:>6}")
    print(f"  Unchanged           : {s['unchanged']:>6}")
    print(f"  Modified            : {s['modified']:>6}")
    print(f"  Deleted             : {s['deleted']:>6}")
    print(f"  Added (new)         : {s['new']:>6}")
    print(f"  % changed           : {s['pct_changed']:>5.1f}%")

    # ── New chunks breakdown ──────────────────────────────────────────────────
    new_chunks = report.get("new_chunks", [])
    if new_chunks:
        cat_new = Counter(c["category"] for c in new_chunks)
        print(f"\n  {'─'*55}")
        print(f"  NEW CHUNKS by category ({len(new_chunks)} total)")
        print(f"  {'─'*55}")
        for cat, n in cat_new.most_common():
            print(f"  {cat:<16} {n:>4}")
        print(f"\n  Sample new chunks (first 5):")
        for c in new_chunks[:5]:
            print(f"    [{c['category']:<12}]  {c['page_title'][:40]:<40}"
                  f"  {c['url'][:55]}")

    # ── Modified chunks breakdown ─────────────────────────────────────────────
    mod_chunks = report.get("modified_chunks", [])
    if mod_chunks:
        cat_mod = Counter(c["category"] for c in mod_chunks)
        content_changed  = sum(1 for c in mod_chunks if c.get("content_changed"))
        embedding_changed= sum(1 for c in mod_chunks if c.get("embedding_changed"))
        print(f"\n  {'─'*55}")
        print(f"  MODIFIED CHUNKS ({len(mod_chunks)} total)")
        print(f"  {'─'*55}")
        print(f"  Content changed     : {content_changed}")
        print(f"  Embedding changed   : {embedding_changed}")
        print(f"  By category:")
        for cat, n in cat_mod.most_common():
            print(f"    {cat:<16} {n:>4}")
        print(f"\n  Sample modified chunks (first 5):")
        for c in mod_chunks[:5]:
            flags = []
            if c.get("content_changed"):   flags.append("CONTENT")
            if c.get("embedding_changed"): flags.append("EMBEDDING")
            print(f"    [{c['category']:<12}]  {'+'.join(flags):<20}"
                  f"  {c['page_title'][:35]}")

    # ── Deleted chunks breakdown ──────────────────────────────────────────────
    del_chunks = report.get("deleted_chunks", [])
    if del_chunks:
        cat_del = Counter(c["category"] for c in del_chunks)
        print(f"\n  {'─'*55}")
        print(f"  DELETED CHUNKS ({len(del_chunks)} total)")
        print(f"  {'─'*55}")
        for cat, n in cat_del.most_common():
            print(f"  {cat:<16} {n:>4}")
        print(f"\n  Sample deleted chunks (first 5):")
        for c in del_chunks[:5]:
            print(f"    [{c['category']:<12}]  {c['page_title'][:40]:<40}"
                  f"  {c['url'][:55]}")

    # ── Unchanged breakdown ───────────────────────────────────────────────────
    unch = report.get("unchanged_chunks", [])
    cat_unch = Counter(c["category"] for c in unch)
    print(f"\n  {'─'*55}")
    print(f"  UNCHANGED CHUNKS by category ({len(unch)} total)")
    print(f"  {'─'*55}")
    for cat, n in cat_unch.most_common():
        pct = n / s["total_new"] * 100 if s["total_new"] else 0
        bar = "█" * (n // 20)
        print(f"  {cat:<16} {n:>5}  ({pct:4.1f}%)  {bar}")

    # ── Action recommendation ─────────────────────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  RECOMMENDED ACTION")
    print(f"  {'─'*55}")
    if s["modified"] == 0 and s["deleted"] == 0 and s["new"] == 0:
        print("  ✓ Index is fully up to date. No rebuild needed.")
    else:
        needs = []
        if s["new"] > 0:
            needs.append(f"add {s['new']} new vectors")
        if s["modified"] > 0:
            needs.append(f"replace {s['modified']} modified vectors")
        if s["deleted"] > 0:
            needs.append(f"remove {s['deleted']} deleted vectors")
        print(f"  Next step: {', '.join(needs)}.")
        print(f"  Only {s['pct_changed']:.1f}% of chunks changed —",
              "incremental update is sufficient." if s['pct_changed'] < 30
              else "consider a full rebuild.")

    # ── Report file size ──────────────────────────────────────────────────────
    size_kb = path.stat().st_size / 1024
    print(f"\n  {'─'*55}")
    print(f"  REPORT FILE")
    print(f"  {'─'*55}")
    print(f"  File size           : {size_kb:.1f} KB")
    print(f"  Path                : {path.resolve()}")
    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
