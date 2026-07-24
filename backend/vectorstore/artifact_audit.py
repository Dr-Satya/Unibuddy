"""
artifact_audit.py — Remaining artifact audit on gdgu_clean.json and gdgu_chunks_final.json
Read-only. Does NOT modify any file.
"""
import json, re
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent.parent

# ── Load ──────────────────────────────────────────────────────────────────
with open(BASE / "data/processed/gdgu_clean.json", encoding="utf-8") as f:
    pages = json.load(f)

chunks_path = BASE / "data/processed/gdgu_chunks_final.json"
chunks = []
if chunks_path.exists():
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

# ── Patterns ──────────────────────────────────────────────────────────────
patterns = {
    "markdown_link":   re.compile(r'\[([^\]]{1,60})\]\((https?://[^)]+)\)'),
    "image_markdown":  re.compile(r'!\[([^\]]*)\]\([^)]+\)'),
    "broken_url":      re.compile(r'https?://[^\s)]*https?://[^\s)]*'),
    "footer_text":     re.compile(r'Sterco Digitex|Website Design and Development', re.I),
    "cta_text":        re.compile(r'\b(Apply Now|Download Brochure|Read More|View More|Download PDF|Click Here|Enquire Now|Get Brochure)\b', re.I),
    "social_media":    re.compile(r'facebook\.com|twitter\.com|instagram\.com|linkedin\.com|youtube\.com|x\.com/[a-z]', re.I),
}

SEP = "=" * 65

def audit_dataset(items, text_field, label):
    stats = {k: {"count": 0, "affected": 0, "examples": []} for k in patterns}
    for item in items:
        text = item.get(text_field, "") or ""
        url  = item.get("url", item.get("page_url", ""))
        for name, pat in patterns.items():
            hits = pat.findall(text)
            flat = [h if isinstance(h, str) else h[0] for h in hits]
            if flat:
                stats[name]["count"]    += len(flat)
                stats[name]["affected"] += 1
                if len(stats[name]["examples"]) < 3:
                    m = pat.search(text)
                    stats[name]["examples"].append({
                        "url":     url[:80],
                        "match":   m.group(0)[:120] if m else "",
                        "context": text[max(0,m.start()-30):m.end()+60].replace("\n"," ") if m else "",
                    })
    print(f"\n{SEP}")
    print(f"  {label}  ({len(items)} items)")
    print(SEP)
    print(f"  {'Artifact':<25} {'Count':>7}  {'Affected':>9}  {'Affected %':>10}")
    print(f"  {'-'*25} {'-'*7}  {'-'*9}  {'-'*10}")
    total = len(items)
    for name, s in stats.items():
        pct = s["affected"] / total * 100 if total else 0
        print(f"  {name:<25} {s['count']:>7,}  {s['affected']:>9,}  {pct:>9.1f}%")
    print()
    for name, s in stats.items():
        if s["examples"]:
            print(f"  ── {name} examples ──")
            for ex in s["examples"]:
                print(f"    URL: {ex['url']}")
                print(f"    CTX: {ex['context'][:110]}")
            print()
    return stats

stats_pages  = audit_dataset(pages,  "content", "gdgu_clean.json (pages)")
stats_chunks = audit_dataset(chunks, "text",    "gdgu_chunks_final.json (chunks)")

# ── Verdict ───────────────────────────────────────────────────────────────
print(SEP)
print("  VERDICT")
print(SEP)
issues = []
if stats_pages["footer_text"]["count"] > 0:
    issues.append(f"Footer text: {stats_pages['footer_text']['count']} occurrences on {stats_pages['footer_text']['affected']} pages")
if stats_pages["broken_url"]["count"] > 0:
    issues.append(f"Broken URLs: {stats_pages['broken_url']['count']} occurrences on {stats_pages['broken_url']['affected']} pages")
if stats_pages["cta_text"]["count"] > 100:
    issues.append(f"CTA noise: {stats_pages['cta_text']['count']} occurrences (high)")
if stats_pages["markdown_link"]["count"] > 500:
    issues.append(f"Markdown links: {stats_pages['markdown_link']['count']} survive (may add noise)")
if stats_pages["image_markdown"]["count"] > 100:
    issues.append(f"Image markdown: {stats_pages['image_markdown']['count']} survive")

if issues:
    print("  Another cleaning pass is RECOMMENDED for:")
    for i in issues:
        print(f"    • {i}")
else:
    print("  Data is clean enough to proceed to chunking/embedding.")
print()
