"""
faculty_audit.py — Faculty Coverage Audit (read-only, no modifications)
Reads: data/processed/gdgu_chunks_final.json
       data/vectordb/metadata.json  (old KB, for comparison)
Writes: vectorstore/faculty_coverage_report.json
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent

NEW_CHUNKS_FILE = BACKEND_DIR / "data" / "processed" / "gdgu_chunks_final.json"
OLD_META_FILE   = BACKEND_DIR / "data" / "vectordb"  / "metadata.json"
REPORT_FILE     = SCRIPT_DIR  / "faculty_coverage_report.json"

# ── Load datasets ─────────────────────────────────────────────────────────────
with open(NEW_CHUNKS_FILE, "r", encoding="utf-8") as f:
    new_chunks = json.load(f)

with open(OLD_META_FILE, "r", encoding="utf-8") as f:
    old_meta_raw = json.load(f)
old_chunks = list(old_meta_raw.values())

# ── Keyword sets ──────────────────────────────────────────────────────────────
FACULTY_URL_KEYWORDS = [
    "faculty", "people", "staff", "teacher", "academic",
    "department", "school", "dean", "director", "hod",
]

TITLE_KEYWORDS = re.compile(
    r'\b(assistant professor|associate professor|professor|dean|'
    r'head of department|director|faculty|dr\.|phd|ph\.d)\b',
    re.IGNORECASE
)

NAME_RE = re.compile(
    r'\b(Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}|'
    r'Prof\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}|'
    r'Mr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}|'
    r'Ms\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}|'
    r'Mrs\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'
)

SCHOOL_PATTERNS = {
    "School of Engineering & Sciences":    re.compile(r'school.of.engineering|engineering.sciences?', re.I),
    "School of Management":                re.compile(r'school.of.management|management', re.I),
    "School of Law":                       re.compile(r'school.of.law|\blaw\b', re.I),
    "School of Liberal Arts":              re.compile(r'liberal.arts|school.of.liberal', re.I),
    "School of Healthcare & Allied Sci":   re.compile(r'healthcare|allied.science|pharmacy|physiotherapy|nursing|optometry|psychology', re.I),
    "School of Agricultural Sciences":     re.compile(r'agricultur|agronomy|horticulture|floriculture', re.I),
    "School of Hospitality & Tourism":     re.compile(r'hospitality|tourism|culinary', re.I),
    "UID – School of Design":              re.compile(r'design|fashion|communication.design|interior', re.I),
    "Centre for Excellence (OHSFE)":       re.compile(r'ohsfe|occupational.health|fire.safety|safety.engineering|environmental.engineering', re.I),
    "Centre of Aerospace & Energy":        re.compile(r'aerospace|energy.engineering', re.I),
    "Research & Development":              re.compile(r'research.and.development|phd.scholar|publication', re.I),
}

FACULTY_NAME_BLACKLIST = {
    "Dr University", "Dr Goenka", "Dr School", "Dr College",
    "Prof University", "Mr University", "Ms University",
    "Dr Campus", "Dr Faculty", "Dr Department",
}

# ═══════════════════════════════════════════════════════════════════════
# 1. URL COVERAGE
# ═══════════════════════════════════════════════════════════════════════
all_urls = list({c["page_url"] for c in new_chunks})

faculty_urls = []
for url in all_urls:
    url_lower = url.lower()
    if any(kw in url_lower for kw in FACULTY_URL_KEYWORDS):
        faculty_urls.append(url)

# School-wise URL grouping
school_urls = defaultdict(list)
for url in faculty_urls:
    matched = False
    for school, pat in SCHOOL_PATTERNS.items():
        if pat.search(url):
            school_urls[school].append(url)
            matched = True
    if not matched:
        school_urls["Other / General"].append(url)

print(f"[1] Faculty-related URLs: {len(faculty_urls)} / {len(all_urls)}")

# ═══════════════════════════════════════════════════════════════════════
# 2. CHUNK COVERAGE — keyword presence in text
# ═══════════════════════════════════════════════════════════════════════
faculty_chunks     = []
non_faculty_chunks = []

for chunk in new_chunks:
    text = chunk.get("text", "")
    if TITLE_KEYWORDS.search(text):
        faculty_chunks.append(chunk)
    else:
        non_faculty_chunks.append(chunk)

pct_faculty = len(faculty_chunks) / len(new_chunks) * 100
print(f"[2] Faculty keyword chunks: {len(faculty_chunks)} / {len(new_chunks)}  ({pct_faculty:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════
# 3. NAMED PERSON DETECTION
# ═══════════════════════════════════════════════════════════════════════
all_names = []
for chunk in new_chunks:
    text = chunk.get("text", "")
    for m in NAME_RE.finditer(text):
        name = m.group(0).strip()
        # Filter noise
        if name in FACULTY_NAME_BLACKLIST:
            continue
        if len(name.split()) < 2:
            continue
        all_names.append(name)

name_counts  = Counter(all_names)
unique_names = list(name_counts.keys())
top_names    = name_counts.most_common(30)

print(f"[3] Unique named persons detected: {len(unique_names)}")

# ═══════════════════════════════════════════════════════════════════════
# 4. DEPARTMENT / SCHOOL COVERAGE
# ═══════════════════════════════════════════════════════════════════════
school_chunk_counts = defaultdict(int)
school_page_urls    = defaultdict(set)
school_names_found  = defaultdict(set)

for chunk in new_chunks:
    text     = chunk.get("text",     "")
    url      = chunk.get("page_url", "")
    combined = text + " " + url

    for school, pat in SCHOOL_PATTERNS.items():
        if pat.search(combined):
            school_chunk_counts[school] += 1
            school_page_urls[school].add(url)
            # Collect names found within this school's chunks
            for m in NAME_RE.finditer(text):
                name = m.group(0).strip()
                if len(name.split()) >= 2 and name not in FACULTY_NAME_BLACKLIST:
                    school_names_found[school].add(name)

school_summary = {}
for school in SCHOOL_PATTERNS:
    school_summary[school] = {
        "chunk_count":     school_chunk_counts[school],
        "page_count":      len(school_page_urls[school]),
        "faculty_names_found": sorted(school_names_found[school])[:20],
        "faculty_name_count":  len(school_names_found[school]),
        "sample_urls":     sorted(school_page_urls[school])[:5],
    }

print(f"[4] Schools with chunk coverage:")
for school, data in school_summary.items():
    print(f"    {school:<45} chunks={data['chunk_count']:>4}  "
          f"pages={data['page_count']:>3}  names={data['faculty_name_count']:>3}")

# ═══════════════════════════════════════════════════════════════════════
# 5. METADATA COVERAGE
# ═══════════════════════════════════════════════════════════════════════
# Where does faculty info live in new KB?
faculty_in_title    = sum(1 for c in new_chunks if TITLE_KEYWORDS.search(c.get("page_title","") or ""))
faculty_in_text     = sum(1 for c in new_chunks if TITLE_KEYWORDS.search(c.get("text","") or ""))
faculty_in_url      = sum(1 for c in new_chunks if any(kw in (c.get("page_url","") or "").lower() for kw in FACULTY_URL_KEYWORDS))
faculty_in_category = sum(1 for c in new_chunks if (c.get("category","") or "").lower() == "faculty")

metadata_coverage = {
    "faculty_keyword_in_page_title": faculty_in_title,
    "faculty_keyword_in_text":       faculty_in_text,
    "faculty_keyword_in_url":        faculty_in_url,
    "category_equals_faculty":       faculty_in_category,
    "note": (
        "New KB does not have a dedicated 'faculty' category. "
        "Faculty information is embedded inside 'course', 'about', "
        "and 'research' category chunks."
    )
}

print(f"[5] Faculty info in page_title: {faculty_in_title}  in_text: {faculty_in_text}  "
      f"in_url: {faculty_in_url}  category=faculty: {faculty_in_category}")

# ═══════════════════════════════════════════════════════════════════════
# 6. MISSING SCHOOLS
# ═══════════════════════════════════════════════════════════════════════
missing_schools = [
    school for school, data in school_summary.items()
    if data["chunk_count"] == 0
]
zero_faculty_names = [
    school for school, data in school_summary.items()
    if data["faculty_name_count"] == 0
]

print(f"[6] Schools with zero chunks: {missing_schools}")
print(f"    Schools with zero named faculty: {zero_faculty_names}")

# ═══════════════════════════════════════════════════════════════════════
# 7. COMPARE AGAINST OLD KB — by category coverage (no text similarity)
# ═══════════════════════════════════════════════════════════════════════
# Old KB faculty categories and what they covered
old_faculty_cats = {
    "faculty_profile": [c for c in old_chunks if c.get("content_type") == "faculty_profile"],
    "faculty":         [c for c in old_chunks if c.get("content_type") == "faculty"],
}

# Old KB: unique source URLs per category
old_faculty_urls = {
    cat: list({c.get("source","") for c in chunks})
    for cat, chunks in old_faculty_cats.items()
}

# Old KB: unique named persons
old_names_all = []
for c in old_chunks:
    if c.get("content_type") in ("faculty_profile", "faculty"):
        name = c.get("name", "").strip()
        if name:
            old_names_all.append(name)
old_unique_names = list(set(old_names_all))

# New KB: unique person names
new_unique_names = list(name_counts.keys())

# Overlap between old named persons and new named persons (exact)
old_names_norm = {n.lower().strip() for n in old_unique_names}
new_names_norm = {n.lower().strip() for n in new_unique_names}
names_in_both  = old_names_norm & new_names_norm
names_only_old = old_names_norm - new_names_norm
names_only_new = new_names_norm - old_names_norm

# Old KB schools (from source URLs)
old_school_urls = defaultdict(set)
for c in old_chunks:
    if c.get("content_type") in ("faculty_profile", "faculty"):
        src = c.get("source", "")
        for school, pat in SCHOOL_PATTERNS.items():
            if pat.search(src):
                old_school_urls[school].add(src)

# Coverage decision per school
old_vs_new = {}
for school in SCHOOL_PATTERNS:
    old_pages = len(old_school_urls[school])
    new_pages = len(school_page_urls[school])
    old_names_s = len([c for c in old_chunks
                       if c.get("content_type") in ("faculty_profile","faculty")
                       and SCHOOL_PATTERNS[school].search(c.get("source",""))])
    new_names_s = school_summary[school]["faculty_name_count"]

    if new_pages == 0 and old_pages == 0:
        status = "not_in_either"
    elif new_pages == 0 and old_pages > 0:
        status = "not_present_in_new"
    elif new_names_s >= old_names_s * 0.7:
        status = "covered"
    elif new_names_s > 0:
        status = "partially_covered"
    else:
        status = "not_present_in_new"

    old_vs_new[school] = {
        "old_faculty_pages":  old_pages,
        "new_pages":          new_pages,
        "old_named_persons":  old_names_s,
        "new_named_persons":  new_names_s,
        "coverage_status":    status,
    }

old_kb_comparison = {
    "old_faculty_profile_count": len(old_faculty_cats["faculty_profile"]),
    "old_faculty_count":         len(old_faculty_cats["faculty"]),
    "old_unique_named_persons":  len(old_unique_names),
    "new_unique_named_persons":  len(new_unique_names),
    "names_in_both":             len(names_in_both),
    "names_only_in_old":         len(names_only_old),
    "names_only_in_new":         len(names_only_new),
    "sample_names_only_old":     sorted(names_only_old)[:20],
    "sample_names_only_new":     sorted(names_only_new)[:20],
    "school_level_comparison":   old_vs_new,
}

print(f"[7] Old unique named persons: {len(old_unique_names)}")
print(f"    New unique named persons: {len(new_unique_names)}")
print(f"    Names in both:            {len(names_in_both)}")
print(f"    Names only in old:        {len(names_only_old)}")
print(f"    Names only in new:        {len(names_only_new)}")

# ═══════════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════════
not_present = [s for s, d in old_vs_new.items() if d["coverage_status"] == "not_present_in_new"]
partial     = [s for s, d in old_vs_new.items() if d["coverage_status"] == "partially_covered"]
covered     = [s for s, d in old_vs_new.items() if d["coverage_status"] == "covered"]

if len(not_present) == 0 and len(partial) == 0:
    verdict = "1. Faculty coverage complete"
elif len(not_present) == 0 and len(partial) <= 2:
    verdict = "2. Faculty coverage mostly complete"
elif len(not_present) <= 3:
    verdict = "3. Faculty coverage partial"
else:
    verdict = "4. Faculty pages missing from crawl"

verdict_evidence = {
    "verdict": verdict,
    "schools_covered":         covered,
    "schools_partially_covered": partial,
    "schools_not_in_new_kb":   not_present,
    "named_persons_recovered": f"{len(names_in_both)}/{len(old_unique_names)} old names found in new KB",
    "total_faculty_chunks_new": len(faculty_chunks),
    "total_faculty_chunks_old": len(old_faculty_cats["faculty_profile"]) + len(old_faculty_cats["faculty"]),
    "key_finding": (
        f"New KB has {len(faculty_chunks)} chunks containing faculty keywords "
        f"and {len(unique_names)} named persons. "
        f"Old KB had {len(old_unique_names)} named persons explicitly tagged. "
        f"New KB recovered {len(names_in_both)} of those names inside chunk text. "
        f"{len(not_present)} schools have zero faculty coverage in new KB."
    )
}

# ═══════════════════════════════════════════════════════════════════════
# WRITE REPORT
# ═══════════════════════════════════════════════════════════════════════
report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "faculty_urls": {
        "total_urls_in_new_kb":       len(all_urls),
        "faculty_related_url_count":  len(faculty_urls),
        "faculty_url_list":           sorted(faculty_urls),
        "school_wise_grouping":       {k: sorted(v) for k, v in school_urls.items()},
    },
    "faculty_chunks": {
        "total_chunks":              len(new_chunks),
        "chunks_with_faculty_keywords": len(faculty_chunks),
        "pct_of_total":              round(pct_faculty, 1),
        "sample_chunk_urls":         list({c["page_url"] for c in faculty_chunks})[:20],
    },
    "faculty_names": {
        "total_unique_names":   len(unique_names),
        "top_30_names":         [{"name": n, "count": cnt} for n, cnt in top_names],
        "all_unique_names":     sorted(unique_names),
    },
    "schools": school_summary,
    "missing_schools": {
        "zero_chunk_coverage":        missing_schools,
        "zero_named_faculty":         zero_faculty_names,
    },
    "metadata_coverage": metadata_coverage,
    "old_kb_comparison": old_kb_comparison,
    "verdict": verdict_evidence,
}

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

size_kb = REPORT_FILE.stat().st_size / 1024

print(f"\n{'='*65}")
print(f"  FACULTY COVERAGE AUDIT COMPLETE")
print(f"{'='*65}")
print(f"  Verdict: {verdict}")
print(f"  Report:  {REPORT_FILE.resolve()}  ({size_kb:.1f} KB)")
print(f"{'='*65}\n")
