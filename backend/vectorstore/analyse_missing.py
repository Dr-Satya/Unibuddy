"""Analyse the missing URLs in detail — separate real pages from artefacts."""
import json, re
from collections import Counter
from pathlib import Path

with open(Path(__file__).parent / 'crawl_coverage_audit.json', encoding='utf-8') as f:
    data = json.load(f)

all_missing = []
for cat, items in data['missing_urls'].items():
    for item in items:
        item['category'] = cat
        all_missing.append(item)

print(f"Total missing entries reported: {len(all_missing)}")

# Separate malformed URLs from real page URLs
def is_malformed(url):
    return ('<' in url or '\\' in url or ' ' in url or
            url.endswith('.pdf') or url.endswith('.zip') or
            'siteassets' in url or 'uploads/' in url)

real_missing    = [u for u in all_missing if not is_malformed(u['url'])]
artefact_missing= [u for u in all_missing if is_malformed(u['url'])]

print(f"Real page URLs missing  : {len(real_missing)}")
print(f"Artefact/PDF URLs       : {len(artefact_missing)}")

# Category breakdown of real missing pages
cat_counts = Counter(u['category'] for u in real_missing)
print(f"\nReal missing by category:")
for cat, n in cat_counts.most_common():
    print(f"  {cat:<22} {n:>4}")

# Sample 10 real missing per important category
for cat in ['courses','happenings','schools','faculty','placements','research','other']:
    items = [u for u in real_missing if u['category'] == cat][:8]
    if items:
        print(f"\n  Sample {cat} missing URLs:")
        for i in items:
            print(f"    {i['url']}")

# Check if happenings pages are news/events (not RAG-relevant)
happ = [u for u in real_missing if u['category'] == 'happenings']
print(f"\nHappenings sample (first 10):")
for u in happ[:10]:
    print(f"  {u['url']}")

# Course pages breakdown
courses = [u for u in real_missing if u['category'] == 'courses']
print(f"\nCourses missing ({len(courses)}) — first 15:")
for u in courses[:15]:
    print(f"  {u['url']}")

# Faculty missing
faculty = [u for u in real_missing if u['category'] == 'faculty']
print(f"\nFaculty missing ({len(faculty)}) — all:")
for u in faculty:
    print(f"  {u['url']}")
