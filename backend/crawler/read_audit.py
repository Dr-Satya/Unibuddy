import json
from pathlib import Path

with open(Path(__file__).parent / 'coverage_audit_report.json', encoding='utf-8') as f:
    r = json.load(f)

s = r['summary']
print("=== SUMMARY ===")
print(f"Sitemap URLs       : {s['total_sitemap_urls']}")
print(f"Found in crawl     : {s['found_in_crawl']} ({s['coverage_pct']}%)")
print(f"Missing from crawl : {s['missing_from_crawl']}")
print("\nReason breakdown:")
for k, v in sorted(s['reason_breakdown'].items(), key=lambda x: -x[1]):
    print(f"  {k:<45} {v}")
print("\nMissing by category:")
for k, v in sorted(s['missing_by_category'].items(), key=lambda x: -x[1]):
    print(f"  {k:<25} {v}")

print("\n=== MISSING URLS WITH EVIDENCE ===")
for m in r['missing']:
    print(f"\nURL      : {m['url']}")
    print(f"Category : {m['category']}")
    print(f"Reason   : {m['reason']}")
    print(f"Evidence : {m['evidence']}")
