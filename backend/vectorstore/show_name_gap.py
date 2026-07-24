import json
from pathlib import Path

f = Path(__file__).parent / 'faculty_coverage_report.json'
with open(f, 'r', encoding='utf-8') as fh:
    r = json.load(fh)

old_kb = r['old_kb_comparison']
print("=== OLD NAMES NOT IN NEW KB (first 30) ===")
for n in old_kb['sample_names_only_old'][:30]:
    print(f"  {n}")

print("\n=== NEW NAMES NOT IN OLD KB (first 30) ===")
for n in old_kb['sample_names_only_new'][:30]:
    print(f"  {n}")

print("\n=== TOP 30 NAMED PERSONS IN NEW KB ===")
for entry in r['faculty_names']['top_30_names']:
    print(f"  {entry['count']:>3}x  {entry['name']}")

print("\n=== SCHOOLS — old vs new named persons ===")
for school, data in old_kb['school_level_comparison'].items():
    print(f"  {school[:45]:<45}  "
          f"old={data['old_named_persons']:>3}  "
          f"new={data['new_named_persons']:>3}  "
          f"status={data['coverage_status']}")

print("\n=== VERDICT ===")
v = r['verdict']
print(f"  {v['verdict']}")
print(f"  {v['key_finding']}")
