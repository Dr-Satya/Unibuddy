"""Sample old metadata.json to understand its structure."""
import json
from pathlib import Path

f = Path(__file__).parent.parent / 'data/vectordb/metadata.json'
with open(f, 'r', encoding='utf-8') as fh:
    data = json.load(fh)

print(f"Type: {type(data)}")
if isinstance(data, dict):
    keys = list(data.keys())
    print(f"Total keys: {len(keys)}")
    print(f"First 5 keys: {keys[:5]}")
    sample = data[keys[0]]
    print(f"\nSample record fields: {list(sample.keys())}")
    import pprint; pprint.pprint(sample)
    print("\n--- Second record ---")
    pprint.pprint(data[keys[1]])
    print("\n--- 50th record ---")
    pprint.pprint(data[keys[49]])
elif isinstance(data, list):
    print(f"Total records: {len(data)}")
    import pprint; pprint.pprint(data[0])

# Collect all unique field names across all records
if isinstance(data, dict):
    all_fields = set()
    for v in data.values():
        if isinstance(v, dict):
            all_fields.update(v.keys())
    print(f"\nAll unique fields across all records: {sorted(all_fields)}")

    # Count content_types
    from collections import Counter
    ct_counts = Counter(v.get('content_type','') for v in data.values())
    print(f"\ncontent_type distribution: {ct_counts.most_common()}")

    # Count non-empty fields
    field_fill = {f: 0 for f in all_fields}
    for v in data.values():
        for f in all_fields:
            if v.get(f):
                field_fill[f] += 1
    print(f"\nField fill rate (out of {len(data)}):")
    for f, n in sorted(field_fill.items(), key=lambda x:-x[1]):
        print(f"  {f:<20} {n:>5}  ({n/len(data)*100:.0f}%)")
