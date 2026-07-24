import json
from pathlib import Path
f = Path(__file__).parent.parent / 'data/processed/gdgu_embeddings.json'
with open(f, 'r', encoding='utf-8') as fh:
    d = json.load(fh)
print("Total:", len(d))
print("Keys:", list(d[0].keys()))
print("metadata keys:", list(d[0]['metadata'].keys()))
print("Sample:")
import pprint; pprint.pprint({k: v if k != 'embedding' else f'[{len(v)} floats]' for k, v in d[0].items()})
