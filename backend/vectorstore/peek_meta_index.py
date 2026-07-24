import json
from pathlib import Path
f = Path(__file__).parent / 'metadata_index.json'
with open(f, 'r', encoding='utf-8') as fh:
    d = json.load(fh)
print('Total entries:', len(d))
print('Keys:', list(d[0].keys()))
print('Sample entry:')
import pprint; pprint.pprint(d[0])
