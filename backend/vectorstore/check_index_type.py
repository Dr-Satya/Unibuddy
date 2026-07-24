import faiss, json
from pathlib import Path
idx = faiss.read_index(str(Path(__file__).parent / 'gdgu.index'))
print("Index type :", type(idx).__name__)
print("ntotal     :", idx.ntotal)
print("d          :", idx.d)

# Check id_map structure
with open(Path(__file__).parent / 'id_map.json', 'r', encoding='utf-8') as f:
    m = json.load(f)
print("\nid_map top keys:", [k for k in m.keys() if k != 'rows'])
print("rows[0]:", m['rows'][0])
print("rows[1]:", m['rows'][1])
