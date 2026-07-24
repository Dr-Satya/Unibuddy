import json, faiss
with open('data/vectordb/metadata.json', encoding='utf-8') as f:
    old = json.load(f)
print('OLD KB fields:', list(old['0'].keys()))
print('OLD KB total :', len(old))

with open('vectorstore/retrieval_metadata.json', encoding='utf-8') as f:
    new = json.load(f)
print('NEW KB fields:', list(new['0'].keys()))
print('NEW KB total :', len(new))

old_idx = faiss.read_index('data/vectordb/index.faiss')
new_idx = faiss.read_index('vectorstore/gdgu.index')
print('OLD index:', type(old_idx).__name__, 'ntotal:', old_idx.ntotal, 'dim:', old_idx.d)
print('NEW index:', type(new_idx).__name__, 'ntotal:', new_idx.ntotal, 'dim:', new_idx.d)
