import json
with open('vectorstore/retrieval_metadata.json', encoding='utf-8') as f:
    meta = json.load(f)
for k, v in meta.items():
    src   = (v.get('source','') or '').lower()
    title = (v.get('title','') or '').lower()
    if 'yogesh' in title or 'yogesh' in src:
        print(f"row={k:>5}  chunk_id={v.get('chunk_id','')[:16]}  "
              f"title={v.get('title','')}  type={v.get('content_type','')}")
        print(f"         source={v.get('source','')}")
print("---")
# Also check amit kumar chhabra
for k, v in meta.items():
    src   = (v.get('source','') or '').lower()
    title = (v.get('title','') or '').lower()
    if 'chhabra' in title or 'chhabra' in src:
        print(f"row={k:>5}  chunk_id={v.get('chunk_id','')[:16]}  "
              f"title={v.get('title','')}  type={v.get('content_type','')}")
        print(f"         source={v.get('source','')}")
