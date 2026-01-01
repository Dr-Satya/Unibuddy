import os, json, threading
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, DEBUG_RAG

class ThreadedRAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.vector_db = None
        self.index_to_doc_map = {}
        self.lock = threading.Lock()
        self._initialize()

    def _initialize(self):
        if self.embedding_model:
            return
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f'[RAG] embedding init error: {e}')
        os.makedirs(os.environ.get('VECTOR_DB_PATH','./data/vectordb/'), exist_ok=True)
        idx_path = os.path.join(os.environ.get('VECTOR_DB_PATH','./data/vectordb/'),'index.faiss')
        meta_path = os.path.join(os.environ.get('VECTOR_DB_PATH','./data/vectordb/'),'metadata.json')
        if os.path.exists(idx_path) and os.path.exists(meta_path):
            try:
                self.vector_db = faiss.read_index(idx_path)
                with open(meta_path,'r',encoding='utf-8') as f:
                    self.index_to_doc_map = json.load(f)
                print(f'[RAG] Loaded index with {self.vector_db.ntotal} vectors')
            except Exception as e:
                print(f'[RAG] failed to load index: {e}')
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        dim = 384  # all-MiniLM-L6-v2 dimension
        self.vector_db = faiss.IndexFlatL2(dim)
        self.index_to_doc_map = {}
        print('[RAG] Created new FAISS index')

    def get_candidates(self, query: str, top_k: int = None) -> List[Dict[str,Any]]:
        top_k = top_k or TOP_K_RESULTS
        self._initialize()
        if not self.vector_db or self.vector_db.ntotal == 0:
            return []
        q_emb = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        D, I = self.vector_db.search(q_emb.reshape(1,-1), min(top_k, int(self.vector_db.ntotal)))
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.index_to_doc_map.get(str(idx), {})
            results.append({'score': float(score), 'index': int(idx), 'content': meta.get('content',''), 'metadata': meta})
        if DEBUG_RAG:
            print(f"[RAG] get_candidates for \"{query}\" -> {len(results)} results")
        return results

    def get_context_for_query(self, query: str, top_k: int = None) -> str:
        cands = self.get_candidates(query, top_k=top_k)
        seen_keys = set()
        parts = []
        for c in cands:
            key = f"{c['metadata'].get('doc_id','')}_{c['index']}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            src = c['metadata'].get('source','Unknown')
            title = c['metadata'].get('title','')
            parts.append(f"[Source: {title or src}]\n{c['content']}")
        return '\n\n---\n\n'.join(parts) if parts else ''
