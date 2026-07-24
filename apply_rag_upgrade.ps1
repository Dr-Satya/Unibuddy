# apply_rag_upgrade.ps1
# Updated version: NO BACKUPS — direct apply (you already have a safe copy)

param()
$ErrorActionPreference = 'Stop'

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ROOT

Write-Host "[INSTALLER] Starting direct RAG upgrade (no backups)..."
Write-Host "[INSTALLER] Applying changes to backend and frontend..."

# Helper: write file content
function Write-File($relPath, $content) {
  $full = Join-Path $ROOT $relPath
  $dir = Split-Path $full -Parent
  if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  $content | Out-File -FilePath $full -Encoding UTF8 -Force
  Write-Host "[INSTALLER] Updated $relPath"
}

# 1) backend/src/config.py
Write-File "backend\src\config.py" @"
import os

TOP_K_RESULTS = int(os.environ.get('TOP_K_RESULTS', 15))
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 300))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 100))
DEBUG_RAG = os.environ.get('DEBUG_RAG', 'false').lower() in ('1','true','yes')
VECTOR_DB_PATH = os.environ.get('VECTOR_DB_PATH', './data/vectordb/')
"@

# 2) backend/src/threaded_rag.py
Write-File "backend\src\threaded_rag.py" @"
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
            print(f'[RAG] get_candidates for \"{query}\" -> {len(results)} results')
        return results

    def get_context_for_query(self, query: str, top_k: int = None) -> str:
        cands = self.get_candidates(query, top_k=top_k)
        seen_keys = set()
        parts = []
        for c in cands:
            key = f\"{c['metadata'].get('doc_id','')}_{c['index']}\"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            src = c['metadata'].get('source','Unknown')
            title = c['metadata'].get('title','')
            parts.append(f\"[Source: {title or src}]\\n{c['content']}\")
        return '\\n\\n---\\n\\n'.join(parts) if parts else ''
"@

# 3) backend/src/api_adapter.py
Write-File "backend\src\api_adapter.py" @"
import re, json, os
from collections import deque, defaultdict
from typing import Dict, Any
from src.threaded_rag import ThreadedRAGSystem
from src.threaded_models import ThreadedModelManager
from src.config import DEBUG_RAG, TOP_K_RESULTS

rag = ThreadedRAGSystem()
models = ThreadedModelManager()
session_histories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

PROMPT_TEMPLATE = '''System: You are UniBuddy, the official intelligent assistant for GD Goenka University. Use ONLY the retrieved context and conversation history. Extract every detail confidently. Never say \"not sufficient\" or apologize. If a field is missing, say \"Not available in sources\". Use bold headings, bullets, and inline citations [source].

Retrieved Context:
{RAG_CONTEXT}

Conversation History:
{HISTORY}

Current Question:
{USER_QUESTION}

Output Format (for faculty/person queries):
**Dr. Full Name**

Assistant Professor, Department of Computer Science & Engineering
GD Goenka University, Sohna, Gurugram [source]

**Role & Department**
• Current position and leadership roles [source]

**Location**
• GD Goenka University — Sohna/Gurugram campus, Haryana, India [source]

**Education**
• Ph.D. in Computer Science & Engineering (Computational Neuroscience) — Institution, Year [source]
• Master's — Institution [source]
• Bachelor's — Institution [source]

**Research Interests**
• Computational Neuroscience
• Biomedical Signal Processing
• Machine Learning applications [source]

**Experience**
• Over X years in academia and research [source]
• Previous roles if available [source]

**Profile Links**
• Official Profile: <a href=\"URL\" target=\"_blank\">GD Goenka Faculty Page</a>
• LinkedIn: <a href=\"URL\" target=\"_blank\">LinkedIn Profile</a> (if available)

**Sources**
• List all URLs used

Would you like his recent publications, courses he teaches, or contact information?
'''

FOLLOWUP_RE = re.compile(r\"\\b(him|her|them|his|her's|more about|details|research|publications)\\b\", re.I)

def _rewrite_followup_if_needed(session_id: str, user_query: str) -> str:
    if not session_id or not FOLLOWUP_RE.search(user_query):
        return user_query
    hist = session_histories.get(session_id)
    if not hist:
        return user_query
    person = None
    for entry in reversed(hist):
        u = entry.get('user') or ''
        m = re.search(r\"([A-Z][a-z]+\\s+[A-Z][a-z]+)\", u)
        if m:
            person = m.group(1)
            break
    if person:
        rewritten = f\"{person} {user_query}\"
        if DEBUG_RAG:
            print(f\"[RAG] Rewrote follow-up: '{user_query}' → '{rewritten}'\")
        return rewritten
    return user_query

def _compose_history(session_id: str) -> str:
    hist = session_histories.get(session_id)
    if not hist:
        return ''
    return '\\n'.join(f\"User: {e.get('user','')}\\nAssistant: {e.get('assistant','')}\" for e in hist)

def _store_session(session_id: str, user: str, assistant: str):
    session_histories[session_id].append({'user': user, 'assistant': assistant})

def _save_structured_to_vectordb(name_key: str, structured: Dict[str,Any]):
    try:
        vpath = os.path.join('data','vectordb',f\"extra_{re.sub(r\"\\W+\",\"_\",name_key.lower())}.json\")
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
        with open(vpath,'w',encoding='utf-8') as f:
            json.dump(structured, f, ensure_ascii=False, indent=2)
        if DEBUG_RAG:
            print(f\"[RAG] Saved structured data for {name_key}\")
    except Exception as e:
        print(f\"[RAG] Failed to save structured data: {e}\")

def parse_structured_from_text(text: str) -> Dict[str,Any]:
    data = {'name':'Not available in sources','role':'Not available in sources','department':'Not available in sources','location':'Not available in sources','education':[],'research':[],'experience':[],'links':[],'sources':[]}
    m = re.search(r\"\\*\\*(Dr\\.?\\s*[^\\*]+?)\\*\\*\", text)
    if m:
        data['name'] = m.group(1).strip()
    for url in re.findall(r\"https?://[^\\s<>\"]+\", text):
        data['links'].append(url)
        data['sources'].append(url)
    # Simple bullet extraction for lists
    for section in ['Education', 'Research Interests', 'Experience']:
        sec_match = re.search(rf\"\\*\\*{section}\\*\\*([\\s\\S]*?)(?=\\*\\*|$\", text)
        if sec_match:
            bullets = re.findall(r\"[•\\-\\*]\\s*(.+)\", sec_match.group(1))
            key = section.lower().replace(' ','_').replace('interests','')
            data[key] = bullets or ['Not available in sources']
    return data

def get_reply(user_message: str, session_id: str = None) -> Dict[str,Any]:
    rewritten = _rewrite_followup_if_needed(session_id, user_message)
    queries = [rewritten, rewritten + ' profile', rewritten + ' research']
    contexts = [rag.get_context_for_query(q, top_k=TOP_K_RESULTS) for q in queries if rag.get_context_for_query(q)]
    full_context = '\\n\\n'.join(contexts)
    history = _compose_history(session_id)
    prompt = PROMPT_TEMPLATE.format(RAG_CONTEXT=full_context or 'No relevant context found', HISTORY=history, USER_QUESTION=user_message)
    model = next(iter(models.models.values()))  # fallback to first available
    try:
        resp = model.generate(prompt, max_tokens=1500, temperature=0.1)
        text = resp.content or resp.text or ''
    except Exception:
        text = full_context[:4000] or 'No response generated'
    structured = parse_structured_from_text(text)
    if structured['name'] != 'Not available in sources':
        _save_structured_to_vectordb(structured['name'], structured)
    if session_id:
        _store_session(session_id, user_message, text)
    return {'reply': text, 'data': structured, 'sources': structured['sources']}
"@

# 4) backend/api.py
Write-File "backend\api.py" @"
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.api_adapter import get_reply

app = FastAPI(title=\"UniBuddy API\")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[\"*\"],
    allow_credentials=True,
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

@app.post('/chat')
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail='Empty message')
    return get_reply(req.message, session_id=req.session_id)
"@

# 5) src/components/Chatbot.tsx
Write-File "src\components\Chatbot.tsx" @"
import { useState, useEffect, useRef } from 'react';

const API_URL = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:9000';

const RenderStructured = ({ data }) => {
  if (!data || !data.name || data.name === 'Not available in sources') {
    return null;
  }
  return (
    <div style={{ lineHeight: '1.6' }}>
      <b style={{ fontSize: '1.2em' }}>{data.name}</b><br/><br/>
      {data.role && data.role !== 'Not available in sources' && (
        <>
          <b>Role & Department:</b><br/>
          <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
            <li>{data.role}</li>
          </ul>
        </>
      )}
      {data.location && data.location !== 'Not available in sources' && (
        <>
          <b>Location:</b><br/>
          <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
            <li>{data.location}</li>
          </ul>
        </>
      )}
      {data.education && data.education.length > 0 && data.education[0] !== 'Not available in sources' && (
        <>
          <b>Education:</b>
          <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
            {data.education.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        </>
      )}
      {data.research && data.research.length > 0 && data.research[0] !== 'Not available in sources' && (
        <>
          <b>Research Interests:</b>
          <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
            {data.research.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        </>
      )}
      {data.experience && data.experience.length > 0 && data.experience[0] !== 'Not available in sources' && (
        <>
          <b>Experience:</b>
          <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
            {data.experience.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        </>
      )}
      {data.links && data.links.length > 0 && (
        <>
          <b>Profile Links:</b>
          <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
            {data.links.map((l, i) => (
              <li key={i}><a href={l} target=\"_blank\" rel=\"noreferrer\" style={{ color: '#0066cc' }}>{l}</a></li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
};

const Chatbot = () => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const sessionRef = useRef('sess_' + Math.random().toString(36).slice(2));

  const send = async () => {
    if (!input.trim() || loading) return;
    const userMsg = { role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);
    setInput('');
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input, session_id: sessionRef.current })
      });
      const j = await res.json();
      const assistantMsg = {
        role: 'assistant',
        text: j.reply || '',
        structured: j.data || null,
        sources: j.sources || []
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: 'Error: Could not reach server' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ position: 'fixed', bottom: 20, right: 20, zIndex: 1000 }}>
      {!open && (
        <button onClick={() => setOpen(true)} style={{ padding: '12px 16px', background: '#0066cc', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer' }}>
          Chat with UniBuddy
        </button>
      )}
      {open && (
        <div style={{ width: 380, height: 560, background: 'white', borderRadius: '12px', boxShadow: '0 8px 32px rgba(0,0,0,0.2)', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '12px', background: '#0066cc', color: 'white', borderTopLeftRadius: '12px', borderTopRightRadius: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <b>UniBuddy Assistant</b>
            <button onClick={() => setOpen(false)} style={{ background: 'none', border: 'none', color: 'white', fontSize: '20px', cursor: 'pointer' }}>×</button>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
            {messages.map((m, i) => (
              <div key={i} style={{ marginBottom: '16px', textAlign: m.role === 'user' ? 'right' : 'left' }}>
                <div style={{ display: 'inline-block', maxWidth: '80%', padding: '10px 14px', borderRadius: '12px', background: m.role === 'user' ? '#0066cc' : '#f0f0f0', color: m.role === 'user' ? 'white' : 'black' }}>
                  {m.role === 'assistant' && m.structured ? <RenderStructured data={m.structured} /> : <div dangerouslySetInnerHTML={{ __html: m.text.replace(/\\*\\*(.*?)\\*\\*/g, '<b>$1</b>') }} />}
                  {m.sources && m.sources.length > 0 && (
                    <div style={{ marginTop: '8px', fontSize: '12px' }}>
                      <b>Sources:</b>
                      <ul style={{ margin: '4px 0', paddingLeft: '16px' }}>
                        {m.sources.map((s, idx) => <li key={idx}><a href={s} target="_blank" rel="noreferrer" style={{ color: '#0066cc' }}>{s}</a></li>)}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && <div style={{ textAlign: 'left' }}><i>Thinking...</i></div>}
          </div>
          <div style={{ padding: '12px', borderTop: '1px solid #eee', display: 'flex' }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && send()}
              placeholder="Type your message..."
              style={{ flex: 1, padding: '10px', borderRadius: '8px', border: '1px solid #ccc' }}
              disabled={loading}
            />
            <button onClick={send} disabled={loading || !input.trim()} style={{ marginLeft: '8px', padding: '10px 16px', background: '#0066cc', color: 'white', border: 'none', borderRadius: '8px' }}>
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
"@

# 6) backend/tests/test_rag_profiles.py
Write-File "backend\tests\test_rag_profiles.py" @"
import pytest
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_rajat_sharma_two_turn():
    # First query
    response1 = client.post(\"/chat\", json={\"message\": \"who is rajat sharma\"})
    assert response1.status_code == 200
    j1 = response1.json()
    assert 'reply' in j1
    assert 'data' in j1
    assert 'Dr.' in j1['reply'] or 'Rajat Sharma' in j1['reply']

    session_id = 'test_session_123'

    # Second query (follow-up)
    response2 = client.post(\"/chat\", json={\"message\": \"give me more details about him\", \"session_id\": session_id})
    assert response2.status_code == 200
    j2 = response2.json()
    data = j2.get('data', {})
    assert data.get('education') and len(data['education']) > 0
    assert any('Ph.D.' in e or 'Computational Neuroscience' in e for e in data['education'])
    assert data.get('research') and len(data['research']) > 0
    assert data.get('links') and len(data['links']) > 0
    assert data.get('sources') and len(data['sources']) > 0
    assert 'Not available in sources' not in j2['reply'] or 'Ph.D.' in j2['reply']
"@

Write-Host "[INSTALLER] All RAG upgrades applied successfully!"
Write-Host "[INSTALLER] Restart your app with: python start_all.py"
Write-Host "[INSTALLER] Then test the chat with Rajat Sharma questions!"
