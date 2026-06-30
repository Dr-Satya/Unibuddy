import re, json, os, hashlib
from collections import deque, defaultdict
from typing import Dict, Any, Optional, List
from src.threaded_rag import ThreadedRAGSystem, classify_query, _matches_category
from src.threaded_models import ThreadedModelManager
from src.hybrid_retriever import HybridRetriever
from src.config import DEBUG_RAG, TOP_K_RESULTS
from src.timetable_lookup import get_timetable_context
from src.timetable_store import answer_timetable_query, is_timetable_intent, has_active_timetable_session, is_timetable_escape
from src.mentor_store import answer_mentor_query, is_mentor_intent

rag = ThreadedRAGSystem()
# Phase-1 Hybrid RAG: wraps the existing `rag` instance (FAISS, category
# filtering, exact-match boosting, threshold filtering, dedup all untouched).
# `rag` itself is unmodified, so any other caller keeps identical behavior.
hybrid = HybridRetriever(rag)
models = ThreadedModelManager()
session_histories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

NO_EVIDENCE_FALLBACK = "I couldn't find enough information in the university knowledge base."


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or '').lower()).strip()


def _entity_key(metadata: Dict[str, Any]) -> str:
    name = _normalize_text(metadata.get('name'))
    if name:
        return name
    title = _normalize_text(metadata.get('title') or '')
    if title:
        # Normalize out generic page prefixes used in faculty profile titles.
        title = re.sub(r'^(faculty profile\s*-\s*)', '', title)
        title = re.sub(r'^(profile\s*-\s*)', '', title)
        return title
    doc_id = metadata.get('doc_id') or metadata.get('document_id')
    if doc_id:
        return str(doc_id)
    source = _normalize_text(metadata.get('source'))
    return source


def _is_menu_boilerplate(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True

    menu_signals = [
        'menu', 'about us', 'about gd goenka university', 'about gd goenka',
        'admissions', 'admission', 'fee structure', 'hostel & transport',
        'online application form', 'scholarship 2025-2026', 'vision and mission',
        'governance', 'message from the vice chancellor', 'organogram',
        'mandatory disclosures', 'recognitions and affiliations', 'rankings and awards',
        'regulatory committees', 'department of international partnerships',
        'online application', 'ph.d', 'phd', 'school of engineering',
    ]
    profile_signals = [
        'assistant professor', 'associate professor', 'professor', 'dr ',
        'research', 'qualification', 'experience', 'publication', 'faculty',
        'department', 'school', 'hod', 'head of department', 'postgraduate',
        'undergraduate', 'teaching', 'ph.d', 'phd', 'guided', 'supervisor'
    ]

    menu_count = sum(1 for term in menu_signals if term in normalized)
    profile_count = sum(1 for term in profile_signals if term in normalized)

    # Remove chunks that are clearly navigation/menu boilerplate and not real profile content.
    if menu_count >= 3 and profile_count == 0:
        return True
    if normalized.startswith('menu') and profile_count == 0:
        return True
    if 'about us' in normalized and profile_count == 0:
        return True
    if any(term in normalized for term in ['fee structure', 'online application form', 'scholarship 2025-2026']) and profile_count == 0:
        return True
    if len(normalized) < 300 and menu_count >= 2 and profile_count == 0:
        return True

    return False

PROMPT_TEMPLATE = '''System: You are UniBuddy, the official intelligent assistant for GD Goenka University. Use ONLY the retrieved context and conversation history. Answer concisely (2–6 sentences) and include sources. If a field is missing, say "Not available in sources".

Retrieved Context:
{RAG_CONTEXT}

Conversation History:
{HISTORY}

Current Question:
{USER_QUESTION}

Provide a short factual summary (2–6 sentences), then a structured block with headings: Role, Location, Education (short), Research Interests (short), Experience (short). Do NOT include a Sources or Links section.
'''

FOLLOWUP_RE = re.compile(
    r"\b(?:he|she|they|him|her|them|his|hers|their|theirs|tell me more|more about|details|research|publications|profile|where is|what does|what are)\b",
    re.I)

_NAME_BLACKLIST = {
    'university', 'campus', 'school', 'college', 'department', 'facility',
    'facilities', 'admission', 'admissions', 'course', 'courses',
    'program', 'programs', 'fee', 'fees', 'placement', 'hostel', 'library'
}


def _extract_person_name(text: str) -> Optional[str]:
    if not text:
        return None

    # Prefer explicit academic titles or full names.
    title_patterns = [
        r"\b(Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        r"\b(Prof(?:essor)?\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
    ]
    for pat in title_patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()

    # Fallback to a generic capitalized name candidate.
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
    for cand in reversed(candidates):
        lower = cand.lower()
        if not any(block in lower for block in _NAME_BLACKLIST):
            return cand.strip()

    return None


def _find_previous_person(session_id: str) -> Optional[str]:
    hist = session_histories.get(session_id)
    if not hist:
        return None

    for entry in reversed(hist):
        person = _extract_person_name(entry.get('user', '') or '')
        if person:
            return person
        person = _extract_person_name(entry.get('assistant', '') or '')
        if person:
            return person

    return None


def _rewrite_followup_if_needed(session_id: str, user_query: str) -> str:
    if not session_id or not FOLLOWUP_RE.search(user_query):
        return user_query

    # If the query already includes a person name or title, do not rewrite.
    if _extract_person_name(user_query):
        return user_query

    person = _find_previous_person(session_id)
    if person:
        rewritten = f"{person} {user_query}"
        if DEBUG_RAG:
            print(f"[RAG] Rewrote follow-up: '{user_query}' -> '{rewritten}'")
        return rewritten

    return user_query


def _compose_history(session_id: str) -> str:
    hist = session_histories.get(session_id)
    if not hist:
        return ''
    return '\n'.join(f"User: {e.get('user','')}\nAssistant: {e.get('assistant','')}" for e in hist)


def _store_session(session_id: str, user: str, assistant: str):
    # store sanitized versions to avoid debug fragments
    try:
        user_clean = re.sub(r"(?i)(?:<current_tab_state>[\s\S]*?</current_tab_state>|\\u003ccurrent_tab_state\\u003e[\s\S]*?\\u003c\\/current_tab_state\\u003e)", "", user or '').strip()
        assistant_clean = re.sub(r"(?i)(?:<current_tab_state>[\s\S]*?</current_tab_state>|\\u003ccurrent_tab_state\\u003e[\s\S]*?\\u003c\\/current_tab_state\\u003e)", "", assistant or '').strip()
        session_histories[session_id].append({'user': user_clean, 'assistant': assistant_clean})
    except Exception:
        session_histories[session_id].append({'user': user, 'assistant': assistant})


def _save_structured_to_vectordb(name_key: str, structured: Dict[str,Any]):
    try:
        safe = re.sub(r"\W+","_", name_key.lower())
        vpath = os.path.join('data','vectordb',f"extra_{safe}.json")
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
        with open(vpath,'w',encoding='utf-8') as f:
            json.dump(structured, f, ensure_ascii=False, indent=2)
        if DEBUG_RAG:
            print(f"[RAG] Saved structured data for {name_key} to {vpath}")
    except Exception as e:
        print(f"[RAG] Failed to save structured data: {e}")


def dedupe_sentences(text: str) -> str:
    parts = re.split(r'(?<=[.!?])\s+', (text or '').strip())
    seen = set()
    out = []
    for p in parts:
        s = re.sub(r'\s+', ' ', p).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return ' '.join(out)


def summarize_text(text: str, max_sentences: int = 5) -> str:
    text = dedupe_sentences(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= max_sentences:
        return ' '.join(sentences).strip()
    return ' '.join(sentences[:max_sentences]).strip()


def parse_structured_from_text(text: str) -> Dict[str,Any]:
    data = {'name':None,'role':None,'department':None,'location':None,'education':[],'research':[], 'experience':[], 'links':[], 'sources':[]}
    m = re.search(r"\*\*(Dr\.?\s*[^\*]+?)\*\*", text)
    if m:
        data['name'] = m.group(1).strip()
    else:
        m2 = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", text)
        if m2:
            data['name'] = m2.group(1)
    for url in re.findall(r"https?://[^\s<>\"]+", text):
        if url not in data['links']:
            data['links'].append(url)
            data['sources'].append(url)
    ed = re.search(r"Education[:\*\n\r]+([\s\S]{0,200})", text)
    if ed:
        data['education'] = [s.strip() for s in re.split(r'[\n\r\u2022\-]+', ed.group(1)) if s.strip()][:3]
    rs = re.search(r"Research Interests[:\*\n\r]+([\s\S]{0,200})", text)
    if rs:
        data['research'] = [s.strip() for s in re.split(r'[\n\r\u2022\-]+', rs.group(1)) if s.strip()][:5]
    ex = re.search(r"Experience[:\*\n\r]+([\s\S]{0,200})", text)
    if ex:
        data['experience'] = [s.strip() for s in re.split(r'[\n\r\u2022\-]+', ex.group(1)) if s.strip()][:5]
    return data


def _sanitize_visible_text(text: str) -> str:
    # remove debug tags
    text = re.sub(r"(?i)(?:<current_tab_state>[\s\S]*?</current_tab_state>|\\u003ccurrent_tab_state\\u003e[\s\S]*?\\u003c\\/current_tab_state\\u003e)", "", text)
    # remove inline [Source: ...] markers
    text = re.sub(r"\[Source:[^\]]+\]", "", text)
    # remove "Retrieved Context: ..." lines that the LLM echoes back
    text = re.sub(r"[-*•]\s*Retrieved Context:[^\n]*", "", text)
    text = re.sub(r"Retrieved Context:[^\n]*", "", text)
    # collapse repeated separators
    text = re.sub(r"(\s*---\s*)+", "\n---\n", text)
    # strip debug lines
    text = "\n".join(l for l in text.splitlines() if not re.search(r"^(Local Machine:|Open Widgets:)", l))
    return text.strip()


def _split_chunk_into_paragraphs(text: str) -> List[str]:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'(?i)\b(Menu|About Us|Admissions|Apply Now|Scholarship(?:s?)|Fee Structure|Contact Us|Home|Breadcrumbs|Navigation|Footer|Social|Cookie|Privacy Policy|Terms and Conditions|Instagram|Facebook|Twitter|YouTube|LinkedIn)\b', r'\n\1', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    clean_paragraphs: List[str] = []
    for para in paragraphs:
        if len(para) > 400 and re.search(r'(?i)\b(Menu|About Us|Admissions|Apply Now|Scholarship|Fee Structure|Contact Us|Home|Breadcrumbs|Navigation|Footer|Social|Cookie|Privacy Policy|Terms and Conditions|Instagram|Facebook|Twitter|YouTube|LinkedIn)\b', para):
            subparas = re.split(r'(?i)(?=\b(Menu|About Us|Admissions|Apply Now|Scholarship(?:s?)|Fee Structure|Contact Us|Home|Breadcrumbs|Navigation|Footer|Social|Cookie|Privacy Policy|Terms and Conditions|Instagram|Facebook|Twitter|YouTube|LinkedIn)\b)', para)
            clean_paragraphs.extend(p.strip() for p in subparas if p.strip())
        else:
            clean_paragraphs.append(para)
    return clean_paragraphs


def _is_boilerplate_paragraph(paragraph: str) -> bool:
    normalized = _normalize_text(paragraph)
    if not normalized:
        return True

    boilerplate_signals = [
        'menu', 'about us', 'admissions', 'apply now', 'scholarship',
        'fee structure', 'contact us', 'home', 'breadcrumb', 'navigation',
        'footer', 'social', 'cookie', 'privacy policy', 'terms and conditions',
        'facebook', 'twitter', 'instagram', 'youtube', 'linkedin',
        'follow us', 'share', 'copyright', 'terms', 'cookies'
    ]
    content_signals = [
        'university', 'faculty', 'professor', 'department', 'research',
        'qualification', 'experience', 'course', 'program', 'curriculum',
        'eligibility', 'placement', 'hostel', 'campus', 'students', 'education',
        'laboratory', 'library', 'admission criteria', 'scholarship amount',
        'published', 'teaching', 'training', 'career', 'events', 'programmes'
    ]

    nav_count = sum(1 for term in boilerplate_signals if term in normalized)
    content_count = sum(1 for term in content_signals if term in normalized)

    if normalized.startswith(('menu', 'about us', 'admissions', 'apply now',
                              'scholarship', 'fee structure', 'contact us',
                              'home', 'breadcrumb', 'navigation', 'footer',
                              'social', 'cookie', 'privacy policy', 'terms and conditions')):
        return True

    if any(normalized == term for term in ['menu', 'about us', 'admissions', 'apply now',
                                           'scholarship', 'fee structure', 'contact us',
                                           'home', 'breadcrumb', 'navigation', 'footer',
                                           'social', 'cookie']):
        return True

    if len(normalized.split()) <= 20 and nav_count >= 1 and content_count == 0:
        return True

    if nav_count >= 2 and content_count == 0 and '.' not in normalized:
        return True

    if re.fullmatch(r'[\w\s\-&,:/\.]+', normalized) and nav_count >= 2 and content_count <= 1:
        return True

    return False


def _clean_chunk_content(text: str) -> List[str]:
    paragraphs = _split_chunk_into_paragraphs(text)
    cleaned: List[str] = []
    for para in paragraphs:
        para = re.sub(r'\s+', ' ', para).strip()
        if not para:
            continue
        if _is_boilerplate_paragraph(para):
            continue
        cleaned.append(para)
    return cleaned


def get_reply(user_message: str, session_id: str = None, user_profile: Dict = None) -> Dict[str,Any]:
    # sanitize incoming user message (remove embedded debug fragments)
    if not user_message:
        user_message = ''
    user_message = re.sub(r"(?i)(?:<current_tab_state>[\s\S]*?</current_tab_state>|\\u003ccurrent_tab_state\\u003e[\s\S]*?\\u003c\\/current_tab_state\\u003e)", "", user_message).strip()

    # --- Routing priority ---
    # 1. If timetable session is mid-flow, let it handle everything first
    # 2. Otherwise, mentor queries take priority (to avoid 'how many' / 'faculty' clashes)
    # 3. Then timetable for fresh timetable queries
    # 4. Finally RAG

    history = list(session_histories.get(session_id, []))

    # If already in a timetable conversation, keep it there — unless it's an escape query
    if has_active_timetable_session(session_id) and not is_timetable_escape(user_message):
        timetable_answer = answer_timetable_query(user_message, session_id=session_id, history=history)
        if timetable_answer:
            if session_id:
                _store_session(session_id, user_message, timetable_answer)
            return {'reply': timetable_answer, 'data': {}, 'sources': []}

    # Mentor-Mentee fast-path (before timetable to avoid 'how many'/'faculty' clashes)
    if is_mentor_intent(user_message):
        mentor_answer = answer_mentor_query(user_message)
        if mentor_answer:
            if session_id:
                _store_session(session_id, user_message, mentor_answer)
            return {'reply': mentor_answer, 'data': {}, 'sources': []}

    # Timetable fast-path for fresh queries
    timetable_answer = answer_timetable_query(user_message, session_id=session_id, history=history)
    if timetable_answer:
        if session_id:
            _store_session(session_id, user_message, timetable_answer)
        return {'reply': timetable_answer, 'data': {}, 'sources': []}
    # -------------------------------------------------------------------------

    # Build student context block if profile provided
    student_context = ""
    timetable_context = ""
    if user_profile:
        name     = user_profile.get("name", "")
        degree   = user_profile.get("degree", "")
        branch   = user_profile.get("branch", "")
        year     = user_profile.get("year", "")
        section  = user_profile.get("section", "")
        email    = user_profile.get("email", "")

        student_context = f"""Student Info:
- Name: {name}
- Email: {email}
- Degree: {degree} {branch} Year {year}
- Section: {section}"""

        # Inject timetable if query is about schedule
        timetable_context = get_timetable_context(section, user_message)
        if timetable_context:
            student_context += f"\n\n{timetable_context}"

    rewritten = _rewrite_followup_if_needed(session_id, user_message)

    # -------------------------------------------------------------------------
    # Category detection
    # -------------------------------------------------------------------------
    # Classify the original user message (before follow-up rewriting) so
    # "his research" → rewritten "Dr X his research" doesn't change the
    # detected category, which should already be 'faculty' from context.
    category = classify_query(user_message)

    if DEBUG_RAG:
        print(f"[RAG] Detected Category: {category}")
        print(f"[RAG] Rewritten query  : {rewritten}")

    # -------------------------------------------------------------------------
    # Category-aware query expansion
    # -------------------------------------------------------------------------
    # For faculty queries:   add " profile" and " research" variants to
    #                         broaden recall across bio + publications chunks.
    # For fee queries:        add " annual fee structure" and " admission charges"
    #                         to target fee table rows more precisely.
    # For admission queries:  add " eligibility criteria" variant.
    # For course queries:     add " curriculum" and " specialization" variants.
    # For university_info:    add " GD Goenka" variant for general info pages.
    # For general:            keep the original two generic variants.
    if category == 'faculty':
        variant_suffixes = [' profile', ' research']
    elif category == 'fee':
        variant_suffixes = [' annual fee structure', ' admission charges']
    elif category == 'admission':
        variant_suffixes = [' eligibility criteria', ' gd goenka admission']
    elif category == 'course':
        variant_suffixes = [' curriculum', ' specialization']
    elif category == 'university_info':
        variant_suffixes = [' GD Goenka', ' campus facilities']
    else:  # general
        variant_suffixes = [' profile', ' research']

    queries = []
    for q in ([rewritten] + [f"{rewritten}{s}" for s in variant_suffixes]):
        if q and q not in queries:
            queries.append(q)

    # -------------------------------------------------------------------------
    # Chunk-level retrieval across all query variants (Phase-1 Hybrid RAG)
    # -------------------------------------------------------------------------
    # Each variant now goes through HybridRetriever (dense FAISS + BM25 ->
    # RRF -> cross-encoder rerank -> adjacent-chunk merge) instead of the
    # dense-only rag.get_candidates() call. HybridRetriever returns
    # candidates in the exact same dict shape (score/index/content/metadata/
    # exact_match) that rag.get_candidates() already produced, so every
    # downstream step below -- the merged_chunks dedup, faculty isolation,
    # menu-boilerplate filtering, and context assembly -- is unchanged.
    topk = TOP_K_RESULTS
    merged_chunks: Dict[str, Dict] = {}  # content_hash → best candidate dict
    for q in queries:
        # BM25 only runs on the original (un-suffixed) query -- see
        # HybridRetriever.get_hybrid_candidates' docstring for why running
        # it on the broadened " profile"/" research"-style variants too
        # would let a single generic suffix word manufacture false
        # relevance via literal term overlap. Dense retrieval still runs on
        # every variant exactly as before.
        for cand in hybrid.get_hybrid_candidates(q, top_k=topk, category=category, use_bm25=(q == rewritten)):
            content_norm = ' '.join(cand['content'].split()).lower()
            h = hashlib.md5(content_norm.encode('utf-8')).hexdigest()
            if h not in merged_chunks or cand['score'] < merged_chunks[h]['score']:
                merged_chunks[h] = cand

    # If there is an exact faculty metadata match, keep only that faculty's chunks.
    exact_faculty = [
        cand for cand in merged_chunks.values()
        if cand.get('exact_match', 0) >= 2
        and _matches_category(cand['metadata'], 'faculty')
    ]
    if exact_faculty:
        primary = min(exact_faculty, key=lambda cand: cand['score'])
        primary_entity = _entity_key(primary['metadata'])
        merged_chunks = {
            h: cand for h, cand in merged_chunks.items()
            if _entity_key(cand['metadata']) == primary_entity
        }

    # Drop menu/navigation/footer boilerplate chunks before final assembly.
    merged_chunks = {
        h: cand for h, cand in merged_chunks.items()
        if not _is_menu_boilerplate(cand.get('content', '') or '')
    }

    # Sort merged unique chunks by score (best first), take top_k
    top_chunks = sorted(merged_chunks.values(), key=lambda x: x['score'])[:topk]

    if DEBUG_RAG:
        print(f"[RAG] Candidates Before Filter (merged across variants) : {len(merged_chunks)}")
        print(f"[RAG] Candidates After top_k cap                        : {len(top_chunks)}")

    # -------------------------------------------------------------------------
    # Evidence Check (Phase-1 Hybrid RAG, Step 10)
    # -------------------------------------------------------------------------
    # If hybrid retrieval found nothing relevant after all filtering above,
    # return the grounded "not found" fallback instead of calling the LLM on
    # thin/irrelevant context. Length-only check here; see
    # hybrid.has_sufficient_evidence's docstring for why a numeric
    # cross-encoder threshold isn't safely reusable when the model didn't
    # load for a given candidate set built across multiple query variants.
    if not top_chunks:
        if session_id:
            _store_session(session_id, user_message, NO_EVIDENCE_FALLBACK)
        return {'reply': f"<div>{NO_EVIDENCE_FALLBACK}</div>", 'data': {}, 'sources': []}

    # Sanitize and assemble full_context with paragraph-level cleaning
    parts = []
    seen_paragraphs = set()
    for c in top_chunks:
        src   = c['metadata'].get('source', 'Unknown')
        title = c['metadata'].get('title', '')
        chunk_text = re.sub(
            r"(?i)(?:<current_tab_state>[\s\S]*?</current_tab_state>|"
            r"\\u003ccurrent_tab_state\\u003e[\s\S]*?\\u003c\\/current_tab_state\\u003e)",
            "", c['content']
        )
        paragraphs = _clean_chunk_content(chunk_text)
        for paragraph in paragraphs:
            normalized = ' '.join(paragraph.split()).strip()
            if not normalized:
                continue
            para_hash = hashlib.md5(normalized.lower().encode('utf-8')).hexdigest()
            if para_hash in seen_paragraphs:
                continue
            seen_paragraphs.add(para_hash)
            parts.append(f"[Source: {title or src}]\n{normalized}")

    full_context = '\n\n---\n\n'.join(parts)
    full_context = re.sub(r"(\[Source:[^\]]+\])(?:\s*\1)+", r"\1", full_context)

    history = _compose_history(session_id)
    prompt = PROMPT_TEMPLATE.format(
        STUDENT_CONTEXT=student_context,
        RAG_CONTEXT=full_context or 'No relevant context found',
        HISTORY=history,
        USER_QUESTION=user_message
    )
    model = models.models.get('groq-llama') or next(iter(models.models.values()))
    try:
        resp = model.generate(prompt, max_tokens=800, temperature=0.1)
        text = getattr(resp, 'content', None) or getattr(resp, 'text', '') or ''
    except Exception:
        text = full_context[:3000] or 'No response generated'

    # sanitize visible reply
    text = _sanitize_visible_text(text)
    text = dedupe_sentences(text)
    short = summarize_text(text, max_sentences=5)

    structured = parse_structured_from_text(text)

    # Build clean HTML — summary first, then only non-empty structured fields
    html_parts = []
    if short:
        short_clean = re.sub(r"^[\s\-\.:]+", '', short).strip()
        short_clean = re.sub(r"(---\s*\.?\s*)+", "\n", short_clean)
        # Strip any leftover "Retrieved Context:" lines from summary
        short_clean = re.sub(r"Retrieved Context:[^\n<]*", "", short_clean).strip()
        if short_clean:
            html_parts.append(f"<div>{short_clean}</div>")

    block = []
    if structured.get('role'):
        block.append(f"<div><strong>Role:</strong> {structured['role']}</div>")
    if structured.get('location'):
        block.append(f"<div><strong>Location:</strong> {structured['location']}</div>")
    if structured.get('education'):
        eds = ''.join(f"<li>{e}</li>" for e in structured['education'])
        block.append(f"<div><strong>Education:</strong><ul style='margin:6px 0 0 18px'>{eds}</ul></div>")
    if structured.get('research'):
        rrs = ''.join(f"<li>{r}</li>" for r in structured['research'])
        block.append(f"<div><strong>Research Interests:</strong><ul style='margin:6px 0 0 18px'>{rrs}</ul></div>")
    if structured.get('experience'):
        exs = ''.join(f"<li>{e}</li>" for e in structured['experience'])
        block.append(f"<div><strong>Experience:</strong><ul style='margin:6px 0 0 18px'>{exs}</ul></div>")
    # Only show real URLs, not "Retrieved Context:" strings
    real_links = [l for l in structured.get('links', []) if l.startswith('http')]
    if real_links:
        links = ''.join(f"<li><a href='{l}' target='_blank'>{l}</a></li>" for l in real_links[:3])
        block.append(f"<div><strong>Links:</strong><ul style='margin:6px 0 0 18px'>{links}</ul></div>")
    # Cap sources at 3 unique real URLs only
    real_sources = list(dict.fromkeys(
        s for s in structured.get('sources', []) if s.startswith('http')
    ))[:3]
    if real_sources:
        srcs = ''.join(f"<li><a href='{s}' target='_blank'>{s}</a></li>" for s in real_sources)
        block.append(f"<div><strong>Sources:</strong><ul style='margin:6px 0 0 18px'>{srcs}</ul></div>")

    if block:
        html_parts.append('<div style="margin-top:8px">' + ''.join(block) + '</div>')

    html_reply = ''.join(html_parts) if html_parts else '<div>Not available in sources</div>'

    # save structured
    if structured.get('name') and structured['name'] != 'Not available in sources':
        _save_structured_to_vectordb(structured['name'], structured)
    if session_id:
        _store_session(session_id, user_message, short)

    return {'reply': html_reply, 'data': structured, 'sources': structured.get('sources', [])}