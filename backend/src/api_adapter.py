import re, json, os, hashlib, time
from collections import deque, defaultdict
from typing import Dict, Any, Optional, List
from src.threaded_rag import ThreadedRAGSystem, classify_query, _matches_category
from src.threaded_models import ThreadedModelManager
from src.hybrid_retriever import HybridRetriever
from src.config import DEBUG_RAG, TOP_K_RESULTS
from src.timetable_lookup import get_timetable_context
from src.timetable_store import answer_timetable_query, is_timetable_intent, has_active_timetable_session, is_timetable_escape
from src.mentor_store import answer_mentor_query, is_mentor_intent
from src.rag_debug import RagDebugger, timer as _timer

rag = ThreadedRAGSystem()
# Phase-1 Hybrid RAG: wraps the existing `rag` instance (FAISS, category
# filtering, exact-match boosting, threshold filtering, dedup all untouched).
# `rag` itself is unmodified, so any other caller keeps identical behavior.
hybrid = HybridRetriever(rag)
models = ThreadedModelManager()
session_histories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

NO_EVIDENCE_FALLBACK = "I couldn't find enough information in the university knowledge base."

# FEATURE 5: strict university guardrail fallback message (exact text required).
OUT_OF_SCOPE_FALLBACK = "I can only answer questions related to GD Goenka University and information available in the university knowledge base."

# -------------------------------------------------------------------------
# FEATURE 4: lightweight small-talk / conversational-intent detection
# -------------------------------------------------------------------------
# Runs BEFORE any retrieval. A short, closed list of greeting/courtesy
# patterns -- intentionally conservative (whole-message match after
# trimming punctuation) so it never swallows a real university question
# that happens to start with a greeting word ("hi, who is Dr. Singh" is
# NOT treated as small talk because the full normalized message doesn't
# match a pure-greeting pattern).
_SMALL_TALK_RE = re.compile(
    r"^(hi+|hello+|hey+)(\s+\w+)?[\s!.?]*$|"
    r"^good\s*(morning|afternoon|evening|night)[\s!.?]*$|"
    r"^how\s*(are\s*(you|u|going)|'?s\s*it\s*going|are\s*things)[\s!.?]*$|"
    r"^what'?s\s*up[\s!.?]*$|^sup[\s!.?]*$|^yo[\s!.?]*$|"
    r"^(thanks?(\s*you)?|thank\s*you|thx|ty)[\s!.?]*$|"
    r"^(bye+|goodbye|see\s*you|see\s*ya|good\s*night|gn)[\s!.?]*$|"
    r"^(ok(ay)?|cool|nice|great|awesome)[\s!.?]*$",
    re.IGNORECASE,
)

_SMALL_TALK_RESPONSES = {
    'greeting': "Hello! I'm UniBuddy, your GD Goenka University assistant. Ask me about faculty, fees, admissions, courses, or campus facilities.",
    'how_are_you': "I'm doing well, thanks for asking! How can I help you with GD Goenka University today?",
    'thanks': "You're welcome! Let me know if you have any other questions about GD Goenka University.",
    'bye': "Goodbye! Feel free to come back anytime you have questions about GD Goenka University.",
    'ack': "Got it! Is there anything else about GD Goenka University I can help with?",
}


def _classify_small_talk(user_message: str) -> Optional[str]:
    """Returns a small-talk category if the FULL message (not a substring)
    is pure conversational filler, else None. Conservative by design: a
    real question that merely contains a greeting word is never matched,
    because we anchor on ^...$ against the whole normalized message."""
    normalized = (user_message or '').strip().lower()
    if not normalized:
        return None
    if not _SMALL_TALK_RE.match(normalized):
        return None
    if re.match(r"^(thanks?(\s*you)?|thank\s*you|thx|ty)[\s!.?]*$", normalized, re.I):
        return 'thanks'
    if re.match(r"^(bye+|goodbye|see\s*you|see\s*ya|good\s*night|gn)[\s!.?]*$", normalized, re.I):
        return 'bye'
    if re.match(r"^how\s*(are\s*(you|u|going)|'?s\s*it\s*going|are\s*things)[\s!.?]*$", normalized, re.I):
        return 'how_are_you'
    if re.match(r"^(ok(ay)?|cool|nice|great|awesome)[\s!.?]*$", normalized, re.I):
        return 'ack'
    return 'greeting'


# -------------------------------------------------------------------------
# FEATURE 5: strict university-scope guardrail
# -------------------------------------------------------------------------
# Runs BEFORE retrieval/LLM for queries that are clearly about a
# real-world, non-university topic. This is intentionally a small,
# high-confidence blocklist of well-known off-topic entity/topic types
# rather than a broad classifier, so it does not risk blocking legitimate
# university questions. The evidence check (Step 10, already in place)
# remains the primary safety net for queries this guardrail doesn't catch;
# this feature targets the specific failure seen in the conversation audit
# (general-knowledge questions like "Who is Iron Man" / "Who is Narendra
# Modi" being answered from the LLM's own training data instead of being
# refused).
_OFF_TOPIC_RE = re.compile(
    r"\b(iron\s*man|spider-?man|batman|superman|avengers|marvel|dc\s*comics|"
    r"narendra\s*modi|prime\s*minister\s*of\s*india|president\s*of\s*(india|usa|america)|"
    r"\bipl\b|world\s*cup|fifa|olympics|cricket\s*score|"
    r"capital\s*of\s*(india|france|usa|america)|"
    r"who\s*is\s*the\s*(ceo|founder)\s*of\s*(google|apple|microsoft|amazon|tesla|meta|facebook)|"
    r"elon\s*musk|bill\s*gates|mark\s*zuckerberg|jeff\s*bezos)\b",
    re.IGNORECASE,
)

# University-specific terms that should NEVER be treated as off-topic even
# if they happen to co-occur near a generic word above (defense against
# false positives -- e.g. a real GDGU question mentioning "president" of a
# club/society should not be blocked).
_UNIVERSITY_CONTEXT_RE = re.compile(
    r"\b(gd\s*goenka|gdgu|university|faculty|professor|dr\.?\s|department|"
    r"admission|fee|hostel|campus|course|btech|mba|bca|placement)\b",
    re.IGNORECASE,
)


def _is_off_topic(user_message: str) -> bool:
    """Conservative guardrail: only fires when the message matches a known
    off-topic pattern AND does not also contain university-specific
    context. This avoids blocking a legitimate question that happens to
    mention, say, a company name in a university context (e.g. "does GD
    Goenka have a placement tie-up with Amazon" is NOT blocked)."""
    if not user_message:
        return False
    if _UNIVERSITY_CONTEXT_RE.search(user_message):
        return False
    return bool(_OFF_TOPIC_RE.search(user_message))


# -------------------------------------------------------------------------
# FEATURE 1: pipeline debug logging helper
# -------------------------------------------------------------------------
# Centralizes the "print a clearly-labeled debug section" pattern used
# throughout get_reply() below, controlled entirely by the existing
# DEBUG_RAG flag (imported above). If DEBUG_RAG is False, this is a no-op,
# matching the existing project convention of guarding all [RAG]-prefixed
# prints behind DEBUG_RAG.
def _debug_section(step_num: int, title: str, *lines: str):
    if not DEBUG_RAG:
        return
    print(f"\n{'='*65}\nSTEP {step_num}\n{title}\n{'='*65}")
    for line in lines:
        print(line)


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

# FEATURE 3: category-aware prompt templates. Only 'faculty' queries use the
# structured Role/Education/Research/Experience template; every other
# category (admission, fee, course, university_info, general) gets a plain
# conversational template with no headings, per the conversation audit
# finding that non-faculty answers were incorrectly forced into the
# faculty structured shape.
#
# FEATURE 10: both templates explicitly instruct the model to OMIT any
# field it has no grounded information for, instead of writing
# "Not available in sources" under an empty heading -- the rendering layer
# (parse_structured_from_text / HTML building below) also independently
# enforces this by only rendering a heading if non-empty text was parsed,
# so empty headings are blocked at two layers, not just relying on the
# model to follow instructions.
#
# FEATURE 8: {FOCUS_INSTRUCTION} is empty for a normal query, and is filled
# in by get_reply() when a follow-up like "Research?" or "Qualification?"
# resolves to a known person AND a known requested field, telling the
# model to answer ONLY that field instead of the full profile.
FACULTY_PROMPT_TEMPLATE = '''System: You are UniBuddy, the official intelligent assistant for GD Goenka University. Use ONLY the retrieved context and conversation history below -- never use outside/general knowledge. If specific information (e.g. a field, a fact, a number) is not present in the Retrieved Context, omit it entirely rather than guessing or writing "Not available in sources" -- do not invent or assume anything not explicitly present in the Retrieved Context.

Retrieved Context:
{RAG_CONTEXT}

Conversation History:
{HISTORY}

Current Question:
{USER_QUESTION}
{FOCUS_INSTRUCTION}
Provide a short factual summary (2-6 sentences) of the faculty member based only on the Retrieved Context, then a structured block using ONLY the headings for which you found real information: Role, Department, Qualification, Research Interests, Experience. Do NOT include a heading for any field you have no grounded information for -- skip it completely, do not write "Not available in sources" or similar. Do NOT include a Sources or Links section; sources are added separately by the system.
'''

GENERAL_PROMPT_TEMPLATE = '''System: You are UniBuddy, the official intelligent assistant for GD Goenka University. Use ONLY the retrieved context and conversation history below -- never use outside/general knowledge. If the answer is not present in the Retrieved Context, say so plainly rather than guessing.

Retrieved Context:
{RAG_CONTEXT}

Conversation History:
{HISTORY}

Current Question:
{USER_QUESTION}

Answer in plain, natural conversational prose (2-6 sentences). Do NOT use headings, bullet lists of "Role/Education/Experience", or any structured profile format -- that format is reserved for faculty questions only. Do NOT include a Sources or Links section; sources are added separately by the system.
'''

FOLLOWUP_RE = re.compile(
    r"\b(?:he|she|they|him|her|them|his|hers|their|theirs|tell me more|more about|details|research|publications|profile|where is|what does|what are)\b",
    re.I)

_NAME_BLACKLIST = {
    'university', 'campus', 'school', 'college', 'department', 'facility',
    'facilities', 'admission', 'admissions', 'course', 'courses',
    'program', 'programs', 'fee', 'fees', 'placement', 'hostel', 'library'
}

# FEATURE 8: map a short, focused follow-up question to the single profile
# field it's asking about. Only fires for SHORT queries (a handful of
# words) so a longer, genuinely new question that happens to contain the
# word "research" isn't misread as a narrow follow-up.
_FIELD_FOLLOWUP_PATTERNS = [
    (re.compile(r"^(his|her|their)?\s*research(\s*interests?)?\??$", re.I), 'research'),
    (re.compile(r"^(his|her|their)?\s*qualifications?\??$", re.I), 'qualification'),
    (re.compile(r"^(his|her|their)?\s*experience\??$", re.I), 'experience'),
    (re.compile(r"^(his|her|their)?\s*(department|dept)\??$", re.I), 'department'),
    (re.compile(r"^(his|her|their)?\s*(designation|role|position)\??$", re.I), 'designation'),
    (re.compile(r"^(his|her|their)?\s*(publications?)\??$", re.I), 'research'),
]


def _detect_focus_field(user_query: str) -> Optional[str]:
    """Returns a field name ('research', 'qualification', 'experience',
    'department', 'designation') if user_query is a short, single-field
    follow-up question, else None. Deliberately strict (whole-message
    match) so it never fires on a longer compound question."""
    normalized = (user_query or '').strip().lower()
    if not normalized or len(normalized.split()) > 4:
        return None
    for pattern, field in _FIELD_FOLLOWUP_PATTERNS:
        if pattern.match(normalized):
            return field
    return None


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
    data = {'name':None,'role':None,'department':None,'location':None,
            'qualification':[], 'education':[],'research':[], 'experience':[],
            'links':[], 'sources':[]}
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
    role = re.search(r"Role[:\*\n\r]+([^\n\r]{0,120})", text)
    if role:
        role_val = role.group(1).strip(' *:-')
        if role_val and 'not available' not in role_val.lower():
            data['role'] = role_val
    dept = re.search(r"Department[:\*\n\r]+([^\n\r]{0,120})", text)
    if dept:
        dept_val = dept.group(1).strip(' *:-')
        if dept_val and 'not available' not in dept_val.lower():
            data['department'] = dept_val
    # FEATURE 10 fix: all multi-line field captures stop at the next known
    # field header (via lookahead) rather than a raw character count, so a
    # field's content never bleeds into the next field's label+text (this
    # was verified to happen with the old {0,200}-character-only patterns:
    # "Qualification: PhD\nResearch Interests: X" parsed Qualification as
    # ["PhD", "Research Interests: X"] instead of just ["PhD"]).
    _NEXT_FIELD = r"(?=\n\s*(?:Role|Department|Qualification|Education|Research Interests?|Experience|Designation)\s*[:\*]|\Z)"
    qual = re.search(r"Qualification[:\*\n\r]+([\s\S]{0,300}?)" + _NEXT_FIELD, text)
    if qual:
        data['qualification'] = [s.strip() for s in re.split(r'[\n\r\u2022\-]+', qual.group(1)) if s.strip()][:3]
    ed = re.search(r"Education[:\*\n\r]+([\s\S]{0,300}?)" + _NEXT_FIELD, text)
    if ed:
        data['education'] = [s.strip() for s in re.split(r'[\n\r\u2022\-]+', ed.group(1)) if s.strip()][:3]
    rs = re.search(r"Research Interests?[:\*\n\r]+([\s\S]{0,300}?)" + _NEXT_FIELD, text)
    if rs:
        data['research'] = [s.strip() for s in re.split(r'[\n\r\u2022\-]+', rs.group(1)) if s.strip()][:5]
    ex = re.search(r"Experience[:\*\n\r]+([\s\S]{0,300}?)" + _NEXT_FIELD, text)
    if ex:
        data['experience'] = [s.strip() for s in re.split(r'[\n\r\u2022\-]+', ex.group(1)) if s.strip()][:5]
    # FEATURE 10: never carry forward a "Not available in sources"-style
    # placeholder value into a list field -- strip any such entries so an
    # empty-looking heading never renders even if the model didn't fully
    # follow the "omit empty fields" instruction.
    for key in ('qualification', 'education', 'research', 'experience'):
        data[key] = [v for v in data[key] if v and 'not available' not in v.lower()]
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


def _finish_reply(raw_text: str, category: str, top_chunks: List[Dict],
                   user_message: str, session_id: Optional[str],
                   gen_time: Optional[float] = None, retrieval_time: Optional[float] = None) -> Dict[str, Any]:
    """Shared post-processing pipeline used by BOTH the non-streaming and
    streaming (FEATURE 2) paths in get_reply(), so this logic exists in
    exactly one place. Takes the raw LLM output text plus the retrieval
    context (category, top_chunks) already computed by the caller, and
    returns the final {reply, data, sources} dict -- identical output shape
    and content regardless of which path produced raw_text."""
    text = raw_text

    _debug_section(11, "Raw LLM Output", text)

    # sanitize visible reply
    text = _sanitize_visible_text(text)

    # FEATURE 3/10 fix: parse structured fields (Role/Department/
    # Qualification/Research/Experience) from the sanitized text BEFORE
    # dedupe_sentences runs. dedupe_sentences splits on sentence punctuation
    # and rejoins with single spaces, which collapses the newlines that
    # separate "Role: X" from "Department: Y" in the model's structured
    # block -- parsing after that collapse caused each field's regex to
    # over-capture into the next field's text (verified directly: "Role: X
    # Department: Y" parsed as Role="X Department: Y Qualification: Z...").
    # The prose summary below still uses the deduped/summarized text, which
    # is unaffected by this since summarize_text doesn't care about field
    # boundaries.
    structured = parse_structured_from_text(text)

    text = dedupe_sentences(text)
    short = summarize_text(text, max_sentences=5)

    # FEATURE 7: extract real source URLs directly from the retrieved
    # context's [Source: ...] tags (which trace back to metadata.json's
    # `source` field via threaded_rag.py/hybrid_retriever.py), instead of
    # relying on the LLM to echo a URL in its own generated text. This
    # guarantees sources shown to the user were actually retrieved and
    # used, never hallucinated, and shows ALL contributing sources (not
    # just whichever one the LLM happened to mention).
    context_source_urls = []
    for c in top_chunks:
        src = c['metadata'].get('source', '')
        if isinstance(src, str) and src.startswith(('http://', 'https://')) and src not in context_source_urls:
            context_source_urls.append(src)

    # Build clean HTML — summary first, then (FEATURE 3) only render the
    # structured faculty block when this is actually a faculty query;
    # every other category gets plain prose only, no headings.
    html_parts = []
    if short:
        short_clean = re.sub(r"^[\s\-\.:]+", '', short).strip()
        short_clean = re.sub(r"(---\s*\.?\s*)+", "\n", short_clean)
        # Strip any leftover "Retrieved Context:" lines from summary
        short_clean = re.sub(r"Retrieved Context:[^\n<]*", "", short_clean).strip()
        if short_clean:
            html_parts.append(f"<div>{short_clean}</div>")

    block = []
    if category == 'faculty':
        # FEATURE 10: each line below is only added if the corresponding
        # field is non-empty, so a faculty profile with missing fields
        # never shows an empty "Education: Not available" heading -- the
        # heading simply doesn't appear at all.
        if structured.get('role'):
            block.append(f"<div><strong>Designation:</strong> {structured['role']}</div>")
        if structured.get('department'):
            block.append(f"<div><strong>Department:</strong> {structured['department']}</div>")
        if structured.get('qualification'):
            qs = ''.join(f"<li>{q}</li>" for q in structured['qualification'])
            block.append(f"<div><strong>Qualification:</strong><ul style='margin:6px 0 0 18px'>{qs}</ul></div>")
        if structured.get('education'):
            eds = ''.join(f"<li>{e}</li>" for e in structured['education'])
            block.append(f"<div><strong>Education:</strong><ul style='margin:6px 0 0 18px'>{eds}</ul></div>")
        if structured.get('research'):
            rrs = ''.join(f"<li>{r}</li>" for r in structured['research'])
            block.append(f"<div><strong>Research Interests:</strong><ul style='margin:6px 0 0 18px'>{rrs}</ul></div>")
        if structured.get('experience'):
            exs = ''.join(f"<li>{e}</li>" for e in structured['experience'])
            block.append(f"<div><strong>Experience:</strong><ul style='margin:6px 0 0 18px'>{exs}</ul></div>")
    # For every category (faculty included), show grounded source URLs.
    if context_source_urls:
        srcs = ''.join(f"<li><a href='{s}' target='_blank'>{s}</a></li>" for s in context_source_urls[:5])
        block.append(f"<div><strong>Source{'s' if len(context_source_urls) > 1 else ''}:</strong><ul style='margin:6px 0 0 18px'>{srcs}</ul></div>")

    if block:
        html_parts.append('<div style="margin-top:8px">' + ''.join(block) + '</div>')

    html_reply = ''.join(html_parts) if html_parts else '<div>Not available in sources</div>'

    _debug_section(12, "Post Processing",
                    f"Structured Parsing:\n{structured}",
                    f"Final HTML:\n{html_reply[:1500]}")
    _debug_section(13, "Sources", f"Extracted URLs:\n{context_source_urls}")

    # save structured
    if structured.get('name') and structured['name'] != 'Not available in sources':
        _save_structured_to_vectordb(structured['name'], structured)
    if session_id:
        _store_session(session_id, user_message, short)

    if DEBUG_RAG:
        total_time = (retrieval_time or 0) + (gen_time or 0)
        _debug_section(14, "Total Timing",
                        f"Dense+BM25+RRF+Rerank (combined): {retrieval_time:.3f}s" if retrieval_time is not None else "N/A",
                        f"Generation                      : {gen_time:.3f}s" if gen_time is not None else "N/A",
                        f"Total                            : {total_time:.3f}s")

    return {'reply': html_reply, 'data': structured, 'sources': context_source_urls}


def get_reply(user_message: str, session_id: str = None, user_profile: Dict = None, _stream: bool = False) -> Dict[str,Any]:
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

    _debug_section(1, "Intent Detection")

    # FEATURE 4: small-talk / greeting detection, runs BEFORE any retrieval.
    # Placed after timetable/mentor routing (so it never intercepts a
    # mid-flow timetable/mentor conversation) but before RAG, so greetings
    # like "hi", "how are going", "thanks", "bye" get a simple canned
    # response instead of falling through to retrieval and surfacing an
    # unrelated faculty chunk (the exact failure seen in the conversation
    # audit for "how are going").
    small_talk_kind = _classify_small_talk(user_message)
    if small_talk_kind:
        if DEBUG_RAG:
            print(f"Detected:\nSmall Talk ({small_talk_kind})")
        reply_text = _SMALL_TALK_RESPONSES.get(small_talk_kind, _SMALL_TALK_RESPONSES['greeting'])
        if session_id:
            _store_session(session_id, user_message, reply_text)
        return {'reply': f"<div>{reply_text}</div>", 'data': {}, 'sources': []}

    # FEATURE 5: strict university-scope guardrail, runs BEFORE retrieval/LLM.
    # Catches clearly off-topic general-knowledge questions (the conversation
    # audit found "Who is Iron Man?" and "Who is Narendra Modi?" were both
    # answered in full from the LLM's training data instead of being
    # refused). The existing evidence check further downstream remains the
    # safety net for anything this guardrail's conservative pattern list
    # doesn't catch.
    if _is_off_topic(user_message):
        if DEBUG_RAG:
            print(f"Detected:\nOff-topic / Out-of-scope query")
        if session_id:
            _store_session(session_id, user_message, OUT_OF_SCOPE_FALLBACK)
        return {'reply': f"<div>{OUT_OF_SCOPE_FALLBACK}</div>", 'data': {}, 'sources': []}

    if DEBUG_RAG:
        print("Detected:\nUniversity Knowledge Base Query")

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

    # FEATURE 8: detect whether this is a short, single-field follow-up
    # (e.g. "Research?", "Qualification?") so the prompt can be told to
    # answer ONLY that field instead of repeating the entire faculty
    # profile, which is what happened for every such turn in the
    # conversation audit ("Tell me more", "Research?", "Qualification?"
    # all returned an identical full repeat).
    focus_field = _detect_focus_field(user_message)

    _debug_section(2, "Follow-up Detection",
                    f"Original Query:\n{user_message}",
                    f"Rewritten Query:\n{rewritten}",
                    f"Focus Field:\n{focus_field or '(none -- full answer)'}")

    # -------------------------------------------------------------------------
    # Category detection
    # -------------------------------------------------------------------------
    # Classify the original user message (before follow-up rewriting) so
    # "his research" → rewritten "Dr X his research" doesn't change the
    # detected category, which should already be 'faculty' from context.
    category = classify_query(user_message)

    # FEATURE 8 fix: classify_query alone does NOT recognize bare
    # single-field follow-ups ("Research?", "Qualification?", "Experience?")
    # as faculty category -- verified directly (classify_query('Research?')
    # returns 'general', not 'faculty', since its faculty keyword list
    # requires words like "who is"/"professor"/"dr." that a one-word
    # follow-up doesn't contain). Since focus_field only fires for exactly
    # this kind of short person-attribute follow-up, treat it as a reliable
    # signal that the true category is faculty -- this fixes both the
    # category-based retrieval filtering (_matches_category) and, further
    # below, the structured-vs-prose prompt template selection.
    if focus_field and category == 'general':
        category = 'faculty'
        if DEBUG_RAG:
            print(f"[RAG] Category corrected to 'faculty' due to focus_field='{focus_field}'")

    # FEATURE 9: detect aggregate/"list" style queries (e.g. "List all CSE
    # faculty", "Tell me about CSE department"). These must be answered
    # ONLY from freshly retrieved context, never from conversation history
    # -- the conversation audit found the LLM was fabricating department
    # rosters out of whatever faculty names happened to appear earlier in
    # the same session. We don't change retrieval for this (Step 9 of the
    # original spec said "answer only from retrieval", which the existing
    # RAG pipeline already does); what we change is that the prompt's
    # Conversation History section is suppressed for these queries, so the
    # model has no session history to draw fabricated facts from -- it can
    # only use the History to resolve a follow-up subject, never as a
    # source of facts, which is the literal Feature 9 requirement.
    _LIST_QUERY_RE = re.compile(
        r"\b(list\s+all|all\s+(professors|faculty|teachers)|"
        r"tell\s+me\s+about\s+\w+\s+department|"
        r"\w+\s+department\b)",
        re.IGNORECASE,
    )
    is_list_query = bool(_LIST_QUERY_RE.search(user_message))

    _debug_section(3, "Category Classification",
                    f"Category:\n{category}",
                    f"List/Aggregate Query:\n{is_list_query}")

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
    #
    # FEATURE 1 (Steps 4-7 of the debug spec -- Dense Retrieval, BM25
    # Retrieval, RRF Fusion, Cross-Encoder Reranking): these are already
    # logged inside hybrid_retriever.py's existing [HYBRID]-prefixed
    # DEBUG_RAG prints (Dense Top-K / BM25 Top-K / RRF ranking / Reranker
    # ranking) -- not duplicated here to avoid two competing log formats
    # for the same data. This block only adds timing around the call
    # (Step 14) plus Step 8/9 (merge, evidence) which genuinely live here.
    _retrieval_start = time.time() if DEBUG_RAG else None
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
    _retrieval_time = (time.time() - _retrieval_start) if DEBUG_RAG else None

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

    _debug_section(8, "Chunk Merge",
                    f"Candidates Before Filter (merged across variants): {len(merged_chunks)}",
                    f"Candidates After top_k cap                       : {len(top_chunks)}",
                    "Merged Context (doc_id, score):",
                    *[f"  {c['metadata'].get('doc_id')} | {round(c['score'], 4)}" for c in top_chunks[:10]])

    # -------------------------------------------------------------------------
    # Evidence Check (Phase-1 Hybrid RAG, Step 10)
    # -------------------------------------------------------------------------
    # If hybrid retrieval found nothing relevant after all filtering above,
    # return the grounded "not found" fallback instead of calling the LLM on
    # thin/irrelevant context. Length-only check here; see
    # hybrid.has_sufficient_evidence's docstring for why a numeric
    # cross-encoder threshold isn't safely reusable when the model didn't
    # load for a given candidate set built across multiple query variants.
    evidence_ok = bool(top_chunks)
    _debug_section(9, "Evidence Check",
                    f"Evidence Score: {len(top_chunks)} grounded chunk(s)",
                    f"Decision: {'Proceed' if evidence_ok else 'Fallback'}")
    if not evidence_ok:
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

    # FEATURE 9: list/aggregate queries must answer ONLY from freshly
    # retrieved context, never from conversation history (the conversation
    # audit found department-roster answers were fabricated from whatever
    # names had appeared earlier in the session). History is still composed
    # normally for follow-up *subject* resolution (handled earlier by
    # _rewrite_followup_if_needed, which already ran before this point) --
    # what changes here is only whether the History block is included in
    # the prompt the LLM actually sees, so it has nothing to draw
    # fabricated "facts" from for this turn.
    history = '' if is_list_query else _compose_history(session_id)

    # FEATURE 8: build a focus instruction for the prompt when this is a
    # short single-field follow-up ("Research?", "Qualification?"), so the
    # model answers ONLY that field instead of repeating the whole profile.
    focus_instruction = ''
    if focus_field and category == 'faculty':
        field_labels = {
            'research': 'Research Interests',
            'qualification': 'Qualification',
            'experience': 'Experience',
            'department': 'Department',
            'designation': 'Designation (Role)',
        }
        label = field_labels.get(focus_field, focus_field)
        focus_instruction = (
            f"\nThe user is asking specifically about the {label} of the person "
            f"already discussed in the conversation history. Answer ONLY with "
            f"the {label} information found in the Retrieved Context -- do NOT "
            f"repeat their full profile (no summary, no other headings).\n"
        )

    # FEATURE 3: category-aware template selection. Faculty queries get the
    # structured profile template; everything else gets the plain
    # conversational template (no headings).
    template = FACULTY_PROMPT_TEMPLATE if category == 'faculty' else GENERAL_PROMPT_TEMPLATE
    prompt_kwargs = dict(
        RAG_CONTEXT=full_context or 'No relevant context found',
        HISTORY=history,
        USER_QUESTION=user_message,
    )
    if template is FACULTY_PROMPT_TEMPLATE:
        prompt_kwargs['FOCUS_INSTRUCTION'] = focus_instruction
    prompt = template.format(**prompt_kwargs)

    _debug_section(10, "Prompt", prompt[:2000] + ('...[truncated]' if len(prompt) > 2000 else ''))

    # Model selection happens once, before both the streaming and
    # non-streaming branches below, since both need it.
    model = models.models.get('groq-llama') or next(iter(models.models.values()))

    # FEATURE 2: real token-by-token streaming. When _stream=True, every
    # step above this line (retrieval, grounding/evidence-check, small-talk
    # and guardrail gates, category detection, prompt building) has already
    # run identically to the non-streaming path -- nothing about Module 1's
    # retrieval or safety logic is duplicated or changed. From here, we
    # switch to Groq's native stream=True generation (via the new
    # generate_stream() method added to ThreadedGroqModel) and yield text
    # deltas as they arrive, instead of blocking on the full completion.
    #
    # Streaming intentionally returns RAW incremental text, not the fully
    # post-processed HTML (structured-field parsing, source-URL injection,
    # menu-boilerplate stripping, etc. all require the COMPLETE response
    # text to run correctly -- they can't operate on partial sentences).
    # The streaming endpoint in api.py sends a final 'done' event carrying
    # the same fully-processed {reply, data, sources} dict that the
    # non-streaming path returns, once the full text has arrived, so
    # nothing about response quality or grounding is sacrificed -- only the
    # raw token deltas are unprocessed while they stream in.
    if _stream:
        def _stream_generator():
            full_text_parts = []
            _gen_start = time.time() if DEBUG_RAG else None
            try:
                for delta in model.generate_stream(prompt, max_tokens=1200, temperature=0.1):
                    full_text_parts.append(delta)
                    yield {'type': 'delta', 'text': delta}
            except Exception as e:
                yield {'type': 'delta', 'text': f"[stream error: {e}]"}
            _gen_time = (time.time() - _gen_start) if DEBUG_RAG else None
            full_raw_text = ''.join(full_text_parts)
            # Same shared post-processing pipeline as the non-streaming
            # path -- see _finish_reply, defined once above get_reply, used
            # by both branches so this logic exists in exactly one place.
            final = _finish_reply(
                full_raw_text, category, top_chunks, user_message, session_id,
                gen_time=_gen_time, retrieval_time=_retrieval_time,
            )
            yield {'type': 'done', **final}
        return _stream_generator()

    # FEATURE 6: increase max_tokens so longer, multi-field faculty profiles
    # or multi-part answers aren't cut off mid-sentence. 800 -> 1200 is a
    # conservative bump (Groq's llama-3.1-8b-instant context window
    # comfortably supports this) rather than a large jump that could
    # encourage repetition; temperature is left unchanged at 0.1 since the
    # cutoff issue was about length, not creativity.
    _gen_start = time.time() if DEBUG_RAG else None
    try:
        resp = model.generate(prompt, max_tokens=1200, temperature=0.1)
        text = getattr(resp, 'content', None) or getattr(resp, 'text', '') or ''
    except Exception:
        text = full_context[:3000] or 'No response generated'
    _gen_time = (time.time() - _gen_start) if DEBUG_RAG else None

    return _finish_reply(text, category, top_chunks, user_message, session_id,
                          gen_time=_gen_time, retrieval_time=_retrieval_time)