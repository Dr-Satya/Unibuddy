"""
Research module for Unibuddy
============================

Answers questions about the university's research output (publications) using the
Excel files in the `reserch_data/` folder.

How it works (deliberately simple):
  1. LOAD   - read every .xlsx file and turn each row into a `Paper`.
  2. PARSE  - turn the user's question into filters (year, quartile, scopus,
              department, author, topic...) and an intent (count / list / yearly / doi).
  3. ANSWER - filter the papers in plain Python and format the result.

Counting and filtering ("how many Q1 Scopus papers this year") is done in plain
Python because an LLM or a similarity search gets exact numbers wrong. The vector
index (FAISS) is only used for fuzzy things: "papers about fire retardant coatings"
and finding a paper from a half-remembered title.

A background thread watches the data folder. When an .xlsx file is added, edited or
removed it reloads the papers and re-embeds them, so no restart is needed.
The embeddings are saved to INDEX_DIR and reused on restart if the files are unchanged.

This file is fully standalone: it imports nothing from the rest of the project.
Extra dependency:  pip install openpyxl
(faiss-cpu, sentence-transformers and numpy are already in requirements.txt. If they
 are missing, everything still works - only the topic/fuzzy-title search is skipped.)

Quick test from the project root:
    python src/research_module.py "How many Q1 Scopus papers were published in 2026?"

Hooking it into the chat services (done by hand, in services.py and
threaded_services.py, inside `chat_async` where `rag_context` is built):

    from src.research_module import research_module
    ...
    if research_module.is_research_query(query):
        rag_context = research_module.get_context(query)
    else:
        rag_context = <existing rag call>
"""

import hashlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

# Folder with the .xlsx files. Default: <project root>/reserch_data (sic - that is
# the folder's current spelling). Override with the RESEARCH_DATA_DIR env variable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("RESEARCH_DATA_DIR") or next(
    (p for p in (os.path.join(_PROJECT_ROOT, "reserch_data"),
                 os.path.join(_PROJECT_ROOT, "research_data")) if os.path.isdir(p)),
    os.path.join(_PROJECT_ROOT, "reserch_data"),
)

# Where the saved vector index lives (separate from the rest of the project's vector DB).
INDEX_DIR = os.getenv("RESEARCH_INDEX_DIR") or os.path.join(_PROJECT_ROOT, "data", "research_vectordb")
EMBED_MODEL = "all-MiniLM-L6-v2"   # same model the rest of the project uses
WATCH_INTERVAL = float(os.getenv("RESEARCH_WATCH_SECONDS", "5"))  # how often to check the folder
SEMANTIC_TOP_K = 20                # most topic matches returned
SEMANTIC_MIN_SCORE = 0.30          # cosine similarity below this is treated as "not related"

MAX_LIST = 25  # never print more than this many papers in one answer

# Words that tell us a question is about publications (used by is_research_query).
RESEARCH_KEYWORDS = [
    "paper", "papers", "publication", "publications", "published", "publish",
    "journal", "journals", "scopus", "doi", "research paper", "wos",
    "web of science", "quartile", "h-index", "citation",
    "chapter", "chapters", "book chapter", "book chapters", "conference paper", "conference papers",
    "proceedings",
]

# Departments / schools / domains the user can mention.
#   query : phrases that may appear in the question
#   dept  : regex tested against the paper's "School/Department" cell
#   topic : regex tested against title + journal (finds papers by subject, even if
#           the department cell is vague). Leave "" to skip.
# A paper matches an entry if `dept` OR `topic` matches.
DEPARTMENTS = {
    # All the files are School of Engineering & Sciences records, so this matches everything.
    "School of Engineering and Sciences": {
        "query": ["school of engineering", "soes", "engineering and sciences", "engineering & sciences"],
        "dept": r".", "topic": "",
    },
    "Computer Science": {
        "query": ["computer science", "cse", "cs"],
        "dept": r"\bcse?\b|computer",
        "topic": r"computer|machine learning|deep learning|neural|algorithm|data mining|cyber|software|"
                 r"artificial intelligence|\bai\b|\biot\b|blockchain|big data",
    },
    "Electronics & Communication (ECE)": {
        "query": ["ece", "electronics and communication", "electronics"],
        "dept": r"\bece\b|electronic",
        "topic": r"antenna|wireless|communication|vlsi|signal processing|optical",
    },
    "Mechanical Engineering": {
        "query": ["mechanical"],
        "dept": r"mechanical", "topic": r"mechanical|manufactur|thermal",
    },
    "Basic and Applied Sciences": {
        "query": ["basic and applied", "dbas", "applied sciences"],
        "dept": r"dbas|basic", "topic": "",
    },
    "Fire, Health & Safety (COHSFE)": {
        "query": ["cohsfe", "c-ohsfe", "occupational health", "fire safety"],
        "dept": r"cohsfe|c-ohsfe|occupational", "topic": r"\bfire\b|flame|combustion",
    },
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

JOURNAL = "journal paper"
BOOK = "book chapter / conference paper"


@dataclass
class Paper:
    title: str
    authors: str = ""
    school: str = ""        # raw "School/Department" text from the sheet
    journal: str = ""
    year: Optional[int] = None
    month: str = ""
    doi: str = ""           # bare DOI, e.g. 10.1007/s10973-025-14808-2
    link: str = ""          # link to the article page
    quartile: str = ""      # "Q1".."Q4" or ""
    scopus: bool = False    # True if the row mentions Scopus anywhere
    source: str = ""        # which file it came from
    kind: str = JOURNAL     # JOURNAL or BOOK (book chapters / conference papers)

    @property
    def doi_url(self) -> str:
        return f"https://doi.org/{self.doi}" if self.doi else ""


# --------------------------------------------------------------------------- #
# Step 1: LOAD - read the Excel files
# --------------------------------------------------------------------------- #

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.I)
URL_RE = re.compile(r"https?://\S+", re.I)
QUARTILE_RE = re.compile(r"\bQ([1-4])\b", re.I)


def _text(value) -> str:
    """Cell value -> clean single-line string ('' for empty / 'NA' style cells)."""
    if value is None:
        return ""
    s = " ".join(str(value).split())
    return "" if s.lower() in ("na", "n/a", "nil", "none", "nan") else s


def _find_doi(*texts: str) -> str:
    """First DOI found in any of the given strings (trailing punctuation removed)."""
    for t in texts:
        m = DOI_RE.search(t or "")
        if m:
            return m.group(0).rstrip(".,;)")
    return ""


def _parse_year(value) -> Optional[int]:
    if isinstance(value, (datetime, date)):
        return value.year
    m = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(m.group(0)) if m else None


def _parse_month(value) -> str:
    if isinstance(value, (datetime, date)):
        return MONTHS[value.month - 1]
    s = str(value or "").lower()
    for name in MONTHS:  # matches "January", "18th March, 2025", "Oct.", ...
        if name.lower() in s or re.search(rf"\b{name[:3].lower()}\b", s):
            return name
    return ""


def _find_columns(header: List[str]) -> Dict[str, int]:
    """
    The sheets all use slightly different column names/positions, so we find each
    column by looking for a keyword in the header text instead of using fixed indexes.
    """
    wanted = {  # field -> test on the lower-cased header text
        "title":   lambda h: "title of" in h and "paper" in h,
        "authors": lambda h: "author" in h,          # first match only (see below)
        "school":  lambda h: "school" in h,
        "journal": lambda h: h.startswith("name of journal"),
        "year":    lambda h: "year of publication" in h,
        "month":   lambda h: "month of publication" in h,
        "link":    lambda h: "link for publication" in h or "link to article" in h,
        "doi":     lambda h: "doi" in h,
    }
    cols: Dict[str, int] = {}
    for field, test in wanted.items():
        for i, h in enumerate(header):
            if test(h):
                cols[field] = i
                break  # keep the first matching column only
    return cols


def _load_book_sheet(rows: List[tuple], header_idx: int, source: str) -> List[Paper]:
    """Book chapters / conference papers (the "BBC" sheet) - different columns from journals."""
    header = [str(c).lower().strip() if c is not None else "" for c in rows[header_idx]]

    def col(test):  # index of the first header matching `test`, or None
        return next((i for i, h in enumerate(header) if test(h)), None)

    cols = {
        "authors":  col(lambda h: "name of the faculty" in h),
        "book":     col(lambda h: "title of the book" in h),
        "chapter":  col(lambda h: "title of the chapter" in h),
        "proc":     col(lambda h: "title of the proceedings" in h),
        "year":     col(lambda h: "year of publication" in h),
        "date":     col(lambda h: h == "date of publication"),
        "link":     col(lambda h: h == "link"),
        "booklink": col(lambda h: h.startswith("link for book")),
    }
    papers = []
    for row in rows[header_idx + 1:]:
        def cell(field):
            i = cols.get(field)
            return row[i] if i is not None and i < len(row) else None

        book, chapter, proc = _text(cell("book")), _text(cell("chapter")), _text(cell("proc"))
        title = chapter or book or proc  # conference papers have no chapter title
        if not title:
            continue
        link = _text(cell("link")) or _text(cell("booklink"))
        non_url = URL_RE.sub(" ", " ".join(_text(c) for c in row))
        q = QUARTILE_RE.search(non_url)
        papers.append(Paper(
            title=title,
            authors=_text(cell("authors")),
            school="",  # this sheet has no school/department column, so leave it blank
            journal=book if chapter else proc,  # the book / conference it appeared in
            year=_parse_year(cell("year")) or _parse_year(cell("date")),
            month=_parse_month(cell("date")),
            doi=_find_doi(_text(cell("booklink")), link),
            link=link,
            quartile=f"Q{q.group(1)}" if q else "",
            scopus="scopus" in non_url.lower(),
            source=source,
            kind=BOOK,
        ))
    return papers


def _load_sheet(rows: List[tuple], source: str) -> List[Paper]:
    # The header row is the first row that has a "Title of ... paper" cell (journal
    # sheets) or a "Title of the chapter" cell (book chapter sheets).
    header_idx = None
    for i, row in enumerate(rows[:15]):
        cells = [str(c).lower() for c in row if c is not None]
        if any("title of the chapter" in c for c in cells):
            return _load_book_sheet(rows, i, source)
        if any("title of" in c and "paper" in c for c in cells):
            header_idx = i
            break
    if header_idx is None:
        return []

    header = [str(c).lower().strip() if c is not None else "" for c in rows[header_idx]]
    cols = _find_columns(header)
    papers = []

    for row in rows[header_idx + 1:]:
        def cell(field):
            i = cols.get(field)
            return row[i] if i is not None and i < len(row) else None

        raw_title = _text(cell("title"))
        if not raw_title:
            continue  # sub-header rows, year marker rows, blank rows

        # Some titles have the DOI/URL pasted on the end - take the DOI, drop the URL.
        title = URL_RE.sub("", raw_title).strip(" -:;")
        # Text of every cell except the title/authors; used to spot DOI, quartile, scopus.
        other = " ".join(_text(c) for j, c in enumerate(row)
                         if j not in (cols.get("title"), cols.get("authors")))
        non_url = URL_RE.sub(" ", other)

        q = QUARTILE_RE.search(non_url)
        year = _parse_year(cell("year")) or _parse_year(cell("month"))
        papers.append(Paper(
            title=title,
            authors=_text(cell("authors")),
            school=_text(cell("school")),
            journal=_text(cell("journal")),
            year=year,
            month=_parse_month(cell("month")),
            doi=_find_doi(_text(cell("doi")), raw_title, _text(cell("link")), other),
            link=_text(cell("link")),
            quartile=f"Q{q.group(1)}" if q else "",
            scopus="scopus" in other.lower(),
            source=source,
        ))
    return papers


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def load_papers(data_dir: str = DATA_DIR) -> List[Paper]:
    """Read every .xlsx in data_dir. The same paper may be in two files, so merge duplicates."""
    import openpyxl  # imported here so the rest of the project never needs it

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Research data folder not found: {data_dir}")

    by_title: Dict[str, Paper] = {}
    for filename in sorted(os.listdir(data_dir)):
        if not filename.lower().endswith(".xlsx") or filename.startswith("~$"):
            continue
        wb = openpyxl.load_workbook(os.path.join(data_dir, filename), read_only=True, data_only=True)
        for sheet in wb.worksheets:
            for paper in _load_sheet(list(sheet.iter_rows(values_only=True)), filename):
                key = _normalize_title(paper.title)
                old = by_title.get(key)
                if old is None:
                    by_title[key] = paper
                else:  # duplicate: keep the first, fill in whatever it was missing
                    for field in ("doi", "link", "quartile", "authors", "journal", "month", "year"):
                        if not getattr(old, field) and getattr(paper, field):
                            setattr(old, field, getattr(paper, field))
                    old.scopus = old.scopus or paper.scopus
        wb.close()
    return list(by_title.values())


# --------------------------------------------------------------------------- #
# Step 2: PARSE - understand the question
# --------------------------------------------------------------------------- #

@dataclass
class Query:
    intent: str = "list"                 # "list" | "count" | "yearly" | "doi"
    years: Optional[List[int]] = None
    quartile: str = ""
    scopus: bool = False
    departments: Optional[List[str]] = None
    author: str = ""                     # name words, lower-case
    kind: str = ""                       # JOURNAL / BOOK / "" (both)
    topic: str = ""                      # free-text subject, e.g. "fire retardant coatings"
    title_text: str = ""                 # for DOI lookups


# Words that end a person's name when we cut it out of a sentence.
NAME_STOP = r"\b(in|for|from|during|since|between|published|publications?|papers?|journals?|during|of|and)\b"


def parse_query(text: str) -> Query:
    q = Query()
    low = text.lower()

    # --- intent ---
    if "doi" in low:
        q.intent = "doi"
    elif re.search(r"year on year|year-on-year|year wise|year-wise|yearly|per year|each year|by year", low):
        q.intent = "yearly"
    elif re.search(r"how many|number of|count|total", low):
        q.intent = "count"

    # --- years ("this year", "last year", or explicit 4-digit years) ---
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", low)]
    if "this year" in low or "current year" in low:
        years.append(date.today().year)
    if "last year" in low or "previous year" in low:
        years.append(date.today().year - 1)
    q.years = sorted(set(years)) or None

    # --- journal quality ---
    m = QUARTILE_RE.search(text)
    q.quartile = f"Q{m.group(1)}" if m else ""
    q.scopus = "scopus" in low

    # --- journal papers only, or book chapters / conference papers only? ---
    if re.search(r"\b(books?|chapters?|conferences?|proceedings?)\b", low):
        q.kind = BOOK
    elif re.search(r"\bjournals?\b", low):
        q.kind = JOURNAL

    # --- departments / schools / domains ---
    found = [name for name, d in DEPARTMENTS.items()
             if any(re.search(rf"\b{re.escape(t)}\b", low) for t in d["query"])]
    q.departments = found or None

    # --- author: a name that follows Prof / Dr / Mr / Ms ... ---
    m = re.search(r"\b(?:prof(?:essor)?|dr|mr|mrs|ms)\b\.?\s+([a-z][a-z.\s]+)", low)
    if m:
        name = re.split(NAME_STOP, m.group(1))[0]
        q.author = " ".join(name.replace(".", " ").split())

    # --- topic: "papers about / on / related to <subject>" (stops at year, "by", etc.) ---
    m = re.search(r"\b(?:about|on|related to|regarding|concerning|topic of|field of|area of)\s+(.+)", low)
    if m and q.intent != "doi":
        topic = re.split(r"\b(?:published|in 20\d\d|this year|last year|by|from|during|since|"
                         r"scopus|q[1-4]|in the)\b", m.group(1))[0]
        topic = re.sub(r"[?.!,]+$", "", topic).strip()
        # "papers on computer science" is a department filter, not a free-text topic
        is_dept_name = any(topic == t for d in DEPARTMENTS.values() for t in d["query"])
        q.topic = "" if is_dept_name or len(topic) < 3 else topic

    # --- paper title for DOI lookups: use quoted text if there is any ---
    quoted = re.search(r"[\"“'‘](.{8,}?)[\"”'’]", text)
    q.title_text = quoted.group(1) if quoted else text
    return q


# --------------------------------------------------------------------------- #
# Step 3: ANSWER - filter and format
# --------------------------------------------------------------------------- #

def _matches_department(p: Paper, name: str) -> bool:
    d = DEPARTMENTS[name]
    if re.search(d["dept"], p.school, re.I):
        return True
    return bool(d["topic"]) and bool(re.search(d["topic"], f"{p.title} {p.journal}", re.I))


def _matches_author(p: Paper, author: str) -> bool:
    authors = re.sub(r"[^a-z ]", " ", p.authors.lower())
    return all(word in authors for word in author.split())


def filter_papers(papers: List[Paper], q: Query) -> List[Paper]:
    out = papers
    if q.kind:
        out = [p for p in out if p.kind == q.kind]
    if q.years:
        out = [p for p in out if p.year in q.years]
    if q.quartile:
        out = [p for p in out if p.quartile == q.quartile]
    if q.scopus:
        out = [p for p in out if p.scopus]
    for dept in q.departments or []:
        out = [p for p in out if _matches_department(p, dept)]
    if q.author:
        out = [p for p in out if _matches_author(p, q.author)]
    return out


def _find_by_title(papers: List[Paper], text: str) -> List[Paper]:
    """Best title matches: share of the question's words found in each title (>= 50%)."""
    stop = {"give", "me", "the", "doi", "link", "of", "for", "paper", "titled", "title", "called",
            "a", "an", "to", "what", "is", "its", "please", "named", "this", "that"}
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stop]
    if not words:
        return []
    scored = []
    for p in papers:
        title_words = set(re.findall(r"[a-z0-9]+", p.title.lower()))
        score = sum(w in title_words for w in words) / len(words)
        if score >= 0.5:
            scored.append((score, p))
    scored.sort(key=lambda s: -s[0])
    return [p for _, p in scored[:3]]


def _describe(q: Query) -> str:
    """Human-readable summary of the filters, e.g. 'Q1, Scopus, 2026, Computer Science'."""
    parts = [q.kind] if q.kind else []
    if q.quartile:
        parts.append(q.quartile)
    if q.scopus:
        parts.append("Scopus")
    if q.years:
        parts.append(", ".join(map(str, q.years)))
    parts += q.departments or []
    if q.author:
        parts.append(f"author '{q.author.title()}'")
    if q.topic:
        parts.append(f"topic '{q.topic}'")
    return ", ".join(parts) or "all records"


def _fmt(p: Paper) -> str:
    when = f"{p.month} {p.year}".strip() if p.year else (p.month or "year n/a")
    extras = [x for x in (p.quartile, "Scopus" if p.scopus else "",
                          "book chapter/conference" if p.kind == BOOK else "") if x]
    line = f"- {p.title} ({when}) | {p.journal or 'venue n/a'}"
    if extras:
        line += f" | {', '.join(extras)}"
    if p.authors:
        line += f"\n  Authors: {p.authors}"
    if p.doi_url or p.link:
        line += f"\n  Link: {p.doi_url or p.link}"
    return line


def _list_block(papers: List[Paper], keep_order: bool = False) -> str:
    if not keep_order:  # newest first, unless the order is already "most relevant first"
        papers = sorted(papers, key=lambda p: (-(p.year or 0), p.title))
    lines = [_fmt(p) for p in papers[:MAX_LIST]]
    if len(papers) > MAX_LIST:
        lines.append(f"... and {len(papers) - MAX_LIST} more (narrow the question to see them).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public class - this is what the rest of the project uses
# --------------------------------------------------------------------------- #

@dataclass
class _State:
    """Everything that is replaced together when the data changes (so readers never see a mix)."""
    papers: List[Paper]
    fingerprint: str
    index: object = None   # FAISS index, row i <-> papers[i]; None if semantic search is unavailable


def _folder_fingerprint(data_dir: str) -> str:
    """Short hash of (name, size, modified-time) of every .xlsx file. Changes when a file is
    added, removed or edited. Cheap: it only looks at file metadata, not the contents."""
    parts = []
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if name.lower().endswith(".xlsx") and not name.startswith("~$"):
                st = os.stat(os.path.join(data_dir, name))
                parts.append((name, st.st_size, st.st_mtime_ns))
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def _embedding_text(p: Paper) -> str:
    """What gets embedded for each paper (authors left out so names don't skew topic search)."""
    return f"{p.title}. {p.journal}".strip(". ")


class ResearchModule:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self._state: Optional[_State] = None
        self._lock = threading.RLock()      # only one reload at a time
        self._model = None                  # embedding model, loaded on first use
        self._watcher: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ---- loading + watching ------------------------------------------------

    def _ensure_loaded(self) -> _State:
        """Load on first use and start the folder watcher. Later calls just return the data."""
        if self._state is None:
            with self._lock:
                if self._state is None:
                    self._start_watcher()
                    self._refresh()
        return self._state

    @property
    def papers(self) -> List[Paper]:
        return self._ensure_loaded().papers

    def reload(self):
        """Force a re-read + re-embed now (the watcher does this automatically on changes)."""
        with self._lock:
            self._refresh(force=True)

    def _refresh(self, force: bool = False) -> bool:
        """Re-read the Excel files and rebuild the index. Returns True if something changed.
        If anything fails the old data stays in place."""
        with self._lock:
            fingerprint = _folder_fingerprint(self.data_dir)
            if not force and self._state and self._state.fingerprint == fingerprint:
                return False
            papers = load_papers(self.data_dir)
            index = self._load_or_build_index(papers, fingerprint)
            self._state = _State(papers, fingerprint, index)  # swap in one step
            print(f"[research] loaded {len(papers)} records"
                  f"{' + vector index' if index is not None else ' (no vector index)'}")
            return True

    def _start_watcher(self):
        if self._watcher is None:
            self._watcher = threading.Thread(target=self._watch, name="research-watcher", daemon=True)
            self._watcher.start()

    def stop_watching(self):
        self._stop.set()

    def _watch(self):
        """Background loop: every WATCH_INTERVAL seconds, check whether the folder changed."""
        last_error = ""
        while not self._stop.wait(WATCH_INTERVAL):
            try:
                seen = _folder_fingerprint(self.data_dir)
                if self._state and seen == self._state.fingerprint:
                    continue
                # Changed. Wait a moment and check again so we don't read a file that is
                # still being saved; if it is still changing we simply retry next round.
                time.sleep(1)
                if _folder_fingerprint(self.data_dir) != seen:
                    continue
                self._refresh()
                last_error = ""
            except Exception as e:  # e.g. half-written file: keep old data, try again next round
                if str(e) != last_error:
                    print(f"[research] reload failed, keeping previous data: {e}")
                    last_error = str(e)

    # ---- vector index -------------------------------------------------------

    def _embed(self, texts: List[str]):
        """Texts -> normalised float32 vectors (so inner product == cosine similarity)."""
        import numpy as np
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBED_MODEL)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    def _load_or_build_index(self, papers: List[Paper], fingerprint: str):
        """Reuse the index saved on disk if the files haven't changed, else embed everything."""
        try:
            import faiss
        except ImportError:
            return None  # semantic search simply stays off
        index_file = os.path.join(INDEX_DIR, "papers.faiss")
        meta_file = os.path.join(INDEX_DIR, "meta.json")
        try:
            # 1) saved index still matches the current files -> no embedding needed
            if os.path.exists(index_file) and os.path.exists(meta_file):
                with open(meta_file) as f:
                    meta = json.load(f)
                if (meta.get("fingerprint") == fingerprint and meta.get("model") == EMBED_MODEL
                        and meta.get("count") == len(papers)):
                    return faiss.read_index(index_file)
            # 2) otherwise embed every paper and save the new index
            if not papers:
                return None
            vectors = self._embed([_embedding_text(p) for p in papers])
            index = faiss.IndexFlatIP(vectors.shape[1])  # exact search; tiny dataset
            index.add(vectors)
            os.makedirs(INDEX_DIR, exist_ok=True)
            faiss.write_index(index, index_file)
            with open(meta_file, "w") as f:
                json.dump({"fingerprint": fingerprint, "model": EMBED_MODEL, "count": len(papers)}, f)
            return index
        except Exception as e:
            print(f"[research] vector index unavailable: {e}")
            return None

    def semantic_search(self, text: str, k: int = SEMANTIC_TOP_K,
                        min_score: float = SEMANTIC_MIN_SCORE) -> Optional[List[Paper]]:
        """Papers most similar in meaning to `text`, best first. None if the index is unavailable."""
        state = self._ensure_loaded()
        if state.index is None:
            return None
        try:
            scores, ids = state.index.search(self._embed([text]), min(k, len(state.papers)))
        except Exception as e:
            print(f"[research] semantic search failed: {e}")
            return None
        return [state.papers[i] for s, i in zip(scores[0], ids[0]) if i >= 0 and s >= min_score]

    # ---- answering ----------------------------------------------------------

    def is_research_query(self, query: str) -> bool:
        """True if the question looks like it is about publications/research output."""
        low = query.lower()
        return any(re.search(rf"\b{re.escape(k)}\b", low) for k in RESEARCH_KEYWORDS)

    def answer(self, query: str) -> str:
        """Return the answer to a research question as plain text (computed, not generated)."""
        try:
            papers = self.papers
        except Exception as e:  # missing folder, missing openpyxl, corrupt file...
            return f"The research database could not be loaded: {e}"

        q = parse_query(query)

        # "Give me the DOI of <paper>"
        if q.intent == "doi":
            hits = _find_by_title(papers, q.title_text)
            if not hits:  # exact words didn't match - try by meaning
                hits = (self.semantic_search(q.title_text, k=1, min_score=0.5) or [])
            if not hits:
                return "I couldn't find a paper matching that title in the research records."
            return "\n".join(
                f"- {p.title}\n  DOI: {p.doi_url or 'not recorded'}"
                + (f"\n  Article link: {p.link}" if p.link else "")
                for p in hits)

        # "Papers about <topic>": narrow down by meaning first (best match first),
        # then apply the exact filters (year, quartile...) on top.
        pool = papers
        if q.topic:
            found = self.semantic_search(q.topic)
            if found is None:  # no vector index: fall back to "every topic word is in the title/journal"
                words = q.topic.split()
                found = [p for p in papers if all(w in f"{p.title} {p.journal}".lower() for w in words)]
            pool = found

        matches = filter_papers(pool, q)
        header = f"Filters: {_describe(q)}."
        if q.topic:
            header += " (Topic matches are by similarity, so treat counts as approximate.)"

        if not matches:
            return f"{header} No matching publications found in the research records."

        # "Year on year publication history of <dept / person>"
        if q.intent == "yearly":
            counts: Dict[int, int] = {}
            for p in matches:
                counts[p.year or 0] = counts.get(p.year or 0, 0) + 1
            rows = [f"- {y or 'year n/a'}: {n} paper(s)" for y, n in sorted(counts.items())]
            return f"{header} Total {len(matches)} papers.\nPapers per year:\n" + "\n".join(rows)

        block = _list_block(matches, keep_order=bool(q.topic))

        # "How many ..."
        if q.intent == "count":
            return f"{header} Count: {len(matches)} paper(s).\n{block}"

        # "List the papers ..." / "publication history of Prof X"
        return f"{header} {len(matches)} paper(s) found:\n{block}"

    def get_context(self, query: str) -> str:
        """
        Same facts as answer(), wrapped as RAG context so the chatbot's LLM can phrase
        the reply. This is what you plug in where `rag_context` is built.
        """
        return (
            "Research records from GD Goenka University (computed from the official "
            "publication spreadsheets, so numbers and links are exact):\n\n"
            f"{self.answer(query)}\n\n"
            "Answer the user's question using ONLY these records. Do not add, remove or "
            "change any paper, count or link."
        )


# Module-level singleton, same pattern as the other modules in this project.
research_module = ResearchModule()


# --------------------------------------------------------------------------- #
# Command line:  python src/research_module.py "your question"
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many papers were published this year?"
    print(f"Research query? {research_module.is_research_query(question)}\n")
    print(research_module.answer(question))
