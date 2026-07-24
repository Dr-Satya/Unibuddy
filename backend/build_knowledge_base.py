# encoding: utf-8
"""
build_knowledge_base.py  —  Module 1 knowledge ingestion pipeline.

Standalone script. Does NOT import from src.*
Runs independently of the live API.

Sources processed (in order):
  1. Current FAISS index  → deduplicate + clean existing knowledge
  2. faculty_data_20250907_180419.json  → clean faculty bios
  3. data/raw/gdgoenka_fee_structure.txt → structured fee chunks

Output:
  data/vectordb/index.faiss       (replaces current)
  data/vectordb/metadata.json     (replaces current)

Backups created before replacement:
  data/vectordb/index.faiss.bak
  data/vectordb/metadata.json.bak

Usage:
  cd backend
  python build_knowledge_base.py
"""

import os
import sys
import re
import json
import shutil
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_kb")

# ---------------------------------------------------------------------------
# Paths  (all relative to the backend/ directory)
# ---------------------------------------------------------------------------
BASE          = os.path.dirname(os.path.abspath(__file__))
VDB_DIR       = os.path.join(BASE, "data", "vectordb")
INDEX_PATH    = os.path.join(VDB_DIR, "index.faiss")
META_PATH     = os.path.join(VDB_DIR, "metadata.json")
INDEX_BAK     = os.path.join(VDB_DIR, "index.faiss.bak")
META_BAK      = os.path.join(VDB_DIR, "metadata.json.bak")
FACULTY_JSON  = os.path.join(BASE, "faculty_data_20250907_180419.json")
FEE_TXT       = os.path.join(BASE, "data", "raw", "gdgoenka_fee_structure.txt")

# ---------------------------------------------------------------------------
# Non-faculty URL slugs to skip from faculty JSON
# ---------------------------------------------------------------------------
_NON_FACULTY_SLUGS = {
    "btech-computer-science-and-engineering",
    "btech-computer-science-and-engineering-with-specialization-in-aiml-in-association-with-ibm",
    "bachelor-of-computer-applications-bca-with-specialization-in-data-analytics-in-association-with-microsoft",
    "b-tech-aerospace-engineering",
}

# ---------------------------------------------------------------------------
# Chunk size constants (match existing index settings)
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 400   # chars — slightly larger than 300 to carry full sentences
CHUNK_OVERLAP = 80    # chars

# ---------------------------------------------------------------------------
# Garbage detection patterns (navigation / footer / boilerplate)
# ---------------------------------------------------------------------------
_NAV_RE = re.compile(
    r"menu about us|about gd goenka university|vision and mission"
    r"|governance organogram|mandatory disclosures|recognitions and affiliations"
    r"|announcement student handbook|events orientation reva freshers"
    r"|events campus life|iqac naac iiqa|naac ssr data"
    r"|consultancy and corporate training cuet"
    r"|resource centre vision and mission corporate resource"
    r"|process & efforts gdgu placements past record"
    r"|bank of credits \(abc\)|html->link\("
    r"|profile extended profile naac ssr"
    r"|clubs & committees techno-cultural"
    r"|ideathon udyami bazaar magazine unibuzz"
    r"|feedback analysis po & pso cos green energy"
    r"|governance organogram mandatory disclosure",
    re.I,
)
_FOOTER_RE = re.compile(
    r"admission cell g d goenka education city sohna gurgaon road"
    r"|navigate on google map|admissions@gdgoenka\.ac\.in"
    r"|designing institutes in delhi|best b\.ed college in delhi"
    r"|top engineering college in delhi|top hotel management colleges delhi"
    r"|goenka leadership contact us schools at gd goenka"
    r"|most viewed courses at gd goenka"
    r"|bachelor of pharmacy bca / bca \(h\)"
    r"|contact g d goenka education city sohna gurgaon road sohna, haryana",
    re.I,
)
_LOW_INFO_RE = re.compile(
    r"^faculty profile -\s*\w+(\s+\w+)*\s*(qualification|assistant professor"
    r"|associate professor|professor and head)?$",
    re.I,
)

def _is_garbage(text: str) -> str:
    """Return 'nav', 'footer', 'low_info', or '' (keep)."""
    c = text.strip()
    cn = c.lower()
    if _FOOTER_RE.search(cn):
        return "footer"
    if _NAV_RE.search(cn):
        return "nav"
    if len(c) < 60 and _LOW_INFO_RE.match(c):
        return "low_info"
    if len(c) < 30:
        return "low_info"
    return ""

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    norm = " ".join(text.split()).lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def _doc_id(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:40]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_to_name(slug: str) -> str:
    """
    Convert a URL slug to a human-readable faculty name.
    dr-bhagat-singh        → Dr. Bhagat Singh
    ms-alina-banerjee      → Ms. Alina Banerjee
    mr-yogesh-kumar        → Mr. Yogesh Kumar
    mrfakhruddin-ali-ahmad → Mr. Fakhruddin Ali Ahmad
    sapna-sharma           → Sapna Sharma   (no prefix)
    """
    s = slug.strip().lower()
    prefix = ""
    if s.startswith("dr-"):
        prefix, s = "Dr.", s[3:]
    elif s.startswith("ms-"):
        prefix, s = "Ms.", s[3:]
    elif s.startswith("mr-") and not s.startswith("mrfakhruddin"):
        prefix, s = "Mr.", s[3:]
    elif s.startswith("mrfakhruddin"):
        prefix, s = "Mr.", s[2:]   # mrfakhruddin → fakhruddin
    name_parts = [p.capitalize() for p in s.split("-") if p]
    name = " ".join(name_parts)
    return f"{prefix} {name}".strip() if prefix else name


def _extract_designation(text: str) -> str:
    """Extract designation from first 800 chars of full_content."""
    snippet = text[:800]
    for pat in (
        r"Professor and Head",
        r"Dean,?\s+School",
        r"Associate Professor",
        r"Assistant Professor",
        r"Professor",
        r"\bDean\b",
    ):
        m = re.search(pat, snippet, re.I)
        if m:
            raw = m.group(0).strip()
            # Normalise spacing
            return re.sub(r"\s+", " ", raw)
    return ""


def _chunk_text(text: str, size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text using LangChain's RecursiveCharacterTextSplitter.
    chunk_size=900, chunk_overlap=150 — no infinite-loop risk.
    """
    text = text.strip()
    if not text:
        return []
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]

# ---------------------------------------------------------------------------
# Chunk record builder
# ---------------------------------------------------------------------------

def _make_chunk(content: str, content_type: str, source: str,
                title: str, doc_seed: str, chunk_index: int,
                scraped_at: str, name: str = "",
                designation: str = "") -> Dict[str, Any]:
    return {
        "content":      content,
        "content_type": content_type,
        "source":       source,
        "title":        title,
        "doc_id":       _doc_id(doc_seed),
        "chunk_index":  chunk_index,
        "scraped_at":   scraped_at,
        "name":         name,
        "designation":  designation,
        "metadata":     {},
    }

# ---------------------------------------------------------------------------
# SOURCE 1 — Clean and deduplicate the existing FAISS index
# ---------------------------------------------------------------------------

def load_existing_chunks() -> List[Dict[str, Any]]:
    """
    Load metadata.json, deduplicate by content hash,
    discard navigation / footer / low-info chunks,
    and return a list of clean chunk dicts.
    """
    log.info("Loading existing FAISS metadata …")
    with open(META_PATH, "r", encoding="utf-8") as f:
        raw_meta: Dict[str, Any] = json.load(f)

    log.info(f"  Raw vectors in index : {len(raw_meta)}")

    seen_hashes: set = set()
    kept: List[Dict[str, Any]] = []
    stats = {"dup": 0, "nav": 0, "footer": 0, "low_info": 0, "kept": 0}

    for _idx, entry in raw_meta.items():
        content = entry.get("content", "")
        h = _content_hash(content)

        # exact duplicate
        if h in seen_hashes:
            stats["dup"] += 1
            continue
        seen_hashes.add(h)

        # garbage check
        reason = _is_garbage(content)
        if reason:
            stats[reason] = stats.get(reason, 0) + 1
            continue

        # Build normalised chunk using current schema
        chunk = {
            "content":      content,
            "content_type": entry.get("content_type", ""),
            "source":       entry.get("source", ""),
            "title":        entry.get("title", ""),
            "doc_id":       entry.get("doc_id", ""),
            "chunk_index":  int(entry.get("chunk_index", 0)),
            "scraped_at":   entry.get("scraped_at", ""),
            "name":         entry.get("name", ""),
            "designation":  entry.get("designation", ""),
            "metadata":     entry.get("metadata", {}),
        }
        kept.append(chunk)
        stats["kept"] += 1

    log.info(f"  Duplicates removed   : {stats['dup']}")
    log.info(f"  Navigation removed   : {stats.get('nav', 0)}")
    log.info(f"  Footer removed       : {stats.get('footer', 0)}")
    log.info(f"  Low-info removed     : {stats.get('low_info', 0)}")
    log.info(f"  Chunks kept          : {stats['kept']}")
    return kept

# ---------------------------------------------------------------------------
# SOURCE 2 — Faculty JSON  →  clean bio chunks
# ---------------------------------------------------------------------------

def _extract_bio(full_content: str) -> str:
    """
    Extract the biography section from full_content.
    START: first occurrence of 'Introduction:'
    END:   first of 'Contact us', 'admissions@gdgu', end of string
    """
    fc = full_content
    intro_pos = fc.find("Introduction:")
    if intro_pos == -1:
        return ""
    start = intro_pos + len("Introduction:")

    end = len(fc)
    for marker in ("Contact us\n", "Contact us ", "\nContact us",
                   "admissions@gdgu", "admissions@gdgoenka"):
        p = fc.find(marker, start)
        if p != -1 and p < end:
            end = p

    bio = fc[start:end].strip()
    # Strip unicode non-breaking spaces and control chars
    bio = re.sub(r"[\u00a0\u200b\u200c\u200d\ufeff]", " ", bio)
    bio = re.sub(r"\s{3,}", "\n\n", bio)
    return bio.strip()


def _extract_publications(full_content: str) -> str:
    """
    Extract the publications section.
    Looks for actual journal/conference content (not the nav-menu 'Publications' link).
    """
    fc = full_content
    # Look for patterns that indicate real publication content
    pub_indicators = [
        r"Journal of|Conference on|Proceedings of|IEEE |Springer|Elsevier"
        r"|SCI|Scopus|DOI:|ISSN:|Vol\.|pp\.\s*\d",
    ]
    combined = re.compile("|".join(pub_indicators), re.I)
    m = combined.search(fc)
    if not m:
        return ""

    # Walk back to find start of publications block
    start = m.start()
    # Try to find a "Publications" header before the first indicator
    pub_header = fc.rfind("Publications", max(0, start - 500), start)
    if pub_header != -1:
        start = pub_header

    # Find end — next section header or contact section
    end = len(fc)
    for marker in ("Contact us", "admissions@gdgu", "admissions@gdgoenka",
                   "\nResearch work\n", "\nAwards\n", "\nHonours\n"):
        p = fc.find(marker, start + 20)
        if p != -1 and p < end:
            end = p

    pubs = fc[start:end].strip()
    pubs = re.sub(r"[\u00a0\u200b\u200c\u200d\ufeff]", " ", pubs)
    pubs = re.sub(r"\s{3,}", "\n\n", pubs)
    return pubs.strip() if len(pubs) > 60 else ""


def _extract_research_interests(full_content: str) -> str:
    """Extract research interest statements if present."""
    m = re.search(
        r"(Research Interest[s]?\s*(?:Areas?)?[:\-–]?\s*.{20,400}?)(?:\n\n|\Z)",
        full_content, re.I | re.DOTALL
    )
    if m:
        text = m.group(0).strip()
        text = re.sub(r"[\u00a0\u200b\u200c\u200d\ufeff]", " ", text)
        return text[:600].strip()
    return ""


def build_faculty_chunks() -> List[Dict[str, Any]]:
    """
    Process faculty_data_20250907_180419.json and return clean chunks.

    Per faculty member we create up to 3 chunk groups:
      1. biography   (always, if bio > 100 chars)
      2. publications (if real publication content found)
      3. research interests (if present)
    """
    log.info("Processing faculty JSON …")
    with open(FACULTY_JSON, "r", encoding="utf-8") as f:
        entries = json.load(f)

    chunks: List[Dict[str, Any]] = []
    skipped_non_faculty = 0
    skipped_no_bio = 0
    processed = 0

    for entry in entries:
        url  = entry.get("url", "")
        slug = url.rstrip("/").split("/")[-1]

        # Skip non-faculty course pages
        if slug in _NON_FACULTY_SLUGS:
            skipped_non_faculty += 1
            continue

        full_content = entry.get("full_content", "")
        if not full_content:
            skipped_no_bio += 1
            continue

        name        = _slug_to_name(slug)
        designation = _extract_designation(full_content)
        scraped_at  = entry.get("scraped_at", _now_iso())
        title       = f"Faculty Profile — {name}"
        doc_seed    = url   # stable identifier

        # ── 1. Biography ────────────────────────────────────────────────────
        bio = _extract_bio(full_content)
        if not bio or len(bio) < 100:
            skipped_no_bio += 1
            continue

        # Prefix each bio chunk with the faculty name for retrieval clarity
        bio_prefixed = f"{name}"
        if designation:
            bio_prefixed += f", {designation}"
        bio_prefixed += f"\n\n{bio}"

        for ci, chunk_text in enumerate(_chunk_text(bio_prefixed, CHUNK_SIZE, CHUNK_OVERLAP)):
            reason = _is_garbage(chunk_text)
            if reason:
                continue
            chunks.append(_make_chunk(
                content=chunk_text,
                content_type="faculty_profile",
                source=url,
                title=title,
                doc_seed=doc_seed + "_bio",
                chunk_index=ci,
                scraped_at=scraped_at,
                name=name,
                designation=designation,
            ))

        # ── 2. Publications ─────────────────────────────────────────────────
        pubs = _extract_publications(full_content)
        if pubs and len(pubs) > 60:
            pubs_prefixed = f"{name} — Publications\n\n{pubs}"
            for ci, chunk_text in enumerate(_chunk_text(pubs_prefixed, CHUNK_SIZE, CHUNK_OVERLAP)):
                if not _is_garbage(chunk_text):
                    chunks.append(_make_chunk(
                        content=chunk_text,
                        content_type="faculty_profile",
                        source=url,
                        title=f"{title} — Publications",
                        doc_seed=doc_seed + "_pub",
                        chunk_index=ci,
                        scraped_at=scraped_at,
                        name=name,
                        designation=designation,
                    ))

        # ── 3. Research interests ───────────────────────────────────────────
        ri = _extract_research_interests(full_content)
        if ri and len(ri) > 40:
            ri_prefixed = f"{name} — Research Interests\n\n{ri}"
            for ci, chunk_text in enumerate(_chunk_text(ri_prefixed, CHUNK_SIZE, CHUNK_OVERLAP)):
                if not _is_garbage(chunk_text):
                    chunks.append(_make_chunk(
                        content=chunk_text,
                        content_type="faculty_profile",
                        source=url,
                        title=f"{title} — Research",
                        doc_seed=doc_seed + "_ri",
                        chunk_index=ci,
                        scraped_at=scraped_at,
                        name=name,
                        designation=designation,
                    ))

        processed += 1

    log.info(f"  Faculty processed          : {processed}")
    log.info(f"  Skipped (non-faculty URL)  : {skipped_non_faculty}")
    log.info(f"  Skipped (no bio extracted) : {skipped_no_bio}")
    log.info(f"  Faculty chunks created     : {len(chunks)}")
    return chunks

# ---------------------------------------------------------------------------
# SOURCE 3 — Fee structure TXT  →  programme-level chunks
# ---------------------------------------------------------------------------

# School heading → canonical name mapping
_SCHOOL_CANON = {
    "centre of excellence": "Centre of Excellence in Occupational Health, Safety, Fire & Environment (C-OHSFE)",
    "school of agricultural": "School of Agricultural Sciences",
    "school of engineering": "School of Engineering & Sciences",
    "school of healthcare": "School of Healthcare and Allied Sciences",
    "school of hospitality": "School of Hospitality & Tourism",
    "school of law": "School of Law",
    "school of liberal": "School of Liberal Arts",
    "school of management": "School of Management",
    "uid school": "UID School of Design",
}

FEE_SOURCE_URL  = "https://www.gdgoenkauniversity.com/admissions/fee-structure"
FEE_SCRAPED_AT  = "2025-09-02T23:07:46.377415"
FEE_TITLE       = "Annual Fee Structure 2025-26 | GD Goenka University"


def _parse_fee_txt() -> List[Dict[str, Any]]:
    """
    Parse gdgoenka_fee_structure.txt into programme-level fee chunks.

    The file stores newlines as literal '\\n' sequences (confirmed: 0 real
    newlines, 569 escaped ones).  After normalisation the structure is:

        [header lines 0-3]
        [school dropdown list lines 4-~21]   ← skip
        [blank / boilerplate lines ~22-~119] ← skip
        School heading                        ← marks start of real data
        Programmes
        Duration
        Annual Fee
        Admission Charges
        Security Deposit
        <programme name>
        <duration e.g. "4 Years">
        <annual fee e.g. "₹2,65,000">
        <admission charges e.g. "₹25,000">
        <security deposit e.g. "₹25,000">
        … repeat per programme …
        Next school heading
        …

    Each programme becomes one chunk:
        "School of Engineering & Sciences — Fee Structure\n
         Programme: B.Tech. Computer Science and Engineering\n
         Duration: 4 Years | Annual Fee: ₹2,65,000 | Admission: ₹25,000 | Security Deposit: ₹25,000"
    """
    with open(FEE_TXT, "r", encoding="utf-8") as f:
        raw = f.read()

    # Normalise escaped newlines
    text = raw.replace("\\n", "\n").replace("\\t", "\t")

    lines = [l.strip() for l in text.split("\n")]
    # Remove empty lines and known boilerplate
    skip_exact = {
        "Programmes", "Duration", "Annual Fee", "Admission Charges",
        "Security Deposit", "Fee", "Structure", "GD Goenka University",
        "Fee Structure for Session", "2025 - 2026", "Select School",
        "All", "Select Your Course", "Select",
        "Annual Fee", "Interest (%)", "Select Loan Duration",
        "Total Principal Amount", "Total Interest",
        "*Subject to Approval",
    }
    skip_pat = re.compile(
        r"^(\d+ year$|EMI Calculator|Title:|URL:|Scraped:|-----|"
        r"\d+ year$|1 year|2 year|3 year|4 year|5 year|6 year|"
        r"7 year|8 year|9 year|10 year|11 year|12 year)$",
        re.I,
    )

    clean = []
    for ln in lines:
        if not ln:
            continue
        if ln in skip_exact:
            continue
        if skip_pat.match(ln):
            continue
        clean.append(ln)

    # ── Find where actual data starts (second occurrence of school headers) ──
    # First block (lines ~4-21) is a dropdown list — skip it.
    # Second block starts the real per-school tables.
    school_header_re = re.compile(
        r"^(Centre of Excellence|School of (Agricultural|Engineering|Healthcare"
        r"|Hospitality|Law|Liberal|Management)|UID School)",
        re.I,
    )
    header_positions = [i for i, l in enumerate(clean) if school_header_re.match(l)]

    if len(header_positions) < 2:
        log.warning("Fee txt: could not find second school header block — using all lines")
        data_start = 0
    else:
        # First occurrence is the dropdown list; second is real data
        # Find the index where we see the same header repeated
        # Typically the split is around the midpoint
        mid = len(header_positions) // 2
        data_start = header_positions[mid]

    data_lines = clean[data_start:]
    log.info(f"  Fee data lines to parse : {len(data_lines)}")

    # ── Parse school + programme rows ────────────────────────────────────────
    rupee_re    = re.compile(r"^₹[\d,]+$")
    duration_re = re.compile(r"^\d+[\d\s/]*Years?", re.I)

    chunks: List[Dict[str, Any]] = []
    current_school = "GD Goenka University"
    i = 0
    doc_seed_base = FEE_SOURCE_URL

    while i < len(data_lines):
        line = data_lines[i]

        # Detect school heading
        if school_header_re.match(line):
            for key, canon in _SCHOOL_CANON.items():
                if key in line.lower():
                    current_school = canon
                    break
            else:
                current_school = line.strip()
            i += 1
            continue

        # Detect start of a programme row:
        # pattern: programme_name \n duration \n fee \n charges \n deposit
        # Look-ahead: next line should be a duration, and line+2 a rupee amount
        if (i + 2 < len(data_lines)
                and duration_re.match(data_lines[i + 1])
                and rupee_re.match(data_lines[i + 2])):

            programme = line.strip()
            duration  = data_lines[i + 1].strip()
            ann_fee   = data_lines[i + 2].strip()

            adm_charge = ""
            sec_dep    = ""
            if i + 3 < len(data_lines) and rupee_re.match(data_lines[i + 3]):
                adm_charge = data_lines[i + 3].strip()
            if i + 4 < len(data_lines) and rupee_re.match(data_lines[i + 4]):
                sec_dep = data_lines[i + 4].strip()

            content_lines = [
                f"{current_school} — Fee Structure",
                f"Programme: {programme}",
                f"Duration: {duration}",
                f"Annual Fee: {ann_fee}",
            ]
            if adm_charge:
                content_lines.append(f"Admission Charges: {adm_charge}")
            if sec_dep:
                content_lines.append(f"Security Deposit: {sec_dep}")

            content = "\n".join(content_lines)
            ci = len(chunks)
            chunks.append(_make_chunk(
                content=content,
                content_type="fee_structure",
                source=FEE_SOURCE_URL,
                title=FEE_TITLE,
                doc_seed=f"{doc_seed_base}_{current_school}_{programme}",
                chunk_index=ci,
                scraped_at=FEE_SCRAPED_AT,
            ))
            advance = 3
            if adm_charge:
                advance += 1
            if sec_dep:
                advance += 1
            i += advance
            continue

        i += 1

    log.info(f"  Fee chunks created : {len(chunks)}")
    return chunks

# ---------------------------------------------------------------------------
# MERGE — combine all sources, final dedup pass
# ---------------------------------------------------------------------------

def merge_and_deduplicate(
    existing:  List[Dict[str, Any]],
    faculty:   List[Dict[str, Any]],
    fee:       List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge all chunk lists.
    New sources (faculty, fee) take precedence over old existing chunks
    for the same content hash — existing chunks for the same source URL
    are dropped in favour of the freshly extracted ones.

    Final dedup is by content hash; first occurrence wins.
    """
    log.info("Merging and deduplicating all sources …")

    # Build a set of source URLs that are being fully replaced by new data
    replaced_sources: set = set()
    for c in faculty:
        replaced_sources.add(c["source"])

    # Drop existing chunks whose source URL is being replaced
    existing_filtered = [
        c for c in existing
        if c["source"] not in replaced_sources
    ]
    dropped = len(existing) - len(existing_filtered)
    if dropped:
        log.info(f"  Dropped {dropped} existing chunks superseded by fresh faculty data")

    # Concatenate: existing (filtered) + faculty + fee
    # Order matters: faculty and fee go last so their hashes win on conflict
    combined = existing_filtered + faculty + fee

    seen: set = set()
    final: List[Dict[str, Any]] = []
    for chunk in combined:
        h = _content_hash(chunk["content"])
        if h in seen:
            continue
        seen.add(h)
        final.append(chunk)

    log.info(f"  Total before final dedup : {len(combined)}")
    log.info(f"  Total after  final dedup : {len(final)}")

    # Log breakdown by content_type
    from collections import Counter
    ct = Counter(c["content_type"] for c in final)
    for k, v in ct.most_common():
        log.info(f"    {k:30s}: {v}")

    return final


# ---------------------------------------------------------------------------
# EMBED — generate vectors using the same model as runtime
# ---------------------------------------------------------------------------

def embed_chunks(chunks: List[Dict[str, Any]]) -> "np.ndarray":
    """
    Embed all chunks using SentenceTransformer('all-MiniLM-L6-v2').
    Returns a float32 numpy array of shape (N, 384).
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        log.error(f"Missing dependency: {exc}")
        sys.exit(1)

    log.info("Loading embedding model: all-MiniLM-L6-v2 …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["content"] for c in chunks]
    total = len(texts)
    log.info(f"Embedding {total} chunks …")

    BATCH = 64
    all_vecs = []
    for start in range(0, total, BATCH):
        batch = texts[start : start + BATCH]
        vecs  = model.encode(batch, convert_to_numpy=True,
                              show_progress_bar=False)
        all_vecs.append(vecs)
        done = min(start + BATCH, total)
        if done % 500 == 0 or done == total:
            log.info(f"  Embedded {done}/{total}")

    return np.vstack(all_vecs).astype("float32")


# ---------------------------------------------------------------------------
# SAVE — write index.faiss + metadata.json (with backups)
# ---------------------------------------------------------------------------

def save_index(chunks: List[Dict[str, Any]],
               vectors: "np.ndarray") -> None:
    """
    1. Back up existing index.faiss and metadata.json.
    2. Build a fresh faiss.IndexFlatL2(384).
    3. Add all vectors.
    4. Write index.faiss and metadata.json.
    """
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        log.error(f"Missing dependency: {exc}")
        sys.exit(1)

    os.makedirs(VDB_DIR, exist_ok=True)

    # ── Backup existing files ─────────────────────────────────────────────
    for src, dst in ((INDEX_PATH, INDEX_BAK), (META_PATH, META_BAK)):
        if os.path.exists(src):
            shutil.copy2(src, dst)
            log.info(f"  Backup created: {os.path.basename(dst)}")

    # ── Build FAISS index ─────────────────────────────────────────────────
    dim   = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)
    log.info(f"  Saved index.faiss  ({index.ntotal} vectors, dim={dim})")

    # ── Build metadata dict ───────────────────────────────────────────────
    meta_out: Dict[str, Any] = {}
    for i, chunk in enumerate(chunks):
        meta_out[str(i)] = {
            "content":      chunk["content"],
            "source":       chunk["source"],
            "title":        chunk["title"],
            "content_type": chunk["content_type"],
            "chunk_index":  chunk["chunk_index"],
            "doc_id":       chunk["doc_id"],
            "scraped_at":   chunk["scraped_at"],
            "name":         chunk.get("name", ""),
            "designation":  chunk.get("designation", ""),
            "metadata":     chunk.get("metadata", {}),
        }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    size_kb = os.path.getsize(META_PATH) // 1024
    log.info(f"  Saved metadata.json ({len(meta_out)} entries, {size_kb} KB)")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("Module 1 — Knowledge Base Builder")
    log.info("=" * 60)

    # ── Validate required source files ────────────────────────────────────
    missing = []
    for path, label in (
        (INDEX_PATH,   "data/vectordb/index.faiss"),
        (META_PATH,    "data/vectordb/metadata.json"),
        (FACULTY_JSON, "faculty_data_20250907_180419.json"),
        (FEE_TXT,      "data/raw/gdgoenka_fee_structure.txt"),
    ):
        if not os.path.exists(path):
            missing.append(label)
    if missing:
        for m in missing:
            log.error(f"Required file not found: {m}")
        sys.exit(1)

    # ── Source 1: existing FAISS index ────────────────────────────────────
    log.info("")
    log.info("STEP 1 — Loading and cleaning existing index …")
    existing_chunks = load_existing_chunks()

    # ── Source 2: faculty JSON ────────────────────────────────────────────
    log.info("")
    log.info("STEP 2 — Processing faculty data …")
    faculty_chunks = build_faculty_chunks()

    # ── Source 3: fee structure TXT ───────────────────────────────────────
    log.info("")
    log.info("STEP 3 — Parsing fee structure …")
    fee_chunks = _parse_fee_txt()

    # ── Merge + final dedup ───────────────────────────────────────────────
    log.info("")
    log.info("STEP 4 — Merging all sources …")
    final_chunks = merge_and_deduplicate(existing_chunks, faculty_chunks, fee_chunks)

    if not final_chunks:
        log.error("No chunks produced — aborting.")
        sys.exit(1)

    # ── Embed ─────────────────────────────────────────────────────────────
    log.info("")
    log.info("STEP 5 — Generating embeddings …")
    vectors = embed_chunks(final_chunks)

    # Sanity check
    assert vectors.shape[0] == len(final_chunks), (
        f"Vector count mismatch: {vectors.shape[0]} vs {len(final_chunks)}"
    )

    # ── Save ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("STEP 6 — Saving index and metadata …")
    save_index(final_chunks, vectors)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("BUILD COMPLETE")
    log.info(f"  Total chunks in new index : {len(final_chunks)}")
    log.info(f"  Backups at:")
    log.info(f"    {os.path.relpath(INDEX_BAK, BASE)}")
    log.info(f"    {os.path.relpath(META_BAK,  BASE)}")
    log.info(f"  New index at:")
    log.info(f"    {os.path.relpath(INDEX_PATH, BASE)}")
    log.info(f"    {os.path.relpath(META_PATH,  BASE)}")
    log.info("=" * 60)
    log.info("Restart the FastAPI server to load the new index.")


if __name__ == "__main__":
    main()
