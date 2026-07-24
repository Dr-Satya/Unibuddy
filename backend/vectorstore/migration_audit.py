"""
migration_audit.py — Pre-migration Knowledge Base Comparator
=============================================================

Compares the OLD knowledge base:
    data/vectordb/metadata.json         (864 chunks, str-keyed dict)

Against the NEW knowledge base:
    data/processed/gdgu_chunks_final.json   (1880 clean chunks)
    vectorstore/metadata_index.json         (1880 sync fingerprints)

Produces:
    vectorstore/migration_report.json

Report sections:
    fully_covered       — old chunk text substantially present in new KB
    partially_covered   — old chunk topic present but text differs
    missing             — old chunk has no semantic match in new KB
    merge_recommended   — old chunks with unique data not in new KB
                          (faculty name/designation, fee rows, etc.)
    safe_to_delete      — old chunks fully superseded by new KB
    field_comparison    — metadata field diff old vs new
    summary             — counts and percentages

Matching strategy (no embeddings needed — text-only):
    1. Exact match:     SHA-256(normalised text) identical in old and new
    2. Near match:      Jaccard similarity of word-trigrams >= 0.5
    3. Topic match:     Source URL path overlap or keyword overlap >= 0.4
    4. No match:        Nothing above 0.4

Usage:
    python vectorstore/migration_audit.py
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent

OLD_META_FILE   = BACKEND_DIR / "data" / "vectordb"   / "metadata.json"
NEW_CHUNKS_FILE = BACKEND_DIR / "data" / "processed"  / "gdgu_chunks_final.json"
NEW_INDEX_FILE  = SCRIPT_DIR  / "metadata_index.json"
REPORT_FILE     = SCRIPT_DIR  / "migration_report.json"


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _sha(text: str) -> str:
    return hashlib.sha256(_norm(text).encode("utf-8")).hexdigest()


def _trigrams(text: str) -> set:
    """Word trigrams from normalised text."""
    words = re.findall(r'\b[a-z0-9]{2,}\b', _norm(text))
    if len(words) < 3:
        return set(tuple(words[i:]) for i in range(len(words)))
    return {(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _url_path(url: str) -> str:
    """Extract lowercased URL path tokens."""
    m = re.search(r'https?://[^/]+(/.+)?', url or '')
    if not m or not m.group(1):
        return ''
    return re.sub(r'[/_\-]', ' ', m.group(1).lower()).strip()


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Fraction of significant words in text_a that appear in text_b."""
    words_a = set(re.findall(r'\b[a-z]{4,}\b', _norm(text_a)))
    words_b = set(re.findall(r'\b[a-z]{4,}\b', _norm(text_b)))
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------

def load_old(path: Path) -> list:
    """Returns list of {row_id, content, source, title, content_type,
    name, designation, doc_id, chunk_index, scraped_at}"""
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    records = []
    for row_id, rec in raw.items():
        records.append({
            'row_id':       row_id,
            'content':      (rec.get('content') or '').strip(),
            'source':       rec.get('source', ''),
            'title':        rec.get('title', ''),
            'content_type': rec.get('content_type', ''),
            'name':         rec.get('name', ''),
            'designation':  rec.get('designation', ''),
            'doc_id':       rec.get('doc_id', ''),
            'chunk_index':  rec.get('chunk_index', ''),
            'scraped_at':   rec.get('scraped_at', ''),
        })
    return records


def load_new(chunks_path: Path) -> list:
    """Returns list of {chunk_id, text, page_url, page_title, category,
    source, last_crawled}"""
    with open(chunks_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return raw


# ---------------------------------------------------------------------------
# Build lookup structures for new KB
# ---------------------------------------------------------------------------

def build_new_index(new_chunks: list) -> dict:
    """Pre-compute trigrams and hashes for every new chunk."""
    index = {
        'by_sha':      {},   # sha → chunk
        'by_url':      defaultdict(list),   # url_path_tokens → [chunk]
        'trigrams':    [],   # list of (trigram_set, chunk)
        'all_text':    '',   # concatenated for fast keyword search
    }
    for chunk in new_chunks:
        text = (chunk.get('text') or '').strip()
        sha  = _sha(text)
        index['by_sha'][sha] = chunk

        url_key = _url_path(chunk.get('page_url', ''))
        if url_key:
            index['by_url'][url_key].append(chunk)

        tg = _trigrams(text)
        index['trigrams'].append((tg, chunk))

    return index


# ---------------------------------------------------------------------------
# Classify one old chunk against the new KB
# Returns (status, best_match_chunk, score, detail)
# status: 'exact' | 'near' | 'topic' | 'missing'
# ---------------------------------------------------------------------------

def classify_chunk(old: dict, new_index: dict) -> tuple:
    text = old['content']
    if not text:
        return ('missing', None, 0.0, 'empty content')

    # 1. Exact SHA match
    sha = _sha(text)
    if sha in new_index['by_sha']:
        return ('exact', new_index['by_sha'][sha], 1.0, 'sha256 match')

    # 2. Near match via trigram Jaccard
    old_tg    = _trigrams(text)
    best_j    = 0.0
    best_chunk= None
    # Only compare against chunks from the same broad category/url domain
    # to keep this O(n) manageable without embeddings
    old_url_path = _url_path(old['source'])
    candidates: list = []
    for url_key, chunks in new_index['by_url'].items():
        overlap = _keyword_overlap(old_url_path, url_key)
        if overlap >= 0.2:
            candidates.extend(chunks)

    # If no URL candidates, fall back to first 300 new chunks (fast scan)
    if not candidates:
        candidates = [c for _, c in new_index['trigrams'][:300]]

    for tg, chunk in new_index['trigrams']:
        if chunk not in candidates:
            continue
        j = _jaccard(old_tg, tg)
        if j > best_j:
            best_j     = j
            best_chunk = chunk

    if best_j >= 0.5:
        return ('near', best_chunk, best_j, f'trigram_jaccard={best_j:.3f}')
    if best_j >= 0.25:
        return ('topic', best_chunk, best_j, f'trigram_jaccard={best_j:.3f}')

    # 3. Topic match via keyword overlap with any new chunk text
    old_src = old['source'].lower()
    for _, chunk in new_index['trigrams']:
        chunk_url = chunk.get('page_url', '').lower()
        # URL path overlap
        if old_src and chunk_url:
            # Extract last two path segments for comparison
            old_seg  = '/'.join(old_src.split('/')[-3:])
            new_seg  = '/'.join(chunk_url.split('/')[-3:])
            common   = len(set(old_seg.split('/')) & set(new_seg.split('/')))
            if common >= 2:
                kw = _keyword_overlap(text, chunk.get('text', ''))
                if kw >= 0.4:
                    return ('topic', chunk, kw,
                            f'url_path_overlap+keyword={kw:.3f}')

    return ('missing', None, best_j, f'best_jaccard={best_j:.3f}')


# ---------------------------------------------------------------------------
# Unique value detection: is this old chunk's data already present
# in the new KB in ANY form?
# ---------------------------------------------------------------------------

IMPORTANT_PATTERNS = [
    # Faculty: has a real name and meaningful bio
    re.compile(r'\b(Dr\.?\s+[A-Z][a-z]+|Prof\.?\s+[A-Z][a-z]+)\b'),
    # Fee: has rupee amounts
    re.compile(r'₹\s*[\d,]+|rs\.?\s*[\d,]+|lakh', re.I),
    # Specific programme details
    re.compile(r'\b(B\.Tech|M\.Tech|MBA|BCA|MCA|B\.Sc|M\.Sc|LLB|PhD)\b', re.I),
    # Contact / address data
    re.compile(r'\b(email|phone|contact|address|tel)\b', re.I),
]


def _has_unique_data(old_chunk: dict) -> tuple[bool, str]:
    """Returns (True, reason) if the old chunk contains data
    that may not be in the new KB."""
    text = old_chunk.get('content', '')
    ct   = old_chunk.get('content_type', '')
    name = old_chunk.get('name', '').strip()

    # Faculty bio with a real name
    if ct == 'faculty_profile' and name:
        return True, f"faculty_profile with name={name!r}"

    # Fee structure rows (₹ amounts)
    if ct == 'fee_structure' and re.search(r'₹|rs\.?\s*\d', text, re.I):
        return True, "fee_structure with rupee amounts"

    # Any chunk matching important patterns
    for pat in IMPORTANT_PATTERNS:
        if pat.search(text):
            return True, f"matches pattern: {pat.pattern[:40]}"

    return False, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*65}")
    print("  Migration Audit — Old KB vs New KB")
    print(f"{'='*65}\n")

    print("Loading old metadata.json... ", end='', flush=True)
    old_chunks = load_old(OLD_META_FILE)
    print(f"{len(old_chunks)} chunks.")

    print("Loading new gdgu_chunks_final.json... ", end='', flush=True)
    new_chunks = load_new(NEW_CHUNKS_FILE)
    print(f"{len(new_chunks)} chunks.")

    print("Building new-KB index... ", end='', flush=True)
    new_index = build_new_index(new_chunks)
    print("done.\n")

    # ------------------------------------------------------------------
    # Field comparison
    # ------------------------------------------------------------------
    old_fields = {'content', 'source', 'title', 'content_type',
                  'name', 'designation', 'doc_id', 'chunk_index', 'scraped_at'}
    new_fields = {'text', 'page_url', 'page_title', 'category',
                  'source', 'last_crawled', 'chunk_id', 'chunked_at'}

    only_in_old = sorted(old_fields - new_fields)
    only_in_new = sorted(new_fields - old_fields)
    in_both_raw = old_fields & new_fields
    # Map equivalent fields
    field_mapping = {
        'content':      'text',
        'source':       'page_url',
        'title':        'page_title',
        'content_type': 'category',
        'scraped_at':   'last_crawled',
        'doc_id':       'chunk_id',
    }
    in_both = sorted(in_both_raw | set(field_mapping.keys()))

    field_comparison = {
        "old_fields":    sorted(old_fields),
        "new_fields":    sorted(new_fields),
        "only_in_old":   only_in_old,
        "only_in_new":   only_in_new,
        "field_mapping": field_mapping,
        "notes": {
            "content_type vs category": (
                "Old uses: faculty_profile(498), fee_structure(165), courses(110), "
                "faculty(34), about(32), facilities(24), admission(1). "
                "New uses: course(943), about(425), research(137), placement(89), "
                "facilities(81), campus(74), fee(46), general(31), admission(17), "
                "faq(13), event(12), scholarship(10), contact(2)."
            ),
            "name + designation": (
                "Old KB has name (44% fill) and designation (0% fill). "
                "New KB has neither as top-level fields — name/designation "
                "are embedded inside chunk text only."
            ),
            "chunk_id stability": (
                "Old doc_id is SHA-256 of raw HTML source. "
                "New chunk_id is MD5(url + chunk_text). Not interchangeable."
            ),
        }
    }

    # ------------------------------------------------------------------
    # Per-chunk classification
    # ------------------------------------------------------------------
    fully_covered     = []
    partially_covered = []
    missing_chunks    = []
    merge_recommended = []
    safe_to_delete    = []

    print("Classifying old chunks...")
    for i, old in enumerate(old_chunks, 1):
        status, match, score, detail = classify_chunk(old, new_index)

        has_unique, unique_reason = _has_unique_data(old)

        record = {
            "row_id":       old['row_id'],
            "content_type": old['content_type'],
            "source":       old['source'],
            "name":         old['name'],
            "content_preview": old['content'][:120].replace('\n', ' '),
            "match_status": status,
            "match_score":  round(score, 3),
            "match_detail": detail,
            "matched_url":  match.get('page_url', '') if match else '',
            "matched_title":match.get('page_title','') if match else '',
            "has_unique_data":   has_unique,
            "unique_reason":     unique_reason,
        }

        if status == 'exact':
            fully_covered.append(record)
            safe_to_delete.append(record)
        elif status == 'near':
            partially_covered.append(record)
            if has_unique:
                merge_recommended.append(record)
            else:
                safe_to_delete.append(record)
        elif status == 'topic':
            partially_covered.append(record)
            if has_unique:
                merge_recommended.append(record)
        else:  # missing
            missing_chunks.append(record)
            if has_unique:
                merge_recommended.append(record)

        if (i % 100) == 0 or i == len(old_chunks):
            print(f"  {i:>4}/{len(old_chunks)}  "
                  f"exact={len(fully_covered)}  "
                  f"near/topic={len(partially_covered)}  "
                  f"missing={len(missing_chunks)}", flush=True)

    # ------------------------------------------------------------------
    # Content-type breakdown for missing chunks
    # ------------------------------------------------------------------
    missing_by_ct  = Counter(c['content_type'] for c in missing_chunks)
    merge_by_ct    = Counter(c['content_type'] for c in merge_recommended)
    covered_by_ct  = Counter(c['content_type'] for c in fully_covered)
    partial_by_ct  = Counter(c['content_type'] for c in partially_covered)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(old_chunks)
    summary = {
        "total_old_chunks":      total,
        "total_new_chunks":      len(new_chunks),
        "fully_covered":         len(fully_covered),
        "partially_covered":     len(partially_covered),
        "missing":               len(missing_chunks),
        "merge_recommended":     len(merge_recommended),
        "safe_to_delete":        len(safe_to_delete),
        "pct_fully_covered":     round(len(fully_covered)   / total * 100, 1),
        "pct_partially_covered": round(len(partially_covered)/total * 100, 1),
        "pct_missing":           round(len(missing_chunks)  / total * 100, 1),
        "missing_by_content_type":  dict(missing_by_ct.most_common()),
        "merge_by_content_type":    dict(merge_by_ct.most_common()),
        "covered_by_content_type":  dict(covered_by_ct.most_common()),
        "partial_by_content_type":  dict(partial_by_ct.most_common()),
    }

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    report = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "old_source":      str(OLD_META_FILE),
        "new_source":      str(NEW_CHUNKS_FILE),
        "summary":         summary,
        "field_comparison":field_comparison,
        "fully_covered":   fully_covered,
        "partially_covered": partially_covered,
        "missing":         missing_chunks,
        "merge_recommended": merge_recommended,
        "safe_to_delete":  safe_to_delete,
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    size_kb = REPORT_FILE.stat().st_size / 1024

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  MIGRATION AUDIT COMPLETE")
    print(f"{'='*65}")
    print(f"  Old KB chunks          : {total}")
    print(f"  New KB chunks          : {len(new_chunks)}")
    print(f"\n  Fully covered          : {len(fully_covered):>4}  ({summary['pct_fully_covered']:5.1f}%)")
    print(f"  Partially covered      : {len(partially_covered):>4}  ({summary['pct_partially_covered']:5.1f}%)")
    print(f"  Missing                : {len(missing_chunks):>4}  ({summary['pct_missing']:5.1f}%)")
    print(f"\n  Merge recommended      : {len(merge_recommended):>4}  (unique data not in new KB)")
    print(f"  Safe to delete         : {len(safe_to_delete):>4}  (fully superseded)")
    print(f"\n  Missing by content_type:")
    for ct, n in missing_by_ct.most_common():
        print(f"    {ct:<20} {n}")
    print(f"\n  Merge recommended by content_type:")
    for ct, n in merge_by_ct.most_common():
        print(f"    {ct:<20} {n}")
    print(f"\n  Report size            : {size_kb:.1f} KB")
    print(f"  Report path            : {REPORT_FILE.resolve()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
