"""
embed_chunks.py — Phase 6 embedding generation pipeline
Reads  : data/processed/gdgu_chunks_final.json
Writes : data/processed/gdgu_embeddings.json

Features:
  - sentence-transformers/all-MiniLM-L6-v2 (dim=384)
  - Configurable batch size (default 64)
  - Auto GPU if available, CPU fallback
  - tqdm progress bar
  - Resume support: skips chunk_ids already present in output file
  - Output record format:
      {
        chunk_id,      ← preserved from input, never modified
        text,          ← original chunk text
        embedding,     ← list[float], length 384
        metadata: {    ← all other fields from the input chunk
          page_url, page_title, category, source, last_crawled, chunked_at
        }
      }

Usage:
    python crawler/embed_chunks.py
    python crawler/embed_chunks.py --batch-size 32
    python crawler/embed_chunks.py --batch-size 128 --output custom/path.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
INPUT_FILE  = BACKEND_DIR / "data" / "processed" / "gdgu_chunks_final.json"
OUTPUT_FILE = BACKEND_DIR / "data" / "processed" / "gdgu_embeddings.json"

MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH = 64

# Fields to put into metadata (everything except chunk_id and text)
METADATA_FIELDS = ["page_url", "page_title", "category",
                   "source", "last_crawled", "chunked_at"]


# ---------------------------------------------------------------------------
# Load existing embeddings for resume support
# ---------------------------------------------------------------------------
def load_existing(output_path: Path) -> dict:
    """Return {chunk_id: record} for all already-embedded chunks."""
    if not output_path.exists():
        return {}
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {r["chunk_id"]: r for r in existing}
    except (json.JSONDecodeError, KeyError):
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for gdgu_chunks_final.json"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                        help=f"Embedding batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output file path")
    args = parser.parse_args()

    batch_size  = args.batch_size
    output_path = Path(args.output) if args.output else OUTPUT_FILE

    print(f"\n{'='*65}")
    print("  Phase 6 — Embedding Generation")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Input      : {INPUT_FILE}")
    print(f"  Output     : {output_path}")
    print(f"  Batch size : {batch_size}")
    print(f"{'='*65}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model... ", end="", flush=True)
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(MODEL_NAME, device=device)
    dim    = model.get_sentence_embedding_dimension()
    print(f"OK  (device={device}, dim={dim})")

    # ── Load chunks ───────────────────────────────────────────────────────────
    print("Loading chunks... ", end="", flush=True)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"{len(chunks)} chunks loaded.")

    # ── Resume: load already-done embeddings ──────────────────────────────────
    done = load_existing(output_path)
    if done:
        print(f"Resume: {len(done)} chunks already embedded — skipping them.")

    # Filter to only chunks that still need embedding
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"Chunks to embed: {len(todo)}\n")

    if not todo:
        print("Nothing to do — all chunks already embedded.")
        _write_output(list(done.values()), output_path)
        return

    # ── Embed in batches ──────────────────────────────────────────────────────
    try:
        from tqdm import tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    texts      = [c["text"] for c in todo]
    results    = []
    t_start    = time.time()

    batch_iter = range(0, len(texts), batch_size)
    if _tqdm_available:
        batch_iter = tqdm(list(batch_iter), desc="Embedding", unit="batch")

    for i in batch_iter:
        batch_texts = texts[i: i + batch_size]
        embeddings  = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        for chunk, emb in zip(todo[i: i + batch_size], embeddings):
            record = {
                "chunk_id":  chunk["chunk_id"],
                "text":      chunk["text"],
                "embedding": emb.tolist(),
                "metadata":  {k: chunk.get(k, "") for k in METADATA_FIELDS},
            }
            results.append(record)

        if not _tqdm_available:
            done_count = len(done) + len(results)
            total      = len(chunks)
            pct        = done_count / total * 100
            print(f"  Batch {i//batch_size + 1:>4}  |  "
                  f"{done_count}/{total} ({pct:.0f}%)", flush=True)

    elapsed = time.time() - t_start

    # ── Merge resumed + new ───────────────────────────────────────────────────
    all_records = list(done.values()) + results

    # Reorder to match original chunk order
    order = {c["chunk_id"]: idx for idx, c in enumerate(chunks)}
    all_records.sort(key=lambda r: order.get(r["chunk_id"], 9999))

    # ── Write output ──────────────────────────────────────────────────────────
    _write_output(all_records, output_path)

    out_size_mb = output_path.stat().st_size / (1024 * 1024)
    avg_norm    = float(np.mean([
        np.linalg.norm(r["embedding"]) for r in all_records
    ]))

    print(f"\n{'='*65}")
    print(f"  EMBEDDING COMPLETE")
    print(f"{'='*65}")
    print(f"  Model                  : {MODEL_NAME}")
    print(f"  Device                 : {device}")
    print(f"  Total chunks           : {len(chunks)}")
    print(f"  Newly embedded         : {len(results)}")
    print(f"  Resumed from cache     : {len(done)}")
    print(f"  Embedding dimension    : {dim}")
    print(f"  Avg embedding norm     : {avg_norm:.4f}")
    print(f"  Elapsed time           : {elapsed:.1f}s")
    print(f"  Throughput             : {len(results)/elapsed:.1f} chunks/s")
    print(f"  Output file size       : {out_size_mb:.2f} MB")
    print(f"  Output path            : {output_path.resolve()}")
    print(f"{'='*65}\n")


def _write_output(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records → {path}")


if __name__ == "__main__":
    main()
