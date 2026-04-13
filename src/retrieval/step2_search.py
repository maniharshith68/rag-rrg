"""
src/retrieval/step2_search.py

Subprocess script — do NOT run directly.
Invoked exclusively by retriever.py via subprocess.run().

Responsibility:
    - Load query_vec.npy produced by step1_embed.py
    - Memory-map load faiss_image_index.bin  (IMAGE index — same-modality search)
    - Search top_k * 3 candidates (headroom for deduplication)
    - Slice text_embeddings.npy at retrieved row indices only (mmap — no full load)
    - Fetch reports + metadata from knowledge_base.csv for those rows
    - Deduplicate by study_id — one study may have multiple images → duplicate hits
    - Save top_k unique results to src/retrieval/results.json
    - Exit cleanly so OS reclaims all memory

Usage (called by retriever.py only):
    python3 step2_search.py <top_k>
"""

import sys
import gc
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS  (all derived from this file's location — no .env needed)
# ─────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent   # rag-rrg/src/retrieval/
SRC_DIR      = SCRIPT_DIR.parent                 # rag-rrg/src/
PROJECT_ROOT = SRC_DIR.parent                    # rag-rrg/

# Inputs
QUERY_VEC_FILE       = SCRIPT_DIR / "query_vec.npy"
FAISS_INDEX_FILE     = SRC_DIR / "indexing"  / "faiss_image_index.bin"
TEXT_EMBEDDINGS_FILE = SRC_DIR / "embedding" / "text_embeddings.npy"
METADATA_FILE        = SRC_DIR / "embedding" / "metadata.csv"
KNOWLEDGE_BASE_CSV   = PROJECT_ROOT / "data" / "knowledge_base.csv"

# Output
RESULTS_FILE = SCRIPT_DIR / "results.json"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [step2_search] %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "retrieval.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Validate CLI argument ──
    if len(sys.argv) < 2:
        log.error("Usage: python3 step2_search.py <top_k>")
        sys.exit(1)

    try:
        top_k = int(sys.argv[1])
        if top_k < 1:
            raise ValueError
    except ValueError:
        log.error(f"top_k must be a positive integer, got: {sys.argv[1]}")
        sys.exit(1)

    search_budget = top_k * 3
    log.info(f"top_k         : {top_k}")
    log.info(f"Search budget : {search_budget} candidates (3x for deduplication)")

    # ── Validate all required input files ──
    required = {
        "query_vec.npy"         : QUERY_VEC_FILE,
        "faiss_image_index.bin" : FAISS_INDEX_FILE,
        "text_embeddings.npy"   : TEXT_EMBEDDINGS_FILE,
        "metadata.csv"          : METADATA_FILE,
        "knowledge_base.csv"    : KNOWLEDGE_BASE_CSV,
    }
    for name, path in required.items():
        if not path.exists():
            log.error(f"Required file not found — {name}: {path}")
            sys.exit(1)

    # ── Load query vector ──
    log.info("Loading query_vec.npy …")
    query_vec = np.load(QUERY_VEC_FILE).astype("float32")   # (1, 512)
    log.info(f"Query vector: shape={query_vec.shape}  dtype={query_vec.dtype}")

    # ── Load FAISS IMAGE index (memory-mapped — stays on disk, not in RAM) ──
    log.info("Loading FAISS image index (mmap) …")
    try:
        import faiss
        index = faiss.read_index(
            str(FAISS_INDEX_FILE),
            faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
        )
        log.info(f"FAISS index: {index.ntotal} vectors, dim={index.d}")
    except Exception as e:
        log.error(f"Failed to load FAISS index: {e}")
        sys.exit(1)

    # ── Search IMAGE index — image query vs image embeddings (same modality) ──
    log.info(f"Searching for {search_budget} candidates …")
    try:
        scores, indices = index.search(query_vec, search_budget)   # (1, budget)
        scores  = scores[0].tolist()    # flatten → Python list
        indices = indices[0].tolist()   # flatten → Python list
        log.info(f"Candidates returned: {len(indices)}")
    except Exception as e:
        log.error(f"FAISS search failed: {e}")
        sys.exit(1)

    # Free FAISS from memory before loading CSV and embeddings
    del index
    gc.collect()

    # ── Load metadata.csv (3 columns, ~200K rows — lightweight) ──
    log.info("Loading metadata.csv …")
    try:
        metadata_df = pd.read_csv(METADATA_FILE)
        expected_cols = {"subject_id", "study_id", "image_path"}
        if not expected_cols.issubset(metadata_df.columns):
            log.error(f"metadata.csv missing columns. Found: {list(metadata_df.columns)}")
            sys.exit(1)
        log.info(f"Metadata: {len(metadata_df)} rows")
    except Exception as e:
        log.error(f"Failed to load metadata.csv: {e}")
        sys.exit(1)

    # ── Load knowledge_base.csv — only the 3 columns we need ──
    log.info("Loading knowledge_base.csv …")
    try:
        kb_df = pd.read_csv(KNOWLEDGE_BASE_CSV, usecols=["subject_id", "study_id", "report"])
        log.info(f"Knowledge base: {len(kb_df)} rows")
    except Exception as e:
        log.error(f"Failed to load knowledge_base.csv: {e}")
        sys.exit(1)

    # ── Memory-map text_embeddings.npy — only sliced rows pulled into RAM ──
    log.info("Memory-mapping text_embeddings.npy …")
    try:
        text_embs = np.load(str(TEXT_EMBEDDINGS_FILE), mmap_mode="r")
        log.info(f"Text embeddings: shape={text_embs.shape}")
    except Exception as e:
        log.error(f"Failed to mmap text_embeddings.npy: {e}")
        sys.exit(1)

    # ── Row-count alignment sanity check ──
    n_meta  = len(metadata_df)
    n_tembs = text_embs.shape[0]
    if n_meta != n_tembs:
        log.error(
            f"Row count mismatch — metadata: {n_meta} rows, "
            f"text_embeddings: {n_tembs} rows. "
            f"Re-run the embedding step to realign."
        )
        sys.exit(1)
    log.info(f"Row alignment check passed: {n_meta} rows in both metadata and text_embeddings.")

    # ── Deduplicate by study_id and collect top_k results ──
    log.info("Collecting top-K unique results (deduplicating by study_id) …")
    seen_study_ids = set()
    results = []

    for raw_score, row_idx in zip(scores, indices):
        # FAISS returns -1 for unfilled slots when budget > index size
        if row_idx < 0 or row_idx >= n_meta:
            log.warning(f"Skipping invalid row index: {row_idx}")
            continue

        row_meta   = metadata_df.iloc[row_idx]
        study_id   = str(row_meta["study_id"])
        subject_id = str(row_meta["subject_id"])
        image_path = str(row_meta["image_path"])

        # Skip if we've already collected a result for this study
        if study_id in seen_study_ids:
            log.debug(f"Duplicate study_id skipped: {study_id}")
            continue
        seen_study_ids.add(study_id)

        # Look up the report in knowledge_base.csv by subject_id + study_id
        kb_match = kb_df[
            (kb_df["subject_id"].astype(str) == subject_id) &
            (kb_df["study_id"].astype(str)   == study_id)
        ]

        if kb_match.empty:
            log.warning(f"No report found for subject={subject_id} study={study_id}. Skipping.")
            continue

        report = str(kb_match.iloc[0]["report"]).strip()

        # Slice the text embedding for this row only (mmap — minimal RAM touch)
        text_emb = text_embs[row_idx].tolist()   # (512,) as Python list for JSON

        results.append({
            "rank"           : len(results) + 1,
            "score"          : float(raw_score),
            "row_index"      : int(row_idx),
            "subject_id"     : subject_id,
            "study_id"       : study_id,
            "image_path"     : image_path,
            "report"         : report,
            "text_embedding" : text_emb,   # 512-dim — available for optional reranking
        })

        if len(results) >= top_k:
            break  # Stop as soon as we have top_k unique studies

    log.info(f"Unique results: {len(results)} / {top_k} requested")

    if not results:
        log.error("No valid results after deduplication. Check index/embedding alignment.")
        sys.exit(1)

    # ── Save results.json ──
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved → {RESULTS_FILE}")
    except Exception as e:
        log.error(f"Failed to save results.json: {e}")
        sys.exit(1)

    # ── Cleanup ──
    del text_embs, metadata_df, kb_df, query_vec
    gc.collect()

    log.info("step2_search.py done. Exiting — OS reclaims memory.")
    sys.exit(0)


if __name__ == "__main__":
    main()
