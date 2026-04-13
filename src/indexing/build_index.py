"""
build_index.py
--------------
Builds a FAISS index over IMAGE embeddings from src/embedding/image_embeddings.npy.

Why index IMAGE embeddings (not text)?
    Previous approach indexed text embeddings and searched cross-modally
    (image query → text index). This version uses image-to-image search:

        Query X-ray → image embedding
                    → search IMAGE index → top-K most visually similar X-rays
                    → fetch their text embeddings + reports by row index
                    → use as RAG context for generation

    Image-to-image search is more accurate because:
    - Same modality comparison (no cross-modal gap)
    - Finds genuinely visually similar chest X-rays
    - Their reports are the most relevant clinical references

Pipeline position:
    embedding.py → [build_index.py] → retriever → generator

Config  : config/indexing.yml
Env     : .env
Logging : logs/indexing.log

Output files:
    src/indexing/faiss_image_index.bin  — serialized FAISS index over image embeddings
    src/indexing/index_config.json      — index metadata for retrieval module
"""

import os
import gc
import json
import logging
import numpy as np
import faiss
import psutil
import yaml
from pathlib import Path
from dotenv import load_dotenv


# ─────────────────────────────────────────────
# 0. LOAD ENVIRONMENT + CONFIG
# ─────────────────────────────────────────────

# build_index.py lives at src/indexing/build_index.py
# PROJECT_ROOT is 2 levels up → rag-rrg/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(f"Project root resolved to: {PROJECT_ROOT}")

# Load .env from project root
dotenv_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Load indexing.yml from config/
CONFIG_PATH = PROJECT_ROOT / "config" / "indexing.yml"
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

# ── Resolve all paths relative to PROJECT_ROOT ──
IMAGE_EMB_PATH    = PROJECT_ROOT / CFG["input"]["image_embeddings"]
TEXT_EMB_PATH     = PROJECT_ROOT / CFG["input"]["text_embeddings"]
METADATA_PATH     = PROJECT_ROOT / CFG["input"]["metadata"]
KB_CSV_PATH       = PROJECT_ROOT / CFG["input"]["knowledge_base"]

INDEXING_DIR      = PROJECT_ROOT / "src" / "indexing"
INDEXING_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH  = PROJECT_ROOT / CFG["index"]["index_file"]
INDEX_CONFIG_PATH = PROJECT_ROOT / CFG["index"]["config_file"]

EMBEDDING_DIM     = int(CFG["index"]["embedding_dim"])
CHUNK_SIZE        = int(CFG["memory"]["chunk_size"])
RAM_LIMIT         = float(CFG["memory"]["ram_limit_percent"])

LOG_FILE          = PROJECT_ROOT / CFG["logging"]["log_file"]
LOG_LEVEL         = CFG["logging"]["level"]
LOG_CONSOLE       = CFG["logging"]["console"]

RUN_SELF_CHECK    = CFG["sanity_check"]["run_self_retrieval"]
NUM_SAMPLES       = int(CFG["sanity_check"]["num_samples"])


# ─────────────────────────────────────────────
# 1. LOGGING SETUP
# Logs to both logs/indexing.log and console.
# ─────────────────────────────────────────────

def setup_logging():
    """
    Configures logging to write to logs/indexing.log and optionally console.
    Log directory is created automatically if it does not exist.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    handlers = [logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")]
    if LOG_CONSOLE:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 2. RAM MONITOR
# ─────────────────────────────────────────────

def check_ram(label="", logger=None):
    """
    Logs current RAM usage. Forces gc if above RAM_LIMIT threshold.
    """
    ram      = psutil.virtual_memory()
    used_pct = ram.percent
    used_gb  = ram.used      / (1024 ** 3)
    avail_gb = ram.available / (1024 ** 3)
    total_gb = ram.total     / (1024 ** 3)

    msg = (f"[RAM:{label}] used={used_gb:.2f}GB  free={avail_gb:.2f}GB  "
           f"total={total_gb:.2f}GB  ({used_pct:.1f}%)")

    if logger:
        logger.info(msg)
    else:
        print(msg)

    if used_pct > RAM_LIMIT:
        warn = (f"[RAM WARNING] {used_pct:.1f}% > {RAM_LIMIT}% — "
                f"forcing garbage collection...")
        if logger:
            logger.warning(warn)
        else:
            print(warn)
        gc.collect()


# ─────────────────────────────────────────────
# 3. VALIDATE INPUTS
# ─────────────────────────────────────────────

def validate_inputs(logger):
    """
    Checks that all required input files from embedding.py exist.
    Raises FileNotFoundError with clear message if any are missing.
    """
    logger.info("Validating input files...")
    required = {
        "image_embeddings.npy": IMAGE_EMB_PATH,
        "text_embeddings.npy" : TEXT_EMB_PATH,
        "metadata.csv"        : METADATA_PATH,
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        for m in missing:
            logger.error(f"  Missing: {m}")
        logger.error("Run src/embedding/embedding.py first.")
        raise FileNotFoundError(f"Missing required files: {missing}")
    logger.info("All required input files found ✓")


# ─────────────────────────────────────────────
# 4. LOAD IMAGE EMBEDDINGS
# ─────────────────────────────────────────────

def load_image_embeddings(logger):
    """
    Loads image_embeddings.npy as float32.
    Verifies dimension and L2 normalization.
    Re-normalizes in-place if needed.

    Returns:
        embeddings : (N, 512) float32 numpy array, L2-normalized
    """
    logger.info(f"Loading image embeddings: {IMAGE_EMB_PATH}")
    embeddings = np.load(str(IMAGE_EMB_PATH)).astype(np.float32)

    logger.info(f"  Shape  : {embeddings.shape}")
    logger.info(f"  dtype  : {embeddings.dtype}")
    logger.info(f"  Memory : {embeddings.nbytes / (1024**2):.1f} MB")

    # ── Dimension check ──
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dim mismatch: expected {EMBEDDING_DIM}, "
            f"got {embeddings.shape[1]}. Check config/indexing.yml."
        )

    # ── L2 normalization check ──
    # IndexFlatIP computes inner product. For it to equal cosine similarity,
    # all vectors must lie on the unit hypersphere (L2 norm == 1.0).
    # embedding.py normalizes them — we verify here before indexing.
    sample_norms = np.linalg.norm(embeddings[:100], axis=1)
    mean_norm    = sample_norms.mean()
    logger.info(f"  Mean L2 norm (first 100 vectors): {mean_norm:.6f} (expect ≈ 1.0)")

    if abs(mean_norm - 1.0) > 0.01:
        logger.warning(
            f"Vectors not L2-normalized (mean={mean_norm:.4f}). "
            f"Re-normalizing in-place..."
        )
        faiss.normalize_L2(embeddings)
        logger.info("Re-normalization complete ✓")
    else:
        logger.info("L2 normalization check passed ✓")

    return embeddings


# ─────────────────────────────────────────────
# 5. BUILD FAISS INDEX
# ─────────────────────────────────────────────

def build_faiss_index(embeddings, logger):
    """
    Builds a FAISS IndexFlatIP index from image embeddings.
    Vectors added in chunks (CHUNK_SIZE from config) to keep RAM flat.

    IndexFlatIP chosen because:
    - Exact search (no approximation) → highest retrieval accuracy
    - Inner product on L2-normalized vectors == cosine similarity
    - No training step required
    - Reliable for up to ~500K vectors (199K here — well within range)

    Args:
        embeddings : (N, 512) float32 numpy array, L2-normalized
        logger     : logging.Logger

    Returns:
        index : populated faiss.IndexFlatIP
    """
    n_vectors = embeddings.shape[0]
    logger.info(
        f"Building IndexFlatIP over {n_vectors} image vectors "
        f"(dim={EMBEDDING_DIM}, chunk_size={CHUNK_SIZE})"
    )

    # Create empty index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    # Add in chunks — each chunk is ~20MB, safe for M3 8GB
    for chunk_start in range(0, n_vectors, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_vectors)
        chunk     = np.ascontiguousarray(
            embeddings[chunk_start:chunk_end], dtype=np.float32
        )
        index.add(chunk)
        logger.info(
            f"  Rows {chunk_start}–{chunk_end - 1} added "
            f"| index total: {index.ntotal}"
        )
        check_ram(f"chunk {chunk_start // CHUNK_SIZE}", logger)
        del chunk
        gc.collect()

    logger.info(f"FAISS index built — {index.ntotal} vectors ✓")
    return index


# ─────────────────────────────────────────────
# 6. SAVE INDEX AND CONFIG
# ─────────────────────────────────────────────

def save_index(index, n_vectors, logger):
    """
    Saves FAISS index to faiss_image_index.bin and writes index_config.json.

    index_config.json tells the retrieval module:
    - This is an IMAGE index (search by image, not text)
    - After search: use returned row indices to fetch text_embeddings + reports
    - Row i in index == row i in text_embeddings.npy == row i in metadata.csv
    """
    logger.info(f"Saving FAISS index: {FAISS_INDEX_PATH}")
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    size_mb = FAISS_INDEX_PATH.stat().st_size / (1024 ** 2)
    logger.info(f"Index saved ({size_mb:.1f} MB) ✓")

    config = {
        "index_type"      : "IndexFlatIP",
        "index_modality"  : "image",
        "embedding_dim"   : EMBEDDING_DIM,
        "num_vectors"     : n_vectors,
        "embedding_model" : os.getenv(
            "BIOMEDCLIP_MODEL",
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        ),
        "similarity"      : "cosine (inner product on L2-normalized image vectors)",
        "index_file"      : str(FAISS_INDEX_PATH),
        "text_embeddings" : str(TEXT_EMB_PATH),
        "metadata_file"   : str(METADATA_PATH),
        "knowledge_base"  : str(KB_CSV_PATH),
        "retrieval_note"  : (
            "IMAGE index: embed query image → search → get top-K row indices "
            "→ fetch text_embeddings[indices] + reports for those rows. "
            "Row i in this index == row i in image_embeddings.npy "
            "== row i in text_embeddings.npy == row i in metadata.csv."
        )
    }

    with open(INDEX_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Index config saved: {INDEX_CONFIG_PATH} ✓")


# ─────────────────────────────────────────────
# 7. SANITY CHECKS
# ─────────────────────────────────────────────

def run_sanity_checks(index, embeddings, logger):
    """
    Check 1 — Self-retrieval:
        Query image vectors that ARE in the index.
        Top-1 result must be itself (score ≈ 1.0).

    Check 2 — Cross-file alignment:
        image_embeddings.npy, text_embeddings.npy, metadata.csv
        must all have the same number of rows.
        Row i across all three must refer to the same study.
    """
    import pandas as pd

    logger.info("Running sanity checks...")

    # ── Check 1: Self-retrieval ──
    logger.info("  [Check 1] Self-retrieval test...")
    n         = embeddings.shape[0]
    test_idxs = [0, n // 4, n // 2, (3 * n) // 4, n - 1][:NUM_SAMPLES]
    passed    = True

    for idx in test_idxs:
        query            = np.ascontiguousarray(
            embeddings[idx:idx + 1], dtype=np.float32
        )
        scores, result_ids = index.search(query, k=1)
        top1   = result_ids[0][0]
        score  = scores[0][0]
        status = "PASS ✓" if top1 == idx else "FAIL ✗"
        logger.info(f"    idx={idx} → top1={top1}  score={score:.6f}  [{status}]")
        if top1 != idx:
            passed = False

    if passed:
        logger.info("  [Check 1] Self-retrieval PASSED ✓")
    else:
        logger.warning("  [Check 1] Self-retrieval FAILED — "
                       "check embedding L2 normalization.")

    # ── Check 2: Cross-file alignment ──
    logger.info("  [Check 2] Cross-file row alignment test...")
    text_embs = np.load(str(TEXT_EMB_PATH))
    metadata  = pd.read_csv(METADATA_PATH)

    n_img  = index.ntotal
    n_txt  = text_embs.shape[0]
    n_meta = len(metadata)

    logger.info(f"    image_embeddings rows : {n_img}")
    logger.info(f"    text_embeddings  rows : {n_txt}")
    logger.info(f"    metadata.csv     rows : {n_meta}")

    if n_img == n_txt == n_meta:
        logger.info("  [Check 2] Row alignment PASSED ✓")
    else:
        logger.error(
            f"  [Check 2] ALIGNMENT MISMATCH — "
            f"image={n_img}, text={n_txt}, meta={n_meta}. "
            f"Re-run embedding.py to fix."
        )

    del text_embs, metadata
    gc.collect()


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("FAISS Image Index Builder")
    logger.info(f"Project root : {PROJECT_ROOT}")
    logger.info(f"Config       : {CONFIG_PATH}")
    logger.info(f"Env          : {dotenv_path}")
    logger.info("=" * 60)

    check_ram("startup", logger)

    validate_inputs(logger)

    embeddings = load_image_embeddings(logger)
    n_vectors  = embeddings.shape[0]
    check_ram("after loading embeddings", logger)

    index = build_faiss_index(embeddings, logger)
    check_ram("after building index", logger)

    save_index(index, n_vectors, logger)

    if RUN_SELF_CHECK:
        run_sanity_checks(index, embeddings, logger)

    logger.info("=" * 60)
    logger.info("Build complete.")
    logger.info(f"  Index  : {FAISS_INDEX_PATH}")
    logger.info(f"  Config : {INDEX_CONFIG_PATH}")
    logger.info(f"  Log    : {LOG_FILE}")
    logger.info(
        "Next step: src/retrieval/ — query image → embed → "
        "search image index → fetch text embeddings + reports by row index."
    )
    logger.info("=" * 60)
    check_ram("end", logger)


if __name__ == "__main__":
    main()
