"""
tests/unit/test_indexing.py

Unit tests for the FAISS index and embedding files produced by
src/embedding/embedding.py and src/indexing/build_index.py.

No model loading. No API calls. No subprocess execution.
All tests read from already-built artefacts on disk.

Run:
    python3 -m pytest tests/unit/test_indexing.py -v

Expected runtime: < 30 seconds
"""

import json
import pytest
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # rag-rrg/
SRC_DIR      = PROJECT_ROOT / "src"
EMBEDDING_DIR = SRC_DIR / "embedding"
INDEXING_DIR  = SRC_DIR / "indexing"

IMAGE_EMBEDDINGS_FILE = EMBEDDING_DIR / "image_embeddings.npy"
TEXT_EMBEDDINGS_FILE  = EMBEDDING_DIR / "text_embeddings.npy"
METADATA_FILE         = EMBEDDING_DIR / "metadata.csv"
FAISS_INDEX_FILE      = INDEXING_DIR  / "faiss_image_index.bin"
INDEX_CONFIG_FILE     = INDEXING_DIR  / "index_config.json"

EXPECTED_N_ROWS  = 199_214
EXPECTED_DIM     = 512
SELF_RETRIEVAL_N = 3       # number of random vectors to self-retrieval-check
SELF_RETRIEVAL_SCORE_TOL = 0.001   # score must be >= 1.0 - tol


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def image_embeddings():
    """Load image embeddings once for the whole module."""
    if not IMAGE_EMBEDDINGS_FILE.exists():
        pytest.skip(f"image_embeddings.npy not found at {IMAGE_EMBEDDINGS_FILE}")
    return np.load(IMAGE_EMBEDDINGS_FILE)


@pytest.fixture(scope="module")
def text_embeddings():
    """Load text embeddings once for the whole module."""
    if not TEXT_EMBEDDINGS_FILE.exists():
        pytest.skip(f"text_embeddings.npy not found at {TEXT_EMBEDDINGS_FILE}")
    return np.load(TEXT_EMBEDDINGS_FILE)


@pytest.fixture(scope="module")
def metadata_df():
    """Load metadata CSV once for the whole module."""
    if not METADATA_FILE.exists():
        pytest.skip(f"metadata.csv not found at {METADATA_FILE}")
    import pandas as pd
    return pd.read_csv(METADATA_FILE)


@pytest.fixture(scope="module")
def faiss_index():
    """Load FAISS index once for the whole module (mmap — low RAM cost)."""
    if not FAISS_INDEX_FILE.exists():
        pytest.skip(f"faiss_image_index.bin not found at {FAISS_INDEX_FILE}")
    try:
        import faiss
    except ImportError:
        pytest.skip("faiss-cpu not installed")
    index = faiss.read_index(
        str(FAISS_INDEX_FILE),
        faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
    )
    return index


@pytest.fixture(scope="module")
def index_config():
    """Load index_config.json once for the whole module."""
    if not INDEX_CONFIG_FILE.exists():
        pytest.skip(f"index_config.json not found at {INDEX_CONFIG_FILE}")
    with open(INDEX_CONFIG_FILE) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# EMBEDDING FILE TESTS
# ─────────────────────────────────────────────

class TestEmbeddingFiles:

    def test_image_embeddings_file_exists(self):
        assert IMAGE_EMBEDDINGS_FILE.exists(), (
            f"image_embeddings.npy not found at {IMAGE_EMBEDDINGS_FILE}.\n"
            "Run src/embedding/embedding.py to generate embeddings."
        )

    def test_text_embeddings_file_exists(self):
        assert TEXT_EMBEDDINGS_FILE.exists(), (
            f"text_embeddings.npy not found at {TEXT_EMBEDDINGS_FILE}.\n"
            "Run src/embedding/embedding.py to generate embeddings."
        )

    def test_metadata_file_exists(self):
        assert METADATA_FILE.exists(), (
            f"metadata.csv not found at {METADATA_FILE}.\n"
            "Run src/embedding/embedding.py to generate embeddings."
        )

    def test_image_embeddings_shape(self, image_embeddings):
        assert image_embeddings.ndim == 2, (
            f"Expected 2D array, got shape {image_embeddings.shape}"
        )
        assert image_embeddings.shape[0] == EXPECTED_N_ROWS, (
            f"Expected {EXPECTED_N_ROWS} rows, got {image_embeddings.shape[0]}"
        )
        assert image_embeddings.shape[1] == EXPECTED_DIM, (
            f"Expected embedding dim {EXPECTED_DIM}, got {image_embeddings.shape[1]}"
        )

    def test_text_embeddings_shape(self, text_embeddings):
        assert text_embeddings.ndim == 2, (
            f"Expected 2D array, got shape {text_embeddings.shape}"
        )
        assert text_embeddings.shape[0] == EXPECTED_N_ROWS, (
            f"Expected {EXPECTED_N_ROWS} rows, got {text_embeddings.shape[0]}"
        )
        assert text_embeddings.shape[1] == EXPECTED_DIM, (
            f"Expected embedding dim {EXPECTED_DIM}, got {text_embeddings.shape[1]}"
        )

    def test_image_embeddings_dtype(self, image_embeddings):
        assert image_embeddings.dtype == np.float32, (
            f"Expected float32, got {image_embeddings.dtype}"
        )

    def test_text_embeddings_dtype(self, text_embeddings):
        assert text_embeddings.dtype == np.float32, (
            f"Expected float32, got {text_embeddings.dtype}"
        )

    def test_image_embeddings_no_nan(self, image_embeddings):
        n_nan = np.isnan(image_embeddings).sum()
        assert n_nan == 0, (
            f"image_embeddings.npy contains {n_nan} NaN values. "
            "Re-run embedding.py — some batches may have failed silently."
        )

    def test_text_embeddings_no_nan(self, text_embeddings):
        n_nan = np.isnan(text_embeddings).sum()
        assert n_nan == 0, (
            f"text_embeddings.npy contains {n_nan} NaN values. "
            "Re-run embedding.py — some batches may have failed silently."
        )

    def test_image_embeddings_l2_normalized(self, image_embeddings):
        """
        All image embeddings should be L2-normalized (norm ≈ 1.0).
        We check a random sample of 100 vectors to keep this fast.
        """
        rng     = np.random.default_rng(seed=42)
        indices = rng.integers(0, EXPECTED_N_ROWS, size=100)
        sample  = image_embeddings[indices].astype(np.float64)
        norms   = np.linalg.norm(sample, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), (
            f"Image embeddings are not L2-normalized. "
            f"Norm range: [{norms.min():.6f}, {norms.max():.6f}] — expected all ≈ 1.0"
        )

    def test_text_embeddings_l2_normalized(self, text_embeddings):
        """Same L2-norm check for text embeddings."""
        rng     = np.random.default_rng(seed=42)
        indices = rng.integers(0, EXPECTED_N_ROWS, size=100)
        sample  = text_embeddings[indices].astype(np.float64)
        norms   = np.linalg.norm(sample, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), (
            f"Text embeddings are not L2-normalized. "
            f"Norm range: [{norms.min():.6f}, {norms.max():.6f}] — expected all ≈ 1.0"
        )


# ─────────────────────────────────────────────
# METADATA TESTS
# ─────────────────────────────────────────────

class TestMetadata:

    def test_metadata_row_count(self, metadata_df):
        assert len(metadata_df) == EXPECTED_N_ROWS, (
            f"metadata.csv has {len(metadata_df)} rows, expected {EXPECTED_N_ROWS}."
        )

    def test_metadata_required_columns(self, metadata_df):
        required = {"subject_id", "study_id", "image_path"}
        missing  = required - set(metadata_df.columns)
        assert not missing, (
            f"metadata.csv is missing columns: {missing}. "
            f"Found columns: {list(metadata_df.columns)}"
        )

    def test_metadata_no_null_subject_id(self, metadata_df):
        n_null = metadata_df["subject_id"].isna().sum()
        assert n_null == 0, (
            f"metadata.csv has {n_null} null subject_id values."
        )

    def test_metadata_no_null_study_id(self, metadata_df):
        n_null = metadata_df["study_id"].isna().sum()
        assert n_null == 0, (
            f"metadata.csv has {n_null} null study_id values."
        )

    def test_metadata_no_null_image_path(self, metadata_df):
        n_null = metadata_df["image_path"].isna().sum()
        assert n_null == 0, (
            f"metadata.csv has {n_null} null image_path values."
        )

    def test_metadata_image_paths_have_jpg_extension(self, metadata_df):
        """All image paths in metadata should end with .jpg (MIMIC-CXR format)."""
        sample = metadata_df["image_path"].dropna().head(100)
        non_jpg = [p for p in sample if not str(p).lower().endswith(".jpg")]
        assert not non_jpg, (
            f"Found {len(non_jpg)} image paths without .jpg extension. "
            f"First offender: {non_jpg[0]}"
        )


# ─────────────────────────────────────────────
# ROW ALIGNMENT TESTS
# ─────────────────────────────────────────────

class TestRowAlignment:
    """
    Critical: row i in image_embeddings == row i in text_embeddings
              == row i in metadata.csv == same study in knowledge_base.csv.
    Any misalignment breaks retrieval correctness.
    """

    def test_image_and_text_embeddings_same_row_count(
        self, image_embeddings, text_embeddings
    ):
        assert image_embeddings.shape[0] == text_embeddings.shape[0], (
            f"Row count mismatch: image_embeddings has {image_embeddings.shape[0]} rows, "
            f"text_embeddings has {text_embeddings.shape[0]} rows. "
            "Re-run embedding.py from scratch to realign."
        )

    def test_image_embeddings_and_metadata_same_row_count(
        self, image_embeddings, metadata_df
    ):
        assert image_embeddings.shape[0] == len(metadata_df), (
            f"Row count mismatch: image_embeddings has {image_embeddings.shape[0]} rows, "
            f"metadata.csv has {len(metadata_df)} rows."
        )

    def test_text_embeddings_and_metadata_same_row_count(
        self, text_embeddings, metadata_df
    ):
        assert text_embeddings.shape[0] == len(metadata_df), (
            f"Row count mismatch: text_embeddings has {text_embeddings.shape[0]} rows, "
            f"metadata.csv has {len(metadata_df)} rows."
        )


# ─────────────────────────────────────────────
# FAISS INDEX TESTS
# ─────────────────────────────────────────────

class TestFaissIndex:

    def test_faiss_index_file_exists(self):
        assert FAISS_INDEX_FILE.exists(), (
            f"faiss_image_index.bin not found at {FAISS_INDEX_FILE}.\n"
            "Run src/indexing/build_index.py to build the FAISS index."
        )

    def test_faiss_index_config_exists(self):
        assert INDEX_CONFIG_FILE.exists(), (
            f"index_config.json not found at {INDEX_CONFIG_FILE}.\n"
            "Run src/indexing/build_index.py to build the FAISS index."
        )

    def test_faiss_index_loads(self, faiss_index):
        """Index loads without error via the mmap fixture."""
        assert faiss_index is not None

    def test_faiss_index_vector_count(self, faiss_index):
        assert faiss_index.ntotal == EXPECTED_N_ROWS, (
            f"FAISS index has {faiss_index.ntotal} vectors, "
            f"expected {EXPECTED_N_ROWS}."
        )

    def test_faiss_index_dimension(self, faiss_index):
        assert faiss_index.d == EXPECTED_DIM, (
            f"FAISS index dimension is {faiss_index.d}, expected {EXPECTED_DIM}."
        )

    def test_faiss_index_config_modality(self, index_config):
        assert "index_modality" in index_config, (
            "index_config.json missing 'index_modality' key."
        )
        assert index_config["index_modality"] == "image", (
            f"Expected index_modality='image', got '{index_config['index_modality']}'. "
            "The pipeline uses image-to-image search — the index must be over "
            "image embeddings, not text embeddings."
        )

    def test_faiss_index_config_embedding_dim(self, index_config):
        assert "embedding_dim" in index_config, (
            "index_config.json missing 'embedding_dim' key."
        )
        assert index_config["embedding_dim"] == EXPECTED_DIM, (
            f"index_config.json says embedding_dim={index_config['embedding_dim']}, "
            f"expected {EXPECTED_DIM}."
        )

    def test_faiss_self_retrieval(self, faiss_index, image_embeddings):
        """
        Self-retrieval sanity check.

        Query the FAISS index with SELF_RETRIEVAL_N randomly chosen vectors
        from image_embeddings. Each vector must retrieve itself at rank 1
        with score >= 1.0 - SELF_RETRIEVAL_SCORE_TOL.

        This validates:
          1. The index was built from image_embeddings (not text_embeddings)
          2. L2 normalisation is consistent between build and query time
          3. IndexFlatIP inner product search is working correctly
        """
        rng     = np.random.default_rng(seed=0)
        indices = rng.integers(0, EXPECTED_N_ROWS, size=SELF_RETRIEVAL_N)

        for idx in indices:
            query  = image_embeddings[idx : idx + 1].astype(np.float32)
            scores, retrieved = faiss_index.search(query, 1)
            top1_idx   = int(retrieved[0][0])
            top1_score = float(scores[0][0])

            assert top1_idx == idx, (
                f"Self-retrieval failed for row {idx}: "
                f"top-1 returned row {top1_idx} instead. "
                "The index may have been built from different embeddings than "
                "what is currently in image_embeddings.npy."
            )
            assert top1_score >= 1.0 - SELF_RETRIEVAL_SCORE_TOL, (
                f"Self-retrieval score for row {idx} is {top1_score:.6f}, "
                f"expected >= {1.0 - SELF_RETRIEVAL_SCORE_TOL:.6f}. "
                "Embeddings may not be properly L2-normalized."
            )
