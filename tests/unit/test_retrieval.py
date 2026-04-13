"""
tests/unit/test_retrieval.py

Unit tests for the retrieval stage output files produced by
src/retrieval/retriever.py (step1_embed.py + step2_search.py).

No model loading. No API calls. No subprocess execution.
All tests read from artefacts already written to disk by a prior retrieval run.

Prerequisite:
    python3 src/retrieval/retriever.py   (or python3 pipeline.py)
    must have been run at least once so that:
        src/retrieval/query_vec.npy
        src/retrieval/results.json
    exist on disk.

Run:
    python3 -m pytest tests/unit/test_retrieval.py -v

Expected runtime: < 5 seconds
"""

import json
import pytest
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parents[2]   # rag-rrg/
RETRIEVAL_DIR  = PROJECT_ROOT / "src" / "retrieval"

QUERY_VEC_FILE = RETRIEVAL_DIR / "query_vec.npy"
RESULTS_FILE   = RETRIEVAL_DIR / "results.json"

EXPECTED_QUERY_SHAPE = (1, 512)
EXPECTED_QUERY_DTYPE = np.float32
MIN_SCORE            = 0.0
MAX_SCORE            = 1.0
MIN_REPORT_LENGTH    = 5    # characters — anything shorter is effectively empty
MAX_TOP_K            = 5    # pipeline default — results must not exceed this


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def query_vec():
    if not QUERY_VEC_FILE.exists():
        pytest.skip(
            f"query_vec.npy not found at {QUERY_VEC_FILE}.\n"
            "Run: python3 src/retrieval/retriever.py"
        )
    return np.load(QUERY_VEC_FILE)


@pytest.fixture(scope="module")
def results():
    if not RESULTS_FILE.exists():
        pytest.skip(
            f"results.json not found at {RESULTS_FILE}.\n"
            "Run: python3 src/retrieval/retriever.py"
        )
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    return data


# ─────────────────────────────────────────────
# QUERY VECTOR TESTS
# ─────────────────────────────────────────────

class TestQueryVec:

    def test_query_vec_file_exists(self):
        assert QUERY_VEC_FILE.exists(), (
            f"query_vec.npy not found at {QUERY_VEC_FILE}.\n"
            "Run: python3 src/retrieval/retriever.py"
        )

    def test_query_vec_shape(self, query_vec):
        assert query_vec.shape == EXPECTED_QUERY_SHAPE, (
            f"query_vec.npy has shape {query_vec.shape}, "
            f"expected {EXPECTED_QUERY_SHAPE}. "
            "step1_embed.py should produce a (1, 512) array."
        )

    def test_query_vec_dtype(self, query_vec):
        assert query_vec.dtype == EXPECTED_QUERY_DTYPE, (
            f"query_vec.npy has dtype {query_vec.dtype}, "
            f"expected {EXPECTED_QUERY_DTYPE}."
        )

    def test_query_vec_no_nan(self, query_vec):
        assert not np.isnan(query_vec).any(), (
            "query_vec.npy contains NaN values. "
            "The embedding step may have failed — re-run retriever.py."
        )

    def test_query_vec_no_zero(self, query_vec):
        """A zero vector would indicate a failed encode_image call."""
        norm = float(np.linalg.norm(query_vec))
        assert norm > 0.01, (
            f"query_vec.npy has near-zero L2 norm ({norm:.6f}). "
            "encode_image may have returned a zero vector."
        )

    def test_query_vec_l2_normalized(self, query_vec):
        norm = float(np.linalg.norm(query_vec))
        assert abs(norm - 1.0) < 1e-3, (
            f"query_vec.npy has L2 norm {norm:.6f}, expected ≈ 1.0. "
            "step1_embed.py must L2-normalise the vector before saving."
        )


# ─────────────────────────────────────────────
# RESULTS FILE TESTS
# ─────────────────────────────────────────────

class TestResultsFile:

    def test_results_file_exists(self):
        assert RESULTS_FILE.exists(), (
            f"results.json not found at {RESULTS_FILE}.\n"
            "Run: python3 src/retrieval/retriever.py"
        )

    def test_results_is_list(self, results):
        assert isinstance(results, list), (
            f"results.json should be a JSON array, got {type(results).__name__}."
        )

    def test_results_not_empty(self, results):
        assert len(results) > 0, (
            "results.json is an empty list. "
            "Retrieval returned no results — check FAISS index integrity."
        )

    def test_results_count_within_top_k(self, results):
        assert len(results) <= MAX_TOP_K, (
            f"results.json has {len(results)} entries, "
            f"expected at most {MAX_TOP_K} (DEFAULT_TOP_K)."
        )


# ─────────────────────────────────────────────
# PER-RESULT SCHEMA TESTS
# ─────────────────────────────────────────────

REQUIRED_KEYS = {"rank", "score", "report", "study_id", "subject_id", "image_path"}

class TestResultSchema:

    def test_all_results_have_required_keys(self, results):
        for i, r in enumerate(results):
            missing = REQUIRED_KEYS - set(r.keys())
            assert not missing, (
                f"Result at index {i} is missing keys: {missing}. "
                f"Present keys: {set(r.keys())}"
            )

    def test_ranks_are_sequential_from_one(self, results):
        ranks = [r["rank"] for r in results]
        expected = list(range(1, len(results) + 1))
        assert ranks == expected, (
            f"Ranks are not sequential starting from 1. Got: {ranks}"
        )

    def test_scores_are_floats(self, results):
        for i, r in enumerate(results):
            assert isinstance(r["score"], (int, float)), (
                f"Result {i} score is {type(r['score']).__name__}, expected float."
            )

    def test_scores_within_valid_range(self, results):
        for i, r in enumerate(results):
            score = float(r["score"])
            assert MIN_SCORE <= score <= MAX_SCORE, (
                f"Result {i} has score {score:.4f} outside valid range "
                f"[{MIN_SCORE}, {MAX_SCORE}]. "
                "Scores should be cosine similarities between L2-normalized vectors."
            )

    def test_scores_in_descending_order(self, results):
        """Rank 1 should have the highest score."""
        scores = [float(r["score"]) for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores are not in descending order: {scores}. "
            "Results should be sorted by similarity score, highest first."
        )

    def test_study_ids_are_strings(self, results):
        for i, r in enumerate(results):
            assert isinstance(r["study_id"], str), (
                f"Result {i} study_id is {type(r['study_id']).__name__}, expected str."
            )

    def test_subject_ids_are_strings(self, results):
        for i, r in enumerate(results):
            assert isinstance(r["subject_id"], str), (
                f"Result {i} subject_id is {type(r['subject_id']).__name__}, expected str."
            )

    def test_no_duplicate_study_ids(self, results):
        """
        Deduplication by study_id is critical.
        One study can have multiple images — without deduplication,
        the same report appears multiple times in the retrieved context.
        """
        study_ids = [r["study_id"] for r in results]
        unique    = set(study_ids)
        assert len(study_ids) == len(unique), (
            f"Duplicate study_ids found in results.json: "
            f"{[sid for sid in study_ids if study_ids.count(sid) > 1]}. "
            "step2_search.py deduplication is not working correctly."
        )

    def test_reports_are_non_empty_strings(self, results):
        for i, r in enumerate(results):
            report = r.get("report", "")
            assert isinstance(report, str), (
                f"Result {i} report is {type(report).__name__}, expected str."
            )
            assert len(report.strip()) >= MIN_REPORT_LENGTH, (
                f"Result {i} report is too short ({len(report.strip())} chars). "
                f"Minimum is {MIN_REPORT_LENGTH} chars. "
                "Check knowledge_base.csv for empty report fields."
            )

    def test_image_paths_are_strings(self, results):
        for i, r in enumerate(results):
            assert isinstance(r["image_path"], str), (
                f"Result {i} image_path is {type(r['image_path']).__name__}, expected str."
            )

    def test_image_paths_have_jpg_extension(self, results):
        for i, r in enumerate(results):
            path = str(r["image_path"]).lower()
            assert path.endswith(".jpg"), (
                f"Result {i} image_path does not end with .jpg: {r['image_path']}"
            )

    def test_row_index_present_and_valid(self, results):
        """row_index must be a non-negative int — used for text_embedding lookup."""
        for i, r in enumerate(results):
            assert "row_index" in r, (
                f"Result {i} missing 'row_index' key. "
                "step2_search.py must include row_index for text_embedding alignment."
            )
            assert isinstance(r["row_index"], int), (
                f"Result {i} row_index is {type(r['row_index']).__name__}, expected int."
            )
            assert r["row_index"] >= 0, (
                f"Result {i} has negative row_index {r['row_index']}."
            )
