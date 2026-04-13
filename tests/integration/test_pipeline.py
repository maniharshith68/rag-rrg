"""
tests/integration/test_pipeline.py

Integration tests for the full RAG pipeline.

Unlike the unit tests which validate individual artefact files, this test
validates the complete output of a pipeline run — all stages connected —
by reading the most recent pipeline result JSON from data/pipeline_results/.

No subprocess execution. No model loading. No API calls.
All assertions run against artefacts already on disk.

If you want to trigger a fresh pipeline run as part of the test, see the
OPTIONAL section at the bottom — it is marked and skipped by default because
a full run takes 3–5 minutes and consumes API quota.

Prerequisite:
    python3 pipeline.py
    must have been run at least once.

Run:
    python3 -m pytest tests/integration/test_pipeline.py -v

Expected runtime: < 5 seconds (reads from JSON, no subprocess)
"""

import json
import pytest
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

PROJECT_ROOT     = Path(__file__).resolve().parents[2]   # rag-rrg/
PIPELINE_DIR     = PROJECT_ROOT / "data" / "pipeline_results"
REPORTS_DIR      = PROJECT_ROOT / "data" / "generated_reports"
RETRIEVAL_DIR    = PROJECT_ROOT / "src" / "retrieval"
GENERATION_DIR   = PROJECT_ROOT / "src" / "generation"
TEST_DATASET_CSV = PROJECT_ROOT / "data" / "test_dataset.csv"
IMAGE_BASE_DIR   = PROJECT_ROOT / "data"

# Thresholds
MIN_FINDINGS_LENGTH   = 20
MIN_IMPRESSION_LENGTH = 10
MIN_RETRIEVED_CASES   = 1
MAX_RETRIEVED_CASES   = 5

KNOWN_MODELS = {
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-4-scout-17b-16e-instruct",
}


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def latest_pipeline_result():
    """
    Load the most recently written pipeline result from data/pipeline_results/.
    This is the single source of truth for integration tests — it contains
    the unified output of retrieval + generation for one query image.
    """
    if not PIPELINE_DIR.exists():
        pytest.skip(
            f"data/pipeline_results/ not found at {PIPELINE_DIR}.\n"
            "Run: python3 pipeline.py"
        )
    files = sorted(PIPELINE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        pytest.skip(
            f"No pipeline result JSON files in {PIPELINE_DIR}.\n"
            "Run: python3 pipeline.py"
        )
    with open(files[-1]) as f:
        data = json.load(f)
    return data, files[-1]


@pytest.fixture(scope="module")
def result(latest_pipeline_result):
    data, _ = latest_pipeline_result
    return data


@pytest.fixture(scope="module")
def result_path(latest_pipeline_result):
    _, path = latest_pipeline_result
    return path


# ─────────────────────────────────────────────
# PIPELINE OUTPUT FILE TESTS
# ─────────────────────────────────────────────

class TestPipelineOutputFiles:
    """Verify that all expected output files were written by the last pipeline run."""

    def test_pipeline_results_directory_exists(self):
        assert PIPELINE_DIR.exists(), (
            f"data/pipeline_results/ not found. Run: python3 pipeline.py"
        )

    def test_at_least_one_pipeline_result_exists(self):
        files = list(PIPELINE_DIR.glob("*.json"))
        assert len(files) >= 1, (
            f"No pipeline result files in {PIPELINE_DIR}. Run: python3 pipeline.py"
        )

    def test_pipeline_result_filename_format(self, result_path):
        """Filename must match pattern: pipeline_{stem}_{YYYYMMDD_HHMMSS}.json"""
        name = result_path.stem   # e.g. pipeline_b67453f9_20260411_062715
        assert name.startswith("pipeline_"), (
            f"Pipeline result filename '{result_path.name}' does not start with 'pipeline_'."
        )
        parts = name.split("_")
        assert len(parts) >= 3, (
            f"Pipeline result filename '{name}' does not match expected pattern "
            f"pipeline_{{stem}}_{{YYYYMMDD}}_{{HHMMSS}}."
        )

    def test_reports_directory_exists(self):
        assert REPORTS_DIR.exists(), (
            f"data/generated_reports/ not found."
        )

    def test_at_least_one_generated_report_exists(self):
        files = list(REPORTS_DIR.glob("*.json"))
        assert len(files) >= 1, (
            f"No generated report files in {REPORTS_DIR}."
        )

    def test_retrieval_results_json_exists(self):
        results_file = RETRIEVAL_DIR / "results.json"
        assert results_file.exists(), (
            f"src/retrieval/results.json not found at {results_file}."
        )

    def test_generation_result_json_exists(self):
        gen_file = GENERATION_DIR / "generation_result.json"
        assert gen_file.exists(), (
            f"src/generation/generation_result.json not found at {gen_file}."
        )


# ─────────────────────────────────────────────
# PIPELINE RESULT SCHEMA TESTS
# ─────────────────────────────────────────────

class TestPipelineResultSchema:
    """Verify the structure and types of the unified result dict."""

    def test_result_is_dict(self, result):
        assert isinstance(result, dict), (
            f"Pipeline result is {type(result).__name__}, expected dict."
        )

    def test_top_level_keys_present(self, result):
        required = {
            "image_path", "run_at", "top_k",
            "retrieved_cases", "n_retrieved",
            "caption", "description", "top_phrases",
            "context_used", "findings", "impression",
            "model_used", "raw_llm_output", "ground_truth",
        }
        missing = required - set(result.keys())
        assert not missing, (
            f"Pipeline result missing top-level keys: {missing}"
        )

    def test_image_path_is_non_empty_string(self, result):
        img = result.get("image_path", "")
        assert isinstance(img, str) and len(img) > 0, (
            "image_path is empty or not a string."
        )

    def test_run_at_is_valid_iso_timestamp(self, result):
        ts = result.get("run_at", "")
        try:
            datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            pytest.fail(
                f"run_at='{ts}' is not a valid ISO 8601 timestamp."
            )

    def test_top_k_is_positive_int(self, result):
        top_k = result.get("top_k", 0)
        assert isinstance(top_k, int) and top_k >= 1, (
            f"top_k={top_k}, expected a positive integer."
        )


# ─────────────────────────────────────────────
# RETRIEVAL OUTPUT TESTS
# ─────────────────────────────────────────────

class TestRetrievalOutput:
    """Verify the retrieval portion of the pipeline result."""

    def test_retrieved_cases_is_list(self, result):
        cases = result.get("retrieved_cases", None)
        assert isinstance(cases, list), (
            f"retrieved_cases is {type(cases).__name__}, expected list."
        )

    def test_n_retrieved_within_range(self, result):
        n = result.get("n_retrieved", 0)
        assert MIN_RETRIEVED_CASES <= n <= MAX_RETRIEVED_CASES, (
            f"n_retrieved={n} is outside expected range "
            f"[{MIN_RETRIEVED_CASES}, {MAX_RETRIEVED_CASES}]."
        )

    def test_n_retrieved_matches_cases_length(self, result):
        n     = result.get("n_retrieved", -1)
        cases = result.get("retrieved_cases", [])
        assert n == len(cases), (
            f"n_retrieved={n} does not match len(retrieved_cases)={len(cases)}."
        )

    def test_retrieved_cases_have_required_keys(self, result):
        required = {"rank", "score", "study_id", "subject_id", "report", "image_path"}
        for i, case in enumerate(result.get("retrieved_cases", [])):
            missing = required - set(case.keys())
            assert not missing, (
                f"retrieved_cases[{i}] missing keys: {missing}"
            )

    def test_retrieved_scores_descending(self, result):
        scores = [float(c["score"]) for c in result.get("retrieved_cases", [])]
        assert scores == sorted(scores, reverse=True), (
            f"Retrieved case scores are not in descending order: {scores}"
        )

    def test_no_duplicate_study_ids_in_retrieved(self, result):
        study_ids = [c["study_id"] for c in result.get("retrieved_cases", [])]
        assert len(study_ids) == len(set(study_ids)), (
            f"Duplicate study_ids in retrieved_cases: "
            f"{[s for s in study_ids if study_ids.count(s) > 1]}"
        )

    def test_retrieved_reports_non_empty(self, result):
        for i, case in enumerate(result.get("retrieved_cases", [])):
            report = case.get("report", "").strip()
            assert len(report) >= 5, (
                f"retrieved_cases[{i}] has an empty or near-empty report field."
            )

    def test_context_used_non_empty(self, result):
        ctx = result.get("context_used", "").strip()
        assert len(ctx) > 0, (
            "context_used is empty — retrieved reports were not formatted for the LLM."
        )


# ─────────────────────────────────────────────
# GENERATION OUTPUT TESTS
# ─────────────────────────────────────────────

class TestGenerationOutput:
    """Verify the generation portion of the pipeline result."""

    def test_caption_non_empty(self, result):
        caption = result.get("caption", "").strip()
        assert len(caption) >= 10, (
            f"caption is too short ({len(caption)} chars)."
        )

    def test_description_non_empty(self, result):
        desc = result.get("description", "").strip()
        assert len(desc) >= 20, (
            f"description is too short ({len(desc)} chars)."
        )

    def test_top_phrases_list(self, result):
        top = result.get("top_phrases", [])
        assert isinstance(top, list) and len(top) >= 1, (
            "top_phrases is empty or not a list."
        )

    def test_findings_non_empty(self, result):
        findings = result.get("findings", "").strip()
        assert len(findings) >= MIN_FINDINGS_LENGTH, (
            f"findings is too short ({len(findings)} chars), "
            f"minimum is {MIN_FINDINGS_LENGTH}. "
            "The LLM may not have produced a FINDINGS section. "
            "Check raw_llm_output in the pipeline result JSON."
        )

    def test_impression_non_empty(self, result):
        impression = result.get("impression", "").strip()
        assert len(impression) >= MIN_IMPRESSION_LENGTH, (
            f"impression is too short ({len(impression)} chars), "
            f"minimum is {MIN_IMPRESSION_LENGTH}. "
            "The LLM may not have produced an IMPRESSION section."
        )

    def test_findings_and_impression_are_different(self, result):
        """
        Findings and impression should never be identical — if they are, the
        parse_findings_impression function in step2_generate.py duplicated content.
        """
        findings   = result.get("findings", "").strip()
        impression = result.get("impression", "").strip()
        if findings and impression:
            assert findings != impression, (
                "findings and impression are identical — "
                "parse_findings_impression may have duplicated text."
            )

    def test_model_used_is_known(self, result):
        model = result.get("model_used", "")
        assert model in KNOWN_MODELS, (
            f"model_used='{model}' not in known models: {KNOWN_MODELS}. "
            "Add the new backend name to KNOWN_MODELS in this test file."
        )

    def test_raw_llm_output_non_empty(self, result):
        raw = result.get("raw_llm_output", "").strip()
        assert len(raw) > 0, (
            "raw_llm_output is empty — the LLM returned no text."
        )

    def test_report_path_recorded(self, result):
        path = result.get("report_path", "")
        assert isinstance(path, str) and len(path) > 0, (
            "report_path is empty — generator did not record where it saved the report."
        )

    def test_report_path_file_exists(self, result):
        path = result.get("report_path", "")
        if path:
            assert Path(path).exists(), (
                f"report_path='{path}' is recorded but the file does not exist on disk."
            )


# ─────────────────────────────────────────────
# CROSS-STAGE CONSISTENCY TESTS
# ─────────────────────────────────────────────

class TestCrossStageConsistency:
    """
    Verify that retrieval and generation stages are consistent with each other.
    These tests catch cases where the stages ran on different images.
    """

    def test_n_retrieved_matches_generation_retrieved_cases(self, result):
        """
        The number of retrieved cases in the pipeline result should match
        what generation_result.json recorded as retrieved_cases.
        """
        gen_file = GENERATION_DIR / "generation_result.json"
        if not gen_file.exists():
            pytest.skip("generation_result.json not found — skipping cross-stage check.")

        with open(gen_file) as f:
            gen_data = json.load(f)

        pipeline_n = result.get("n_retrieved", -1)
        gen_n      = gen_data.get("retrieved_cases", -1)

        assert pipeline_n == gen_n, (
            f"Pipeline result n_retrieved={pipeline_n} does not match "
            f"generation_result.json retrieved_cases={gen_n}. "
            "Retrieval and generation may have run on different images."
        )

    def test_caption_consistent_between_result_and_generation_file(self, result):
        """Caption in pipeline result must match caption in generation_result.json."""
        gen_file = GENERATION_DIR / "generation_result.json"
        if not gen_file.exists():
            pytest.skip("generation_result.json not found.")

        with open(gen_file) as f:
            gen_data = json.load(f)

        assert result.get("caption", "").strip() == gen_data.get("caption", "").strip(), (
            "Caption in pipeline result does not match generation_result.json. "
            "The pipeline may have assembled results from mismatched runs."
        )


# ─────────────────────────────────────────────
# OPTIONAL: LIVE PIPELINE RUN
# Skip by default — remove the skip mark to run a fresh pipeline end-to-end.
# WARNING: Takes 3–5 minutes and consumes one API call.
# ─────────────────────────────────────────────

@pytest.mark.skip(
    reason=(
        "Live pipeline run is skipped by default to avoid API calls and 3-5 min runtime. "
        "Remove this skip mark to run a full end-to-end integration test."
    )
)
class TestLivePipelineRun:

    def test_full_pipeline_runs_without_error(self):
        """
        Run the full pipeline on the first valid test image and assert
        that a non-empty report is produced. Requires API access.
        """
        import sys
        import pandas as pd

        sys.path.insert(0, str(PROJECT_ROOT))
        from pipeline import Pipeline

        if not TEST_DATASET_CSV.exists():
            pytest.skip("test_dataset.csv not found.")

        df = pd.read_csv(TEST_DATASET_CSV).dropna(subset=["image_path", "report"])
        test_image = None
        for _, row in df.iterrows():
            candidate = IMAGE_BASE_DIR / row["image_path"]
            if candidate.exists():
                test_image = candidate
                break

        if test_image is None:
            pytest.skip("No valid test image found on disk.")

        pipe   = Pipeline(top_k=5)
        result = pipe.run(str(test_image), save=False)

        assert len(result.get("findings",  "").strip()) >= MIN_FINDINGS_LENGTH
        assert len(result.get("impression","").strip()) >= MIN_IMPRESSION_LENGTH
        assert result.get("n_retrieved", 0) >= MIN_RETRIEVED_CASES
        assert result.get("model_used", "") in KNOWN_MODELS
