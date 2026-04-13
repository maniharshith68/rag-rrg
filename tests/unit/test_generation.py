"""
tests/unit/test_generation.py

Unit tests for the generation stage output files produced by
src/generation/generator.py (step1_caption.py + step2_generate.py).

No model loading. No API calls. No subprocess execution.
All tests read from artefacts already written to disk by a prior generation run.

Prerequisite:
    python3 pipeline.py   (or python3 src/generation/generator.py)
    must have been run at least once so that:
        src/generation/caption_result.json
        src/generation/generation_result.json
        data/generated_reports/*.json
    exist on disk.

Run:
    python3 -m pytest tests/unit/test_generation.py -v

Expected runtime: < 5 seconds
"""

import json
import pytest
from pathlib import Path


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).resolve().parents[2]   # rag-rrg/
GENERATION_DIR  = PROJECT_ROOT / "src" / "generation"
REPORTS_DIR     = PROJECT_ROOT / "data" / "generated_reports"
PIPELINE_DIR    = PROJECT_ROOT / "data" / "pipeline_results"

CAPTION_FILE    = GENERATION_DIR / "caption_result.json"
GENERATION_FILE = GENERATION_DIR / "generation_result.json"

# Thresholds
MIN_CAPTION_LENGTH     = 10    # characters
MIN_DESCRIPTION_LENGTH = 20    # characters
MIN_FINDINGS_LENGTH    = 20    # characters — rules out header-only responses
MIN_IMPRESSION_LENGTH  = 10    # characters
MIN_PHRASE_SCORE       = 0.0
MAX_PHRASE_SCORE       = 1.0
MIN_TOP_PHRASES        = 1
MAX_TOP_PHRASES        = 15

# Known valid backend names
KNOWN_MODELS = {
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-4-scout-17b-16e-instruct",
    # Add new backends here as they are added to step2_generate.py
}


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def caption_data():
    if not CAPTION_FILE.exists():
        pytest.skip(
            f"caption_result.json not found at {CAPTION_FILE}.\n"
            "Run: python3 pipeline.py  (or python3 src/generation/generator.py)"
        )
    with open(CAPTION_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def generation_data():
    if not GENERATION_FILE.exists():
        pytest.skip(
            f"generation_result.json not found at {GENERATION_FILE}.\n"
            "Run: python3 pipeline.py  (or python3 src/generation/generator.py)"
        )
    with open(GENERATION_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def latest_report():
    """Load the most recently written file from data/generated_reports/."""
    if not REPORTS_DIR.exists() or not any(REPORTS_DIR.glob("*.json")):
        pytest.skip(
            f"No report JSON files found in {REPORTS_DIR}.\n"
            "Run: python3 pipeline.py  (or python3 src/generation/generator.py)"
        )
    report_files = sorted(REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    with open(report_files[-1]) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def latest_pipeline_result():
    """Load the most recently written file from data/pipeline_results/."""
    if not PIPELINE_DIR.exists() or not any(PIPELINE_DIR.glob("*.json")):
        pytest.skip(
            f"No pipeline result JSON files found in {PIPELINE_DIR}.\n"
            "Run: python3 pipeline.py"
        )
    result_files = sorted(PIPELINE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    with open(result_files[-1]) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# CAPTION RESULT TESTS
# ─────────────────────────────────────────────

class TestCaptionResult:

    def test_caption_file_exists(self):
        assert CAPTION_FILE.exists(), (
            f"caption_result.json not found at {CAPTION_FILE}."
        )

    def test_caption_is_valid_json(self):
        with open(CAPTION_FILE) as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"caption_result.json is not valid JSON: {e}")

    def test_caption_required_keys(self, caption_data):
        required = {"image_path", "caption", "description", "top_phrases", "model"}
        missing  = required - set(caption_data.keys())
        assert not missing, (
            f"caption_result.json missing keys: {missing}. "
            f"Present: {set(caption_data.keys())}"
        )

    def test_caption_is_non_empty_string(self, caption_data):
        caption = caption_data.get("caption", "")
        assert isinstance(caption, str), (
            f"caption field is {type(caption).__name__}, expected str."
        )
        assert len(caption.strip()) >= MIN_CAPTION_LENGTH, (
            f"caption is too short ({len(caption.strip())} chars), "
            f"minimum is {MIN_CAPTION_LENGTH}. "
            "step1_caption.py may not have produced any high-scoring phrases."
        )

    def test_description_is_non_empty_string(self, caption_data):
        desc = caption_data.get("description", "")
        assert isinstance(desc, str), (
            f"description field is {type(desc).__name__}, expected str."
        )
        assert len(desc.strip()) >= MIN_DESCRIPTION_LENGTH, (
            f"description is too short ({len(desc.strip())} chars), "
            f"minimum is {MIN_DESCRIPTION_LENGTH}."
        )

    def test_top_phrases_is_list(self, caption_data):
        top = caption_data.get("top_phrases", None)
        assert isinstance(top, list), (
            f"top_phrases is {type(top).__name__}, expected list."
        )

    def test_top_phrases_not_empty(self, caption_data):
        top = caption_data.get("top_phrases", [])
        assert len(top) >= MIN_TOP_PHRASES, (
            f"top_phrases has {len(top)} entries, expected at least {MIN_TOP_PHRASES}."
        )

    def test_top_phrases_count_within_limit(self, caption_data):
        top = caption_data.get("top_phrases", [])
        assert len(top) <= MAX_TOP_PHRASES, (
            f"top_phrases has {len(top)} entries, expected at most {MAX_TOP_PHRASES}."
        )

    def test_top_phrases_schema(self, caption_data):
        """Each phrase entry must be a dict with 'phrase' (str) and 'score' (float)."""
        for i, item in enumerate(caption_data.get("top_phrases", [])):
            assert isinstance(item, dict), (
                f"top_phrases[{i}] is {type(item).__name__}, expected dict."
            )
            assert "phrase" in item, (
                f"top_phrases[{i}] missing 'phrase' key. Got: {item}"
            )
            assert "score" in item, (
                f"top_phrases[{i}] missing 'score' key. Got: {item}"
            )
            assert isinstance(item["phrase"], str) and len(item["phrase"]) > 0, (
                f"top_phrases[{i}]['phrase'] is empty or not a string."
            )
            score = float(item["score"])
            assert MIN_PHRASE_SCORE <= score <= MAX_PHRASE_SCORE, (
                f"top_phrases[{i}] score {score:.4f} outside valid range "
                f"[{MIN_PHRASE_SCORE}, {MAX_PHRASE_SCORE}]."
            )

    def test_top_phrases_sorted_descending(self, caption_data):
        """Phrases must be sorted highest score first."""
        scores = [float(item["score"]) for item in caption_data.get("top_phrases", [])]
        assert scores == sorted(scores, reverse=True), (
            f"top_phrases are not sorted in descending score order: {scores[:5]}"
        )

    def test_image_path_recorded(self, caption_data):
        img = caption_data.get("image_path", "")
        assert isinstance(img, str) and len(img) > 0, (
            "caption_result.json has an empty image_path field."
        )


# ─────────────────────────────────────────────
# GENERATION RESULT TESTS
# ─────────────────────────────────────────────

class TestGenerationResult:

    def test_generation_file_exists(self):
        assert GENERATION_FILE.exists(), (
            f"generation_result.json not found at {GENERATION_FILE}."
        )

    def test_generation_is_valid_json(self):
        with open(GENERATION_FILE) as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"generation_result.json is not valid JSON: {e}")

    def test_generation_required_keys(self, generation_data):
        required = {
            "image_path", "generated_at", "model_used",
            "caption", "description", "retrieved_cases",
            "context_used", "findings", "impression", "raw_llm_output",
        }
        missing = required - set(generation_data.keys())
        assert not missing, (
            f"generation_result.json missing keys: {missing}. "
            f"Present: {set(generation_data.keys())}"
        )

    def test_findings_is_non_empty_string(self, generation_data):
        findings = generation_data.get("findings", "")
        assert isinstance(findings, str), (
            f"findings is {type(findings).__name__}, expected str."
        )
        assert len(findings.strip()) >= MIN_FINDINGS_LENGTH, (
            f"findings is too short ({len(findings.strip())} chars), "
            f"minimum is {MIN_FINDINGS_LENGTH}. "
            "The LLM may not have followed the FINDINGS/IMPRESSION format. "
            "Check raw_llm_output in generation_result.json."
        )

    def test_impression_is_non_empty_string(self, generation_data):
        impression = generation_data.get("impression", "")
        assert isinstance(impression, str), (
            f"impression is {type(impression).__name__}, expected str."
        )
        assert len(impression.strip()) >= MIN_IMPRESSION_LENGTH, (
            f"impression is too short ({len(impression.strip())} chars), "
            f"minimum is {MIN_IMPRESSION_LENGTH}. "
            "The LLM may not have produced an IMPRESSION section. "
            "Check raw_llm_output in generation_result.json."
        )

    def test_model_used_is_known(self, generation_data):
        model = generation_data.get("model_used", "")
        assert isinstance(model, str) and len(model) > 0, (
            "model_used field is empty."
        )
        assert model in KNOWN_MODELS, (
            f"model_used='{model}' is not in the known models set: {KNOWN_MODELS}. "
            "If you added a new backend, add its model name to KNOWN_MODELS in this test."
        )

    def test_retrieved_cases_count(self, generation_data):
        n = generation_data.get("retrieved_cases", 0)
        assert isinstance(n, int) and n >= 1, (
            f"retrieved_cases={n}, expected at least 1."
        )

    def test_raw_llm_output_non_empty(self, generation_data):
        raw = generation_data.get("raw_llm_output", "")
        assert isinstance(raw, str) and len(raw.strip()) > 0, (
            "raw_llm_output is empty. The LLM returned no text — "
            "check logs/generation.log for API errors."
        )

    def test_context_used_non_empty(self, generation_data):
        ctx = generation_data.get("context_used", "")
        assert isinstance(ctx, str) and len(ctx.strip()) > 0, (
            "context_used is empty. Retrieved reports were not passed to the LLM."
        )

    def test_generated_at_is_iso_format(self, generation_data):
        from datetime import datetime
        ts = generation_data.get("generated_at", "")
        try:
            datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            pytest.fail(
                f"generated_at='{ts}' is not a valid ISO 8601 timestamp."
            )


# ─────────────────────────────────────────────
# SAVED REPORT FILE TESTS
# ─────────────────────────────────────────────

class TestSavedReports:

    def test_reports_directory_exists(self):
        assert REPORTS_DIR.exists(), (
            f"data/generated_reports/ directory does not exist at {REPORTS_DIR}.\n"
            "Run: python3 pipeline.py"
        )

    def test_at_least_one_report_saved(self):
        reports = list(REPORTS_DIR.glob("*.json"))
        assert len(reports) >= 1, (
            f"No report JSON files found in {REPORTS_DIR}.\n"
            "Run: python3 pipeline.py"
        )

    def test_latest_report_is_valid_json(self, latest_report):
        """Fixture already parses JSON — if it loads, it's valid."""
        assert isinstance(latest_report, dict), (
            "Latest report JSON did not deserialise to a dict."
        )

    def test_latest_report_schema(self, latest_report):
        required = {
            "metadata", "caption", "description",
            "findings", "impression",
        }
        missing = required - set(latest_report.keys())
        assert not missing, (
            f"Latest report JSON missing top-level keys: {missing}. "
            f"Present: {set(latest_report.keys())}"
        )

    def test_latest_report_metadata_schema(self, latest_report):
        meta = latest_report.get("metadata", {})
        required = {"image_path", "generated_at", "model_used", "retrieved_cases"}
        missing  = required - set(meta.keys())
        assert not missing, (
            f"Latest report metadata missing keys: {missing}. "
            f"Present: {set(meta.keys())}"
        )

    def test_latest_report_findings_non_empty(self, latest_report):
        findings = latest_report.get("findings", "")
        assert len(findings.strip()) >= MIN_FINDINGS_LENGTH, (
            f"Latest report findings is too short ({len(findings.strip())} chars)."
        )

    def test_latest_report_impression_non_empty(self, latest_report):
        impression = latest_report.get("impression", "")
        assert len(impression.strip()) >= MIN_IMPRESSION_LENGTH, (
            f"Latest report impression is too short ({len(impression.strip())} chars)."
        )


# ─────────────────────────────────────────────
# PIPELINE RESULT TESTS
# ─────────────────────────────────────────────

class TestPipelineResult:

    def test_pipeline_results_directory_exists(self):
        assert PIPELINE_DIR.exists(), (
            f"data/pipeline_results/ directory does not exist at {PIPELINE_DIR}.\n"
            "Run: python3 pipeline.py"
        )

    def test_at_least_one_pipeline_result_saved(self):
        results = list(PIPELINE_DIR.glob("*.json"))
        assert len(results) >= 1, (
            f"No pipeline result JSON files found in {PIPELINE_DIR}.\n"
            "Run: python3 pipeline.py"
        )

    def test_latest_pipeline_result_schema(self, latest_pipeline_result):
        required = {
            "image_path", "run_at", "top_k",
            "retrieved_cases", "n_retrieved",
            "caption", "description", "findings", "impression",
            "model_used", "ground_truth",
        }
        missing = required - set(latest_pipeline_result.keys())
        assert not missing, (
            f"Latest pipeline result missing keys: {missing}. "
            f"Present: {set(latest_pipeline_result.keys())}"
        )

    def test_latest_pipeline_n_retrieved_matches_cases(self, latest_pipeline_result):
        n        = latest_pipeline_result.get("n_retrieved", -1)
        cases    = latest_pipeline_result.get("retrieved_cases", [])
        assert n == len(cases), (
            f"n_retrieved={n} does not match len(retrieved_cases)={len(cases)}."
        )

    def test_latest_pipeline_top_k_valid(self, latest_pipeline_result):
        top_k = latest_pipeline_result.get("top_k", 0)
        assert isinstance(top_k, int) and top_k >= 1, (
            f"top_k={top_k} is invalid, expected a positive integer."
        )

    def test_latest_pipeline_run_at_is_iso(self, latest_pipeline_result):
        from datetime import datetime
        ts = latest_pipeline_result.get("run_at", "")
        try:
            datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            pytest.fail(
                f"Pipeline result run_at='{ts}' is not a valid ISO 8601 timestamp."
            )
