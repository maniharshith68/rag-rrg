"""
src/generation/generator.py

Entry point and orchestrator for the generation stage of the RAG pipeline.

Standalone test (uses first valid image from test_dataset.csv):
    python3 src/generation/generator.py

Importable by pipeline.py for end-to-end use:
    from src.generation.generator import Generator
    gen    = Generator()
    report = gen.generate("/path/to/image.jpg")
    gen.print_report(report)

What it does:
    1. Validates that results.json exists (retriever.py must have run first)
    2. Runs step1_caption.py as a subprocess
           → BioMedCLIP embeds image + clinical phrases
           → saves caption_result.json
           → exits, OS reclaims ~450MB
    3. Runs step2_generate.py as a subprocess
           → reads caption_result.json + results.json
           → calls Gemini Flash API with image + prompt
           → saves generation_result.json
           → exits
    4. Reads generation_result.json
    5. Assembles final_report dict
    6. Saves to data/generated_reports/report_{stem}_{timestamp}.json
    7. Returns the final_report dict

No model or API client is ever held in THIS process's memory.
Each subprocess gets a clean memory space — OS reclaims all memory on subprocess exit.
"""

import sys
import json
import logging
import subprocess
import psutil
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# PATHS  (all derived from this file's location — no .env needed)
# ─────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent   # rag-rrg/src/generation/
SRC_DIR      = SCRIPT_DIR.parent                 # rag-rrg/src/
PROJECT_ROOT = SRC_DIR.parent                    # rag-rrg/

STEP1_SCRIPT     = SCRIPT_DIR / "step1_caption.py"
STEP2_SCRIPT     = SCRIPT_DIR / "step2_generate.py"
CAPTION_FILE     = SCRIPT_DIR / "caption_result.json"
GENERATION_FILE  = SCRIPT_DIR / "generation_result.json"
RETRIEVAL_FILE   = SRC_DIR / "retrieval" / "results.json"

REPORTS_DIR      = PROJECT_ROOT / "data" / "generated_reports"
TEST_DATASET_CSV = PROJECT_ROOT / "data" / "test_dataset.csv"
IMAGE_BASE_DIR   = PROJECT_ROOT / "data"


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [generator] %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "generation.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def check_ram(label: str = "") -> None:
    """Log current RAM usage — mirrors the same helper in retriever.py."""
    ram      = psutil.virtual_memory()
    used_gb  = ram.used      / (1024 ** 3)
    avail_gb = ram.available / (1024 ** 3)
    total_gb = ram.total     / (1024 ** 3)
    log.info(
        f"[RAM {label}] "
        f"used={used_gb:.2f}GB  free={avail_gb:.2f}GB  "
        f"total={total_gb:.2f}GB  ({ram.percent:.1f}%)"
    )


def _run_subprocess(script_path: Path, args: list[str], label: str) -> None:
    """
    Run a Python script as a subprocess using the same interpreter as this process.
    Streams stdout/stderr live to the terminal.
    Raises RuntimeError if the subprocess exits with a non-zero code.
    """
    cmd = [sys.executable, str(script_path)] + args
    log.info(f"[{label}] Launching: {' '.join(cmd)}")
    print(f"\n{'─' * 60}")

    result = subprocess.run(
        cmd,
        stdout=None,   # inherit → streams live to terminal
        stderr=None,   # inherit → streams live to terminal
    )

    print(f"{'─' * 60}")

    if result.returncode != 0:
        raise RuntimeError(
            f"[{label}] Subprocess failed (exit code {result.returncode}).\n"
            f"  Script : {script_path}\n"
            f"  Args   : {args}\n"
            f"See output above for details."
        )
    log.info(f"[{label}] Subprocess finished successfully ✓")


# ─────────────────────────────────────────────
# GENERATOR CLASS
# ─────────────────────────────────────────────

class Generator:
    """
    Two-phase subprocess-based report generator for the RAG pipeline.

    Phase 1 — step1_caption.py subprocess:
        Loads BioMedCLIP fp16, computes zero-shot similarity against clinical
        phrase vocabulary, saves caption_result.json, exits.
        OS reclaims ~450MB of model memory.

    Phase 2 — step2_generate.py subprocess:
        Reads caption_result.json + results.json, encodes query image as base64,
        builds radiologist prompt, calls Gemini Flash API, parses FINDINGS and
        IMPRESSION, saves generation_result.json, exits.

    Orchestrator reads generation_result.json, assembles the final report dict,
    saves it to data/generated_reports/, and returns it.

    No model or API client is ever resident in this process.

    Prerequisites:
        - results.json must exist at src/retrieval/results.json
          (produced by running src/retrieval/retriever.py first)
        - GEMINI_API_KEY must be set in .env
    """

    def __init__(self):
        for script in [STEP1_SCRIPT, STEP2_SCRIPT]:
            if not script.exists():
                raise FileNotFoundError(
                    f"Subprocess script not found: {script}\n"
                    f"Ensure step1_caption.py and step2_generate.py "
                    f"are in {SCRIPT_DIR}"
                )
        log.info("Generator initialised.")
        log.info(f"  step1_caption.py  → {STEP1_SCRIPT}")
        log.info(f"  step2_generate.py → {STEP2_SCRIPT}")
        log.info(f"  Reports saved to  → {REPORTS_DIR}")

    def generate(
        self,
        image_path : str | Path,
        save       : bool = True,
    ) -> dict:
        """
        Run the full two-phase generation pipeline for one query image.

        Args:
            image_path : Absolute or relative path to the query chest X-ray.
            save       : If True, write the final report to
                         data/generated_reports/report_{stem}_{timestamp}.json

        Returns:
            dict with keys:
                metadata        dict    image_path, generated_at, model_used,
                                        retrieved_cases, report_path (if saved)
                caption         str     top-3 BioMedCLIP phrases as a sentence
                description     str     structured paragraph from phrase scores
                top_phrases     list    top-10 [{phrase, score}] from BioMedCLIP
                context_used    str     formatted retrieved reports sent to LLM
                findings        str     FINDINGS section of the generated report
                impression      str     IMPRESSION section of the generated report
                raw_llm_output  str     unmodified text returned by Gemini

        Raises:
            FileNotFoundError : query image not found, or subprocess scripts missing
            RuntimeError      : results.json missing, subprocess failure,
                                or generation_result.json not created
        """
        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Query image not found: {image_path}")

        if not RETRIEVAL_FILE.exists():
            raise RuntimeError(
                f"results.json not found at {RETRIEVAL_FILE}.\n"
                f"Run src/retrieval/retriever.py first to generate retrieval results,\n"
                f"then run generator.py."
            )

        log.info("=" * 60)
        log.info("Generation started")
        log.info(f"  Image : {image_path.name}")
        log.info("=" * 60)
        check_ram("before Phase 1")

        # ── Phase 1: Caption + description via BioMedCLIP ──
        _run_subprocess(
            STEP1_SCRIPT,
            args  = [str(image_path)],
            label = "Phase 1 | caption",
        )

        if not CAPTION_FILE.exists():
            raise RuntimeError(
                "Phase 1 exited cleanly but caption_result.json was not created.\n"
                f"Expected at: {CAPTION_FILE}"
            )
        check_ram("after Phase 1 — BioMedCLIP memory reclaimed by OS")

        # ── Phase 2: Gemini report generation ──
        _run_subprocess(
            STEP2_SCRIPT,
            args  = [str(image_path)],
            label = "Phase 2 | generate",
        )

        if not GENERATION_FILE.exists():
            raise RuntimeError(
                "Phase 2 exited cleanly but generation_result.json was not created.\n"
                f"Expected at: {GENERATION_FILE}"
            )
        check_ram("after Phase 2")

        # ── Read generation result ──
        gen_data = self._load_generation_result()

        # ── Assemble final report ──
        final_report = {
            "metadata": {
                "image_path"      : str(image_path),
                "generated_at"    : gen_data.get("generated_at", datetime.now().isoformat()),
                "model_used"      : gen_data.get("model_used", "unknown"),
                "retrieved_cases" : gen_data.get("retrieved_cases", 0),
            },
            "caption"        : gen_data.get("caption",        ""),
            "description"    : gen_data.get("description",    ""),
            "top_phrases"    : gen_data.get("top_phrases",    []),
            "context_used"   : gen_data.get("context_used",   ""),
            "findings"       : gen_data.get("findings",       ""),
            "impression"     : gen_data.get("impression",     ""),
            "raw_llm_output" : gen_data.get("raw_llm_output", ""),
        }

        # ── Save to data/generated_reports/ ──
        if save:
            report_path = self._save_report(final_report, image_path)
            final_report["metadata"]["report_path"] = str(report_path)

        log.info("Generation complete ✓")
        return final_report

    # ─────────────────────────────────────────
    # PUBLIC HELPERS
    # ─────────────────────────────────────────

    def print_report(self, report: dict) -> None:
        """Pretty-print a final report dict to stdout."""
        meta = report.get("metadata", {})
        print(f"\n{'=' * 60}")
        print("GENERATED RADIOLOGY REPORT")
        print(f"{'=' * 60}")
        print(f"Image       : {meta.get('image_path', 'unknown')}")
        print(f"Model       : {meta.get('model_used', 'unknown')}")
        print(f"Generated   : {meta.get('generated_at', 'unknown')}")
        print(f"Retrieved   : {meta.get('retrieved_cases', 0)} similar cases")

        saved = meta.get("report_path")
        if saved:
            print(f"Saved to    : {saved}")

        print(f"\nCaption:\n  {report.get('caption', '')}")
        print(f"\nDescription:\n  {report.get('description', '')}")

        top = report.get("top_phrases", [])[:5]
        if top:
            print("\nTop BioMedCLIP phrases:")
            for item in top:
                print(f"  {item['score']:.4f}  {item['phrase']}")

        print(f"\n{'─' * 60}")
        print("FINDINGS:")
        findings = report.get("findings", "").strip()
        print(findings if findings else "[empty — check logs/generation.log]")

        print(f"\nIMPRESSION:")
        impression = report.get("impression", "").strip()
        print(impression if impression else "[empty — check logs/generation.log]")

        print(f"{'=' * 60}\n")

    # ─────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────

    def _load_generation_result(self) -> dict:
        """Read and validate generation_result.json written by step2_generate.py."""
        try:
            with open(GENERATION_FILE) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"generation_result.json is not valid JSON: {e}")
        except OSError as e:
            raise RuntimeError(f"Could not read generation_result.json: {e}")

        if not isinstance(data, dict):
            raise RuntimeError(
                "generation_result.json has unexpected structure "
                f"(expected dict, got {type(data).__name__})."
            )

        # Warn — but don't crash — if the key sections are empty
        if not data.get("findings"):
            log.warning(
                "generation_result.json has an empty 'findings' field. "
                "Check logs/generation.log for Gemini API or parsing errors."
            )
        if not data.get("impression"):
            log.warning(
                "generation_result.json has an empty 'impression' field. "
                "The LLM response may not have followed the expected format."
            )

        return data

    def _save_report(self, report: dict, image_path: Path) -> Path:
        """
        Write final report JSON to data/generated_reports/.

        Filename format: report_{image_stem}_{YYYYMMDD_HHMMSS}.json
        Returns the Path of the saved file.
        """
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem     = image_path.stem
        filename = f"report_{stem}_{ts}.json"
        out_path = REPORTS_DIR / filename

        try:
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            log.info(f"Report saved → {out_path}")
        except OSError as e:
            log.error(
                f"Failed to save report to {out_path}: {e}\n"
                f"The in-memory report dict is still returned to the caller."
            )

        return out_path


# ─────────────────────────────────────────────
# STANDALONE TEST  (entry point)
# ─────────────────────────────────────────────

def main():
    import pandas as pd

    print("=" * 60)
    print("generator.py — Generation pipeline test")
    print("=" * 60)
    check_ram("baseline")

    # ── Validate prerequisite: results.json from retriever ──
    if not RETRIEVAL_FILE.exists():
        log.error(
            f"results.json not found at {RETRIEVAL_FILE}.\n"
            "You must run the retrieval stage first:\n"
            "    python3 src/retrieval/retriever.py\n"
            "That produces results.json for a specific query image.\n"
            "generator.py then generates a report for that same image."
        )
        sys.exit(1)

    # ── Find a valid test image from test_dataset.csv ──
    # results.json stores retrieved image paths (knowledge base),
    # not the original query image path. The most reliable way to
    # get a valid query image is to scan test_dataset.csv — the same
    # approach used in retriever.py's main().
    if not TEST_DATASET_CSV.exists():
        log.error(
            f"test_dataset.csv not found at {TEST_DATASET_CSV}.\n"
            "Ensure the dataset split files are in place."
        )
        sys.exit(1)

    test_df = pd.read_csv(TEST_DATASET_CSV).dropna(subset=["image_path", "report"])
    if test_df.empty:
        log.error("test_dataset.csv has no valid rows (need image_path + report).")
        sys.exit(1)

    test_image_path     = None
    ground_truth_report = None

    for _, row in test_df.iterrows():
        candidate = IMAGE_BASE_DIR / row["image_path"]
        if candidate.exists():
            test_image_path     = candidate
            ground_truth_report = str(row["report"]).strip()
            break

    if test_image_path is None:
        log.error(
            "Could not find any test image on disk.\n"
            f"IMAGE_BASE_DIR = {IMAGE_BASE_DIR}\n"
            "Check that MIMIC-CXR image files are present under data/ and that\n"
            "image_path values in test_dataset.csv match the actual file layout."
        )
        sys.exit(1)

    # ── Load retrieval data for display ──
    with open(RETRIEVAL_FILE) as f:
        retrieval_data = json.load(f)

    # ── Print test summary ──
    print(f"\n{'─' * 60}")
    print(f"Query image         : {test_image_path}")
    print(f"Retrieval results   : {len(retrieval_data)} cases in results.json")
    if ground_truth_report:
        print(f"Ground truth (first 300 chars):")
        print(f"  {ground_truth_report[:300]}…")
    print(f"{'─' * 60}")

    # ── Run generation ──
    try:
        gen    = Generator()
        report = gen.generate(str(test_image_path), save=True)
    except (FileNotFoundError, RuntimeError) as e:
        log.error(str(e))
        sys.exit(1)

    # ── Print the generated report ──
    gen.print_report(report)

    # ── Side-by-side ground truth for qualitative inspection ──
    if ground_truth_report:
        print(f"{'─' * 60}")
        print("GROUND TRUTH REPORT (for qualitative comparison):")
        print(f"{'─' * 60}")
        print(ground_truth_report[:1500])
        if len(ground_truth_report) > 1500:
            print("… [truncated to 1500 chars]")
        print()

    check_ram("end")
    print(f"{'=' * 60}")
    print("Generation test complete ✓")
    print(f"Report saved to : {report['metadata'].get('report_path', 'not saved')}")
    print(f"Next step       : build tests/ and evaluate.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
