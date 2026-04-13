"""
pipeline.py

End-to-end RAG pipeline for radiology report generation.
Place this file at the project root: rag-rrg/pipeline.py

Standalone test (uses first valid image from test_dataset.csv):
    python3 pipeline.py

Importable by evaluate.py for batch evaluation:
    from pipeline import Pipeline
    pipe   = Pipeline()
    result = pipe.run("/path/to/image.jpg", ground_truth="...")
    pipe.print_result(result)

What it does:
    1. Instantiates Retriever and Generator
    2. Calls retriever.retrieve(image_path, top_k)
           → runs step1_embed + step2_search as subprocesses
           → writes src/retrieval/results.json
           → returns list of retrieved report dicts
    3. Calls generator.generate(image_path)
           → runs step1_caption + step2_generate as subprocesses
           → reads results.json written in step 2
           → writes data/generated_reports/report_*.json
           → returns final report dict
    4. Assembles a single unified result dict with all fields
    5. Optionally saves the unified result to data/pipeline_results/

This file contains no model loading and no API calls.
All heavy work happens inside subprocess scripts managed by
Retriever and Generator respectively.
"""

import sys
import json
import logging
import psutil
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent   # rag-rrg/
SRC_DIR      = PROJECT_ROOT / "src"

TEST_DATASET_CSV   = PROJECT_ROOT / "data" / "test_dataset.csv"
IMAGE_BASE_DIR     = PROJECT_ROOT / "data"
PIPELINE_RESULTS_DIR = PROJECT_ROOT / "data" / "pipeline_results"

# Add src/ to path so we can import Retriever and Generator
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [pipeline] %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "pipeline.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PIPELINE CONFIG
# ─────────────────────────────────────────────

DEFAULT_TOP_K = 5   # number of similar cases to retrieve


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def check_ram(label: str = "") -> None:
    """Log current RAM — consistent with retriever.py and generator.py helpers."""
    ram      = psutil.virtual_memory()
    used_gb  = ram.used      / (1024 ** 3)
    avail_gb = ram.available / (1024 ** 3)
    total_gb = ram.total     / (1024 ** 3)
    log.info(
        f"[RAM {label}] "
        f"used={used_gb:.2f}GB  free={avail_gb:.2f}GB  "
        f"total={total_gb:.2f}GB  ({ram.percent:.1f}%)"
    )


# ─────────────────────────────────────────────
# PIPELINE CLASS
# ─────────────────────────────────────────────

class Pipeline:
    """
    End-to-end RAG pipeline: retrieval → generation → unified result.

    Retriever and Generator are both imported lazily inside __init__
    so that import errors surface clearly rather than at call time.

    Usage:
        pipe   = Pipeline()
        result = pipe.run("/path/to/image.jpg")
        pipe.print_result(result)

    For evaluation with ground truth comparison:
        result = pipe.run("/path/to/image.jpg", ground_truth="Findings: ...")
        # result["ground_truth"] and result["findings"] are both populated
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        """
        Initialise the pipeline.

        Args:
            top_k : Number of similar cases to retrieve per query. Default 5.
        """
        self.top_k = top_k

        # Import here so path manipulation above has taken effect
        try:
            from retrieval.retriever import Retriever
            from generation.generator import Generator
        except ImportError as e:
            raise ImportError(
                f"Could not import Retriever or Generator: {e}\n"
                f"Ensure pipeline.py is at the project root (rag-rrg/) and that\n"
                f"src/retrieval/retriever.py and src/generation/generator.py exist."
            ) from e

        self.retriever = Retriever()
        self.generator = Generator()

        log.info(f"Pipeline initialised  top_k={self.top_k}")

    def run(
        self,
        image_path    : str | Path,
        ground_truth  : str  = "",
        save          : bool = True,
    ) -> dict:
        """
        Run the full RAG pipeline for one query chest X-ray.

        Args:
            image_path   : Path to the query chest X-ray image (JPG / PNG).
            ground_truth : Optional ground truth report text. When provided it
                           is stored in the result dict for evaluation comparison.
                           Does not affect retrieval or generation behaviour.
            save         : If True, write the unified result dict to
                           data/pipeline_results/pipeline_{stem}_{timestamp}.json

        Returns:
            Unified result dict with keys:
                image_path        str    absolute path to query image
                run_at            str    ISO timestamp of this run
                top_k             int    number of cases retrieved

                — retrieval —
                retrieved_cases   list   full list of result dicts from Retriever
                                         each has: rank, score, study_id,
                                         subject_id, image_path, report
                n_retrieved       int    number of unique cases actually returned

                — generation —
                caption           str    top-3 BioMedCLIP phrases as a sentence
                description       str    structured paragraph with confidence scores
                top_phrases       list   top-10 [{phrase, score}] from BioMedCLIP
                context_used      str    formatted retrieved reports sent to LLM
                findings          str    FINDINGS section of generated report
                impression        str    IMPRESSION section of generated report
                model_used        str    name of LLM backend that produced report
                raw_llm_output    str    unmodified LLM response text
                report_path       str    path to saved generation JSON (if saved)

                — evaluation —
                ground_truth      str    ground truth report (empty if not provided)

                — metadata —
                pipeline_result_path  str  path to this unified result JSON (if saved)

        Raises:
            FileNotFoundError : image_path does not exist on disk
            RuntimeError      : any subprocess failure in retrieval or generation
        """
        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Query image not found: {image_path}")

        log.info("=" * 60)
        log.info("Pipeline run started")
        log.info(f"  Image      : {image_path.name}")
        log.info(f"  top_k      : {self.top_k}")
        log.info(f"  Ground truth provided : {'yes' if ground_truth else 'no'}")
        log.info("=" * 60)
        check_ram("pipeline start")

        # ── Stage 1: Retrieval ──────────────────────────────────────
        log.info("─" * 40)
        log.info("STAGE 1 — Retrieval")
        log.info("─" * 40)

        try:
            retrieved = self.retriever.retrieve(
                image_path = str(image_path),
                top_k      = self.top_k,
            )
        except (FileNotFoundError, RuntimeError) as e:
            raise RuntimeError(f"Retrieval stage failed: {e}") from e

        log.info(f"Retrieval complete — {len(retrieved)} cases returned")
        check_ram("after retrieval")

        # ── Stage 2: Generation ─────────────────────────────────────
        log.info("─" * 40)
        log.info("STAGE 2 — Generation")
        log.info("─" * 40)

        try:
            report = self.generator.generate(
                image_path = str(image_path),
                save       = True,   # always save the individual generation result
            )
        except (FileNotFoundError, RuntimeError) as e:
            raise RuntimeError(f"Generation stage failed: {e}") from e

        log.info("Generation complete")
        check_ram("after generation")

        # ── Assemble unified result ─────────────────────────────────
        result = {
            # Run metadata
            "image_path"  : str(image_path),
            "run_at"      : datetime.now().isoformat(),
            "top_k"       : self.top_k,

            # Retrieval outputs
            "retrieved_cases" : retrieved,
            "n_retrieved"     : len(retrieved),

            # Generation outputs — pulled from the generator's report dict
            "caption"         : report.get("caption",        ""),
            "description"     : report.get("description",    ""),
            "top_phrases"     : report.get("top_phrases",    []),
            "context_used"    : report.get("context_used",   ""),
            "findings"        : report.get("findings",       ""),
            "impression"      : report.get("impression",     ""),
            "model_used"      : report.get("metadata", {}).get("model_used", "unknown"),
            "raw_llm_output"  : report.get("raw_llm_output", ""),
            "report_path"     : report.get("metadata", {}).get("report_path", ""),

            # Evaluation
            "ground_truth"    : ground_truth.strip() if ground_truth else "",
        }

        # ── Save unified pipeline result ────────────────────────────
        if save:
            pipeline_result_path = self._save_result(result, image_path)
            result["pipeline_result_path"] = str(pipeline_result_path)

        log.info("Pipeline run complete ✓")
        return result

    # ─────────────────────────────────────────
    # PUBLIC HELPERS
    # ─────────────────────────────────────────

    def print_result(self, result: dict, show_retrieved: bool = True) -> None:
        """
        Pretty-print a pipeline result dict to stdout.

        Args:
            result         : Dict returned by run().
            show_retrieved : If True, print the retrieved cases summary.
                             Set to False during batch evaluation to reduce noise.
        """
        print(f"\n{'=' * 60}")
        print("RAG PIPELINE RESULT")
        print(f"{'=' * 60}")
        print(f"Image      : {result.get('image_path', '')}")
        print(f"Run at     : {result.get('run_at', '')}")
        print(f"Model      : {result.get('model_used', '')}")
        print(f"Retrieved  : {result.get('n_retrieved', 0)} cases (top_k={result.get('top_k', 0)})")

        saved = result.get("pipeline_result_path", "")
        if saved:
            print(f"Saved to   : {saved}")

        # Retrieved cases summary
        if show_retrieved and result.get("retrieved_cases"):
            print(f"\n{'─' * 60}")
            print("RETRIEVED SIMILAR CASES:")
            for r in result["retrieved_cases"]:
                preview = r.get("report", "")[:120].replace("\n", " ")
                print(
                    f"  [{r['rank']}] score={r['score']:.4f}  "
                    f"study={r['study_id']}  "
                    f"patient={r['subject_id']}"
                )
                print(f"      {preview}…")

        # Caption and description
        print(f"\n{'─' * 60}")
        print(f"Caption:\n  {result.get('caption', '')}")
        print(f"\nDescription:\n  {result.get('description', '')}")

        # Top BioMedCLIP phrases
        top = result.get("top_phrases", [])[:5]
        if top:
            print("\nTop BioMedCLIP phrases:")
            for item in top:
                print(f"  {item['score']:.4f}  {item['phrase']}")

        # Generated report
        print(f"\n{'─' * 60}")
        print("FINDINGS:")
        findings = result.get("findings", "").strip()
        print(findings if findings else "[empty — check logs/generation.log]")

        print(f"\nIMPRESSION:")
        impression = result.get("impression", "").strip()
        print(impression if impression else "[empty — check logs/generation.log]")

        # Ground truth comparison
        gt = result.get("ground_truth", "").strip()
        if gt:
            print(f"\n{'─' * 60}")
            print("GROUND TRUTH REPORT (for comparison):")
            print(gt[:1500])
            if len(gt) > 1500:
                print("… [truncated to 1500 chars]")

        print(f"{'=' * 60}\n")

    # ─────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────

    def _save_result(self, result: dict, image_path: Path) -> Path:
        """
        Write the unified pipeline result to data/pipeline_results/.

        The retrieved_cases list contains text_embedding fields (512 floats each)
        from step2_search.py. These are stripped before saving to keep file sizes
        manageable — they are only needed for optional reranking, not for evaluation.

        Filename: pipeline_{image_stem}_{YYYYMMDD_HHMMSS}.json
        """
        # Strip large embedding vectors before saving to disk
        cleaned_cases = []
        for case in result.get("retrieved_cases", []):
            cleaned = {k: v for k, v in case.items() if k != "text_embedding"}
            cleaned_cases.append(cleaned)

        saveable = {**result, "retrieved_cases": cleaned_cases}

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem     = image_path.stem
        filename = f"pipeline_{stem}_{ts}.json"
        out_path = PIPELINE_RESULTS_DIR / filename

        try:
            with open(out_path, "w") as f:
                json.dump(saveable, f, indent=2)
            log.info(f"Pipeline result saved → {out_path}")
        except OSError as e:
            log.error(
                f"Failed to save pipeline result to {out_path}: {e}\n"
                "In-memory result dict is still returned to caller."
            )

        return out_path


# ─────────────────────────────────────────────
# STANDALONE TEST  (entry point)
# ─────────────────────────────────────────────

def main():
    import pandas as pd

    print("=" * 60)
    print("pipeline.py — End-to-end RAG pipeline test")
    print("=" * 60)
    check_ram("baseline")

    # ── Find first valid test image from test_dataset.csv ──────────
    if not TEST_DATASET_CSV.exists():
        log.error(
            f"test_dataset.csv not found at {TEST_DATASET_CSV}.\n"
            "Ensure the dataset split files are in place under data/."
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
            "Check that MIMIC-CXR files are present under data/ and that\n"
            "image_path values in test_dataset.csv match the actual layout."
        )
        sys.exit(1)

    # ── Print test summary ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Query image    : {test_image_path}")
    print(f"Ground truth   : {ground_truth_report[:200]}…")
    print(f"{'─' * 60}\n")

    # ── Run the pipeline ───────────────────────────────────────────
    try:
        pipe   = Pipeline(top_k=DEFAULT_TOP_K)
        result = pipe.run(
            image_path   = str(test_image_path),
            ground_truth = ground_truth_report,
            save         = True,
        )
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        log.error(str(e))
        sys.exit(1)

    # ── Print full result ──────────────────────────────────────────
    pipe.print_result(result, show_retrieved=True)

    check_ram("end")
    print(f"{'=' * 60}")
    print("Pipeline test complete ✓")
    print(f"Unified result : {result.get('pipeline_result_path', 'not saved')}")
    print(f"Generation JSON: {result.get('report_path', 'not saved')}")
    print(f"Next step      : python3 -m pytest tests/unit/ -v")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
