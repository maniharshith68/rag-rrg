"""
src/retrieval/retriever.py

Entry point for the retrieval stage of the RAG pipeline.

    python3 src/retrieval/retriever.py

What it does:
    1. Picks the first valid image from data/test_dataset.csv as the query
    2. Runs step1_embed.py as a subprocess  → embeds query image, saves query_vec.npy, exits
    3. Runs step2_search.py as a subprocess → searches FAISS IMAGE index, deduplicates
                                               by study_id, saves results.json, exits
    4. Reads results.json and prints a structured report + formatted context

No model or FAISS index is ever held in THIS process's memory.
Each subprocess gets a clean memory space — OS reclaims all memory on subprocess exit.

Retriever class can also be imported by pipeline.py for end-to-end use:
    from src.retrieval.retriever import Retriever
    retriever = Retriever()
    results   = retriever.retrieve("/path/to/image.jpg", top_k=5)
    context   = retriever.format_context(results)
"""

import sys
import json
import logging
import subprocess
import psutil
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS  (all derived from this file's location — no .env needed)
# ─────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent   # rag-rrg/src/retrieval/
SRC_DIR      = SCRIPT_DIR.parent                 # rag-rrg/src/
PROJECT_ROOT = SRC_DIR.parent                    # rag-rrg/

STEP1_SCRIPT   = SCRIPT_DIR / "step1_embed.py"
STEP2_SCRIPT   = SCRIPT_DIR / "step2_search.py"
QUERY_VEC_FILE = SCRIPT_DIR / "query_vec.npy"
RESULTS_FILE   = SCRIPT_DIR / "results.json"

TEST_DATASET_CSV = PROJECT_ROOT / "data" / "test_dataset.csv"
IMAGE_BASE_DIR   = PROJECT_ROOT / "data"

# ─────────────────────────────────────────────
# RETRIEVAL CONFIG  (fixed constants)
# ─────────────────────────────────────────────

DEFAULT_TOP_K       = 5
MAX_REPORT_CHARS    = 2000   # truncate reports in format_context to avoid prompt overflow
FORMAT_MAX_REPORTS  = 3      # number of reports to include in formatted context by default

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [retriever] %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "retrieval.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def check_ram(label: str = "") -> None:
    """Print current RAM usage for monitoring memory across subprocess phases."""
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
    Streams stdout/stderr live to the terminal (capture_output=False).
    Raises RuntimeError if subprocess exits with non-zero code.
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
# RETRIEVER CLASS
# ─────────────────────────────────────────────

class Retriever:
    """
    Two-phase subprocess-based image retriever for the RAG pipeline.

    Phase 1 — step1_embed.py subprocess:
        Loads BioMedCLIP fp16, embeds query image, saves query_vec.npy, exits.
        OS reclaims ~450MB of model memory.

    Phase 2 — step2_search.py subprocess:
        Loads query_vec.npy, mmap-loads FAISS IMAGE index, searches top_k*3
        candidates, deduplicates by study_id, fetches reports, saves results.json,
        exits. OS reclaims memory.

    Orchestrator reads results.json and returns Python list of dicts.
    No model or index is ever resident in this process.
    """

    def __init__(self):
        for script in [STEP1_SCRIPT, STEP2_SCRIPT]:
            if not script.exists():
                raise FileNotFoundError(
                    f"Subprocess script not found: {script}\n"
                    f"Ensure step1_embed.py and step2_search.py are in {SCRIPT_DIR}"
                )
        log.info("Retriever initialised.")
        log.info(f"  step1_embed.py  → {STEP1_SCRIPT}")
        log.info(f"  step2_search.py → {STEP2_SCRIPT}")

    def retrieve(self, image_path: str | Path, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """
        Retrieve the top-K most visually similar radiology reports for a query X-ray.

        Args:
            image_path : Path to the query chest X-ray (JPG / PNG).
            top_k      : Number of unique studies to return.

        Returns:
            List of result dicts sorted by rank (rank 1 = most similar).
            Each dict contains:
                rank            int     1-indexed rank
                score           float   cosine similarity score
                row_index       int     row in embedding / metadata files
                subject_id      str     patient ID
                study_id        str     study ID
                image_path      str     path relative to knowledge base
                report          str     full radiology report text
                text_embedding  list    512-dim text embedding (for optional reranking)

        Raises:
            FileNotFoundError : query image not found
            RuntimeError      : subprocess failure or missing output file
        """
        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Query image not found: {image_path}")

        log.info("=" * 60)
        log.info("Retrieval started")
        log.info(f"  Image : {image_path.name}")
        log.info(f"  top_k : {top_k}")
        log.info("=" * 60)
        check_ram("before Phase 1")

        # ── Phase 1: Embed query image in subprocess ──
        _run_subprocess(STEP1_SCRIPT, args=[str(image_path)], label="Phase 1 | embed")

        if not QUERY_VEC_FILE.exists():
            raise RuntimeError(
                "Phase 1 exited cleanly but query_vec.npy was not created.\n"
                f"Expected at: {QUERY_VEC_FILE}"
            )
        check_ram("after Phase 1 — model memory reclaimed by OS")

        # ── Phase 2: Search index + fetch reports in subprocess ──
        _run_subprocess(STEP2_SCRIPT, args=[str(top_k)], label="Phase 2 | search")

        if not RESULTS_FILE.exists():
            raise RuntimeError(
                "Phase 2 exited cleanly but results.json was not created.\n"
                f"Expected at: {RESULTS_FILE}"
            )
        check_ram("after Phase 2")

        # ── Read results from disk ──
        results = self._load_results()
        log.info(f"Retrieval complete — {len(results)} results returned.")
        return results

    def format_context(
        self,
        results: list[dict],
        max_reports: int = FORMAT_MAX_REPORTS,
        max_report_chars: int = MAX_REPORT_CHARS,
    ) -> str:
        """
        Format retrieved reports into a prompt-ready context string for the generator.

        Args:
            results          : List of result dicts from retrieve().
            max_reports      : How many reports to include (default 3).
            max_report_chars : Truncate each report to this many characters.

        Returns:
            Multi-line string ready to inject into the generator's prompt.

        Example output:
            Retrieved Radiology Reports (most similar chest X-rays):

            [1] Study 12345678 | Patient 10000032 | Similarity: 0.9231
            Findings: The lungs are clear. No focal consolidation ...
            ---
            [2] Study 12345679 | Patient 10000033 | Similarity: 0.9104
            ...
        """
        if not results:
            return "No similar cases retrieved."

        lines = ["Retrieved Radiology Reports (most similar chest X-rays):\n"]
        for r in results[:max_reports]:
            report = r.get("report", "").strip()
            if len(report) > max_report_chars:
                report = report[:max_report_chars] + "… [truncated]"

            lines.append(
                f"[{r['rank']}] Study {r['study_id']} | "
                f"Patient {r['subject_id']} | "
                f"Similarity: {r['score']:.4f}"
            )
            lines.append(report)
            lines.append("---")

        return "\n".join(lines)

    def _load_results(self) -> list[dict]:
        try:
            with open(RESULTS_FILE) as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(f"Failed to read results.json: {e}")
        if not isinstance(results, list) or not results:
            raise RuntimeError("results.json is empty or malformed.")
        return results


# ─────────────────────────────────────────────
# STANDALONE TEST  (entry point)
# ─────────────────────────────────────────────

def main():
    import pandas as pd

    print("=" * 60)
    print("retriever.py — Retrieval pipeline test")
    print("=" * 60)
    check_ram("baseline")

    # ── Validate test dataset ──
    if not TEST_DATASET_CSV.exists():
        log.error(f"test_dataset.csv not found at {TEST_DATASET_CSV}")
        sys.exit(1)

    test_df = pd.read_csv(TEST_DATASET_CSV).dropna(subset=["image_path", "report"])
    if test_df.empty:
        log.error("test_dataset.csv has no valid rows (image_path + report both required).")
        sys.exit(1)

    # ── Find first test image that actually exists on disk ──
    test_image_path = None
    ground_truth_report = None

    for _, row in test_df.iterrows():
        candidate = IMAGE_BASE_DIR / row["image_path"]
        if candidate.exists():
            test_image_path     = candidate
            ground_truth_report = str(row["report"]).strip()
            break

    if test_image_path is None:
        log.error(
            "Could not find any test image on disk. "
            "Check that IMAGE_BASE_DIR and image_path values in test_dataset.csv are correct.\n"
            f"IMAGE_BASE_DIR = {IMAGE_BASE_DIR}"
        )
        sys.exit(1)

    # ── Print query info ──
    print(f"\n{'─' * 60}")
    print(f"Query image      : {test_image_path}")
    print(f"Ground truth (first 300 chars):")
    print(f"  {ground_truth_report[:300]}…")
    print(f"{'─' * 60}")

    # ── Run retrieval ──
    try:
        retriever = Retriever()
        results   = retriever.retrieve(str(test_image_path), top_k=DEFAULT_TOP_K)
    except (FileNotFoundError, RuntimeError) as e:
        log.error(str(e))
        sys.exit(1)

    # ── Print results table ──
    print(f"\n{'=' * 60}")
    print(f"RETRIEVAL RESULTS — top {len(results)} studies")
    print(f"{'=' * 60}")
    for r in results:
        print(f"\n  Rank       : {r['rank']}")
        print(f"  Study ID   : {r['study_id']}")
        print(f"  Subject ID : {r['subject_id']}")
        print(f"  Score      : {r['score']:.4f}")
        print(f"  Image      : {r['image_path']}")
        preview = r["report"][:300].replace("\n", " ")
        print(f"  Report     : {preview}…")

    # ── Print formatted context ──
    print(f"\n{'=' * 60}")
    print(f"FORMATTED CONTEXT (top {FORMAT_MAX_REPORTS} reports — for generator input)")
    print(f"{'=' * 60}")
    context = retriever.format_context(results)
    print(context)

    check_ram("end")
    print(f"\n{'=' * 60}")
    print("Retrieval test complete ✓")
    print("Next step: build src/generation/generator.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
