"""
evaluate.py

Evaluation script for the RAG radiology report generation pipeline.
Place this file at the project root: rag-rrg/evaluate.py

Runs the full pipeline on a stratified random sample of held-out test images
and measures report quality against ground truth using three metric families:

    BLEU        n-gram precision  (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
    ROUGE       n-gram recall     (ROUGE-1, ROUGE-2, ROUGE-L F1)
    BERTScore   semantic similarity via contextual RoBERTa embeddings
                (precision, recall, F1)

Note on RadGraph: Evaluated but dropped due to a hard incompatibility between
the radgraph library's bundled AllenNLP tokenizer and transformers 4.x+
(AttributeError: BertTokenizer / TokenizersBackend has no attribute encode_plus).
Both radgraph-xl and modern-radgraph-xl model types exhibit the same failure.
The three working metrics above are standard in the RRG literature.

Each run creates a new timestamped subfolder under rag-rrg/evaluation/ so
previous results are never overwritten and every run is independently archived.

Results are saved incrementally so evaluation can be interrupted and resumed
with --resume pointing at the existing run folder.

Output layout:
    evaluation/
    └── run_YYYYMMDD_HHMMSS/
        ├── sample_results/
        │   ├── result_{study_id}.json    per-image: report + GT + scores
        │   └── ...
        ├── aggregate_metrics.json        mean ± std for all metrics
        ├── evaluation_summary.txt        human-readable table
        └── run_config.json              config snapshot for this run

Usage:
    python3 evaluate.py                               # 100 samples, new run
    python3 evaluate.py --n_samples 50                # quick smoke-test
    python3 evaluate.py --n_samples 200               # fuller evaluation
    python3 evaluate.py --resume evaluation/run_YYYYMMDD_HHMMSS
                                                      # resume interrupted run
"""

import sys
import json
import time
import logging
import argparse
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent   # rag-rrg/
EVAL_ROOT    = PROJECT_ROOT / "evaluation"       # rag-rrg/evaluation/
TEST_CSV     = PROJECT_ROOT / "data" / "test_dataset.csv"
IMAGE_BASE   = PROJECT_ROOT / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────
# LOGGING
# Root logger starts with stdout only.
# A file handler is added once the run folder is created.
# ─────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
EVAL_ROOT.mkdir(parents=True, exist_ok=True)

_LOG_FORMAT = "%(asctime)s [evaluate] %(levelname)s — %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT,
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def _add_file_handlers(run_dir: Path):
    """Attach per-run and shared log file handlers after run folder is ready."""
    fmt = logging.Formatter(_LOG_FORMAT)
    for path in [run_dir / "evaluation.log", LOG_DIR / "evaluation.log"]:
        fh = logging.FileHandler(path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logging.getLogger().addHandler(fh)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEFAULT_N_SAMPLES   = 100
RANDOM_SEED         = 42
INTER_IMAGE_SLEEP_S = 3      # seconds between runs — Groq free tier buffer

MIN_GT_WORD_COUNT   = 30
COMPARATIVE_PHRASES = [
    "unchanged",
    "no significant interval change",
    "no interval change",
    "stable compared",
    "similar to prior",
    "similar to previous",
    "as compared to",
    "compared to prior",
    "compared to previous",
    "as before",
    "no change since",
    "no change from",
    "no acute change",
    "no new",
]


# ─────────────────────────────────────────────
# GROUND TRUTH FILTER
# ─────────────────────────────────────────────

class GroundTruthFilter:
    """
    Filters MIMIC-CXR ground truth reports unsuitable for evaluating
    a de-novo generation system.

    Rejects:
        1. Too short — under MIN_GT_WORD_COUNT words.
        2. Comparative language — references prior studies the pipeline
           has no access to.
    """

    def is_valid(self, report: str) -> bool:
        return self._reject_reason(report) is None

    def reject_reason(self, report: str) -> str:
        return self._reject_reason(report) or ""

    def _reject_reason(self, report: str):
        if not report or not report.strip():
            return "empty"
        words = report.strip().split()
        if len(words) < MIN_GT_WORD_COUNT:
            return f"too_short ({len(words)} words < {MIN_GT_WORD_COUNT})"
        lower = report.lower()
        for phrase in COMPARATIVE_PHRASES:
            if phrase in lower:
                return f"comparative (contains '{phrase}')"
        return None


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

class EvaluationMetrics:
    """
    BLEU, ROUGE, and BERTScore for generated vs. ground truth reports.
    Each metric is computed independently — failure of one does not stop others.
    Returns flat dicts of floats; uses None for failed computations.
    """

    # ── BLEU ───────────────────────────────────────────────────────────────

    def compute_bleu(self, hypothesis: str, reference: str) -> dict:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize

            hyp_tokens = word_tokenize(hypothesis.lower())
            ref_tokens = word_tokenize(reference.lower())
            if not hyp_tokens or not ref_tokens:
                return self._null_bleu()

            sf = SmoothingFunction().method1
            scores = {}
            for n in range(1, 5):
                weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
                scores[f"bleu_{n}"] = float(
                    sentence_bleu([ref_tokens], hyp_tokens,
                                  weights=weights, smoothing_function=sf)
                )
            return scores
        except Exception as e:
            log.warning(f"BLEU failed: {e}")
            return self._null_bleu()

    def _null_bleu(self):
        return {"bleu_1": None, "bleu_2": None, "bleu_3": None, "bleu_4": None}

    # ── ROUGE ──────────────────────────────────────────────────────────────

    def compute_rouge(self, hypothesis: str, reference: str) -> dict:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            s = scorer.score(reference, hypothesis)
            return {
                "rouge_1_precision" : float(s["rouge1"].precision),
                "rouge_1_recall"    : float(s["rouge1"].recall),
                "rouge_1_f1"        : float(s["rouge1"].fmeasure),
                "rouge_2_precision" : float(s["rouge2"].precision),
                "rouge_2_recall"    : float(s["rouge2"].recall),
                "rouge_2_f1"        : float(s["rouge2"].fmeasure),
                "rouge_l_precision" : float(s["rougeL"].precision),
                "rouge_l_recall"    : float(s["rougeL"].recall),
                "rouge_l_f1"        : float(s["rougeL"].fmeasure),
            }
        except Exception as e:
            log.warning(f"ROUGE failed: {e}")
            return self._null_rouge()

    def _null_rouge(self):
        return {k: None for k in [
            "rouge_1_precision", "rouge_1_recall", "rouge_1_f1",
            "rouge_2_precision", "rouge_2_recall", "rouge_2_f1",
            "rouge_l_precision", "rouge_l_recall", "rouge_l_f1",
        ]}

    # ── BERTScore ──────────────────────────────────────────────────────────

    def compute_bertscore(self, hypothesis: str, reference: str) -> dict:
        try:
            from bert_score import score as bert_score
            P, R, F1 = bert_score(
                cands=[hypothesis], refs=[reference],
                lang="en", model_type="roberta-large",
                verbose=False, device="cpu",
            )
            return {
                "bertscore_precision" : float(P[0]),
                "bertscore_recall"    : float(R[0]),
                "bertscore_f1"        : float(F1[0]),
            }
        except Exception as e:
            log.warning(f"BERTScore failed: {e}")
            return self._null_bertscore()

    def _null_bertscore(self):
        return {"bertscore_precision": None,
                "bertscore_recall": None, "bertscore_f1": None}

    # ── Combined ───────────────────────────────────────────────────────────

    def compute_all(self, hypothesis: str, reference: str) -> dict:
        result = {}
        result.update(self.compute_bleu(hypothesis, reference))
        result.update(self.compute_rouge(hypothesis, reference))
        result.update(self.compute_bertscore(hypothesis, reference))
        return result


# ─────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────

def load_evaluation_sample(n_samples: int) -> tuple[list[dict], int]:
    """
    Load, filter, and sample from test_dataset.csv.

    Returns (sample_rows, n_filtered_gt).
    Each row: study_id, subject_id, image_path (absolute str), report.
    """
    import pandas as pd

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"test_dataset.csv not found at {TEST_CSV}.")

    df = pd.read_csv(TEST_CSV)
    log.info(f"test_dataset.csv: {len(df)} rows")

    before = len(df)
    df = df.dropna(subset=["image_path", "report"])
    log.info(f"After dropna: {len(df)} rows ({before - len(df)} dropped)")

    before = len(df)
    df["_abs_path"] = df["image_path"].apply(lambda p: IMAGE_BASE / p)
    df = df[df["_abs_path"].apply(lambda p: p.exists())]
    log.info(f"After image existence check: {len(df)} rows ({before - len(df)} missing)")

    gt_filter  = GroundTruthFilter()
    before     = len(df)
    valid_mask = df["report"].apply(lambda r: gt_filter.is_valid(str(r)))
    df_valid   = df[valid_mask]
    n_filtered = before - len(df_valid)

    log.info(f"After GT filter: {len(df_valid)} rows ({n_filtered} filtered out)")
    if n_filtered > 0:
        for r in df[~valid_mask]["report"].head(5):
            log.info(f"  [{gt_filter.reject_reason(str(r))}]  {str(r)[:80]}…")

    df = df_valid
    if len(df) == 0:
        raise RuntimeError("No valid evaluation samples after filtering.")

    actual_n = min(n_samples, len(df))
    if actual_n < n_samples:
        log.warning(f"Only {actual_n} valid rows available (requested {n_samples}).")

    df_sample = df.sample(n=actual_n, random_state=RANDOM_SEED).reset_index(drop=True)
    log.info(f"Sampled {actual_n} images (seed={RANDOM_SEED})")

    rows = []
    for _, row in df_sample.iterrows():
        rows.append({
            "study_id"   : str(row.get("study_id",   "unknown")),
            "subject_id" : str(row.get("subject_id", "unknown")),
            "image_path" : str(row["_abs_path"]),
            "report"     : str(row["report"]).strip(),
        })
    return rows, n_filtered


# ─────────────────────────────────────────────
# AGGREGATE STATISTICS
# ─────────────────────────────────────────────

def compute_aggregate(results_dir: Path) -> dict:
    """Compute mean ± std per metric from all completed result JSONs."""
    all_metrics: dict[str, list[float]] = {}

    for f in sorted(results_dir.glob("result_*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if data.get("status") != "completed":
                continue
            for key, val in data.get("metrics", {}).items():
                if val is not None:
                    all_metrics.setdefault(key, []).append(float(val))
        except Exception as e:
            log.warning(f"Could not read {f.name}: {e}")

    aggregate = {}
    for key, values in all_metrics.items():
        if values:
            aggregate[key] = {
                "mean" : float(np.mean(values)),
                "std"  : float(np.std(values)),
                "min"  : float(np.min(values)),
                "max"  : float(np.max(values)),
                "n"    : len(values),
            }
    return aggregate


# ─────────────────────────────────────────────
# SUMMARY WRITER
# ─────────────────────────────────────────────

def write_summary(
    run_dir: Path, aggregate: dict,
    n_requested: int, n_completed: int, n_failed: int,
    n_filtered: int, model_used: str, elapsed_mins: float,
) -> str:
    lines = []
    lines.append("=" * 64)
    lines.append("RAG Radiology Report Generation — Evaluation Results")
    lines.append("=" * 64)
    lines.append(f"Date           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Run folder     : {run_dir.name}")
    lines.append(f"Dataset        : MIMIC-CXR test split")
    lines.append(f"Model          : {model_used}")
    lines.append(f"Samples req.   : {n_requested}")
    lines.append(f"GT filtered    : {n_filtered}  (comparative / too short)")
    lines.append(f"Completed      : {n_completed}")
    lines.append(f"Failed         : {n_failed}")
    lines.append(f"Elapsed        : {elapsed_mins:.1f} minutes")
    lines.append(f"Metrics        : BLEU, ROUGE, BERTScore")
    lines.append(f"Note           : RadGraph dropped — transformers incompatibility")
    lines.append("")

    metric_groups = [
        ("BLEU",      ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]),
        ("ROUGE",     ["rouge_1_f1", "rouge_2_f1", "rouge_l_f1"]),
        ("BERTScore", ["bertscore_precision", "bertscore_recall", "bertscore_f1"]),
    ]

    lines.append(
        f"{'Metric':<28}  {'Mean':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}  {'N':>5}"
    )
    lines.append("─" * 64)

    for group_name, keys in metric_groups:
        lines.append(group_name)
        for key in keys:
            if key in aggregate:
                a = aggregate[key]
                lines.append(
                    f"  {key:<26}  "
                    f"{a['mean']:>7.4f}  {a['std']:>7.4f}  "
                    f"{a['min']:>7.4f}  {a['max']:>7.4f}  {a['n']:>5}"
                )
            else:
                lines.append(f"  {key:<26}  {'N/A':>7}")
        lines.append("")

    lines.append("=" * 64)
    summary = "\n".join(lines)

    try:
        with open(run_dir / "evaluation_summary.txt", "w") as f:
            f.write(summary)
        log.info(f"Summary saved → {run_dir / 'evaluation_summary.txt'}")
    except OSError as e:
        log.error(f"Could not write summary: {e}")

    return summary


# ─────────────────────────────────────────────
# RUN FOLDER SETUP
# ─────────────────────────────────────────────

def _setup_run_folder(resume: str | None) -> tuple[Path, Path, bool]:
    """
    Create a new timestamped run folder, or validate an existing one for resume.

    Returns (run_dir, results_dir, is_resume).
    """
    if resume:
        run_dir = Path(resume)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Resume folder not found: {run_dir}\n"
                "Pass the path relative to the project root, "
                "e.g. evaluation/run_20260412_091233"
            )
        results_dir = run_dir / "sample_results"
        results_dir.mkdir(exist_ok=True)
        log.info(f"Resuming from: {run_dir}")
        return run_dir, results_dir, True

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EVAL_ROOT / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = run_dir / "sample_results"
    results_dir.mkdir()
    log.info(f"New run folder: {run_dir}")
    return run_dir, results_dir, False


# ─────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────

def run_evaluation(n_samples: int, resume: str | None = None):

    # ── Run folder ─────────────────────────────────────────────────────────
    run_dir, results_dir, is_resume = _setup_run_folder(resume)
    _add_file_handlers(run_dir)

    log.info("=" * 64)
    log.info("Evaluation started")
    log.info(f"  n_samples  : {n_samples}")
    log.info(f"  run_dir    : {run_dir}")
    log.info(f"  resume     : {is_resume}")
    log.info("=" * 64)
    start_time = time.time()

    # ── Pipeline ───────────────────────────────────────────────────────────
    log.info("Importing Pipeline …")
    try:
        from pipeline import Pipeline
    except ImportError as e:
        log.error(f"Could not import Pipeline: {e}")
        sys.exit(1)

    pipe    = Pipeline(top_k=5)
    metrics = EvaluationMetrics()

    # ── Sample ─────────────────────────────────────────────────────────────
    log.info("Loading evaluation sample …")
    try:
        sample, n_filtered = load_evaluation_sample(n_samples)
    except (FileNotFoundError, RuntimeError) as e:
        log.error(str(e))
        sys.exit(1)

    n_requested = len(sample)
    n_completed = 0
    n_failed    = 0
    n_skipped   = 0
    model_used  = "unknown"

    # ── Save run config ────────────────────────────────────────────────────
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump({
                "started_at"   : datetime.now().isoformat(),
                "n_samples"    : n_samples,
                "n_requested"  : n_requested,
                "n_filtered_gt": n_filtered,
                "random_seed"  : RANDOM_SEED,
                "metrics"      : ["bleu", "rouge", "bertscore"],
                "note"         : "RadGraph dropped — transformers incompatibility",
            }, f, indent=2)

    # ── Per-image loop ─────────────────────────────────────────────────────
    for i, row in enumerate(sample, start=1):
        study_id     = row["study_id"]
        image_path   = row["image_path"]
        ground_truth = row["report"]

        result_file = results_dir / f"result_{study_id}.json"

        log.info(
            f"[{i:>3}/{n_requested}]  "
            f"study={study_id}  "
            f"image={Path(image_path).name}"
        )

        # Resume: skip already-completed images
        if result_file.exists():
            try:
                with open(result_file) as f:
                    existing = json.load(f)
                if existing.get("status") == "completed":
                    log.info("  → Already completed. Skipping.")
                    n_skipped   += 1
                    n_completed += 1
                    if model_used == "unknown":
                        model_used = existing.get("model_used", "unknown")
                    continue
            except Exception:
                pass  # corrupt — re-run

        # Run pipeline
        pipeline_result = None
        try:
            pipeline_result = pipe.run(
                image_path   = image_path,
                ground_truth = ground_truth,
                save         = True,
            )
            model_used = pipeline_result.get("model_used", model_used)
        except Exception as e:
            log.error(f"  Pipeline failed: {e}")
            log.debug(traceback.format_exc())
            n_failed += 1
            try:
                with open(result_file, "w") as f:
                    json.dump({
                        "study_id"     : study_id,
                        "image_path"   : image_path,
                        "ground_truth" : ground_truth,
                        "status"       : "failed",
                        "error"        : str(e),
                        "evaluated_at" : datetime.now().isoformat(),
                    }, f, indent=2)
            except OSError:
                pass
            time.sleep(INTER_IMAGE_SLEEP_S)
            continue

        # Build hypothesis
        findings   = pipeline_result.get("findings",   "").strip()
        impression = pipeline_result.get("impression", "").strip()
        hypothesis = " ".join(filter(None, [findings, impression]))

        if not hypothesis:
            log.warning(f"  Empty hypothesis for study {study_id} — skipping metrics.")
            n_failed += 1
            time.sleep(INTER_IMAGE_SLEEP_S)
            continue

        # Compute metrics
        log.info("  Computing metrics …")
        metric_scores = metrics.compute_all(hypothesis, ground_truth)

        def _fmt(v) -> str:
            return f"{v:.4f}" if v is not None else "N/A"

        log.info(
            f"  BLEU-1={_fmt(metric_scores.get('bleu_1'))}  "
            f"ROUGE-1={_fmt(metric_scores.get('rouge_1_f1'))}  "
            f"BERTScore={_fmt(metric_scores.get('bertscore_f1'))}"
        )

        # Save per-image result
        per_image_result = {
            "study_id"             : study_id,
            "subject_id"           : row["subject_id"],
            "image_path"           : image_path,
            "model_used"           : model_used,
            "evaluated_at"         : datetime.now().isoformat(),
            "status"               : "completed",
            "caption"              : pipeline_result.get("caption",     ""),
            "description"          : pipeline_result.get("description", ""),
            "n_retrieved"          : pipeline_result.get("n_retrieved",  0),
            "generated_findings"   : findings,
            "generated_impression" : impression,
            "generated_full"       : hypothesis,
            "ground_truth"         : ground_truth,
            "metrics"              : metric_scores,
        }

        try:
            with open(result_file, "w") as f:
                json.dump(per_image_result, f, indent=2)
            log.info(f"  Saved → {result_file.name}")
        except OSError as e:
            log.error(f"  Could not save result: {e}")

        n_completed += 1

        if i < n_requested:
            time.sleep(INTER_IMAGE_SLEEP_S)

    # ── Aggregate ──────────────────────────────────────────────────────────
    log.info("=" * 64)
    log.info("Computing aggregate metrics …")

    aggregate    = compute_aggregate(results_dir)
    elapsed_mins = (time.time() - start_time) / 60.0

    agg_output = {
        "evaluated_at"    : datetime.now().isoformat(),
        "run_folder"      : str(run_dir),
        "n_requested"     : n_requested,
        "n_completed"     : n_completed - n_skipped,
        "n_resumed"       : n_skipped,
        "n_failed"        : n_failed,
        "n_filtered_gt"   : n_filtered,
        "model_used"      : model_used,
        "random_seed"     : RANDOM_SEED,
        "elapsed_minutes" : round(elapsed_mins, 2),
        "metrics"         : aggregate,
    }

    agg_path = run_dir / "aggregate_metrics.json"
    try:
        with open(agg_path, "w") as f:
            json.dump(agg_output, f, indent=2)
        log.info(f"Aggregate saved → {agg_path}")
    except OSError as e:
        log.error(f"Could not save aggregate_metrics.json: {e}")

    # ── Summary ────────────────────────────────────────────────────────────
    summary = write_summary(
        run_dir=run_dir, aggregate=aggregate,
        n_requested=n_requested, n_completed=n_completed,
        n_failed=n_failed, n_filtered=n_filtered,
        model_used=model_used, elapsed_mins=elapsed_mins,
    )

    print(f"\n{summary}")
    log.info(f"Done in {elapsed_mins:.1f} minutes.")
    log.info(f"Run folder  → {run_dir}")
    log.info(f"Aggregate   → {agg_path}")
    log.info(f"Summary     → {run_dir / 'evaluation_summary.txt'}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG radiology report generation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_samples", type=int, default=DEFAULT_N_SAMPLES,
        help="Number of test images to evaluate.",
    )
    parser.add_argument(
        "--resume", type=str, default=None, metavar="RUN_FOLDER",
        help=(
            "Path to an existing run folder to resume. "
            "Example: evaluation/run_20260412_091233"
        ),
    )
    args = parser.parse_args()

    if args.n_samples < 1:
        log.error("--n_samples must be at least 1.")
        sys.exit(1)

    log.info(
        f"evaluate.py  n_samples={args.n_samples}  "
        f"resume={args.resume or 'no (new run)'}"
    )
    run_evaluation(n_samples=args.n_samples, resume=args.resume)


if __name__ == "__main__":
    main()
