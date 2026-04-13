# ─────────────────────────────────────────────────────────────
# RAG Radiology Report Generation — Makefile
# Place at project root: rag-rrg/Makefile
#
# Usage:
#   make run              Run full pipeline on one test image
#   make retrieve         Run retrieval stage only
#   make generate         Run generation stage only (needs retrieve first)
#   make evaluate         Evaluate on 100 test images
#   make evaluate-full    Evaluate on 200 test images
#   make test             Run all tests
#   make test-unit        Run unit tests only (fast, no API)
#   make test-integration Run integration tests
#   make api              Start the FastAPI server locally
#   make install          Install all dependencies
#   make deploy-check     Verify project is ready for Render deployment
#   make clean            Remove intermediate pipeline files
#   make clean-eval       Remove all evaluation run folders
#   make clean-reports    Remove all generated report JSONs
#   make logs             Tail the pipeline log live
# ─────────────────────────────────────────────────────────────

PYTHON = python3
PORT   = 8000

.PHONY: run retrieve generate evaluate evaluate-full \
        test test-unit test-integration \
        api install deploy-check \
        clean clean-eval clean-reports logs help

# ── Pipeline ──────────────────────────────────────────────────

run:
	$(PYTHON) pipeline.py

retrieve:
	$(PYTHON) src/retrieval/retriever.py

generate:
	$(PYTHON) src/generation/generator.py

# ── Evaluation ────────────────────────────────────────────────

evaluate:
	$(PYTHON) evaluate.py --n_samples 100

evaluate-full:
	$(PYTHON) evaluate.py --n_samples 200

evaluate-smoke:
	$(PYTHON) evaluate.py --n_samples 10

# ── Tests ─────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v

test-unit:
	$(PYTHON) -m pytest tests/unit/ -v

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v

test-live:
	$(PYTHON) -m pytest tests/integration/ -v -s

# ── API server ────────────────────────────────────────────────

api:
	$(PYTHON) -m uvicorn api:app --host 0.0.0.0 --port $(PORT) --reload

api-prod:
	$(PYTHON) -m uvicorn api:app --host 0.0.0.0 --port $(PORT) --workers 1

# ── Install ───────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-nltk:
	$(PYTHON) -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# ── Deployment check ──────────────────────────────────────────

deploy-check:
	@echo "Checking deployment readiness..."
	@test -f requirements.txt     && echo "  OK  requirements.txt" || echo "  MISSING  requirements.txt"
	@test -f api.py               && echo "  OK  api.py"           || echo "  MISSING  api.py — build this before deploying"
	@test -f pipeline.py          && echo "  OK  pipeline.py"      || echo "  MISSING  pipeline.py"
	@test -f .env                 && echo "  OK  .env"             || echo "  MISSING  .env — copy from .env.example and fill values"
	@test -f .env.example         && echo "  OK  .env.example"     || echo "  MISSING  .env.example"
	@test -d src/embedding        && echo "  OK  src/embedding/"   || echo "  MISSING  src/embedding/"
	@test -d src/indexing         && echo "  OK  src/indexing/"    || echo "  MISSING  src/indexing/"
	@test -d src/retrieval        && echo "  OK  src/retrieval/"   || echo "  MISSING  src/retrieval/"
	@test -d src/generation       && echo "  OK  src/generation/"  || echo "  MISSING  src/generation/"
	@test -f src/indexing/faiss_image_index.bin \
	                              && echo "  OK  faiss index"      || echo "  MISSING  faiss_image_index.bin — required at runtime"
	@test -f src/embedding/image_embeddings.npy \
	                              && echo "  OK  image embeddings" || echo "  MISSING  image_embeddings.npy — required at runtime"
	@echo ""
	@echo "NOTE: Large binary files (faiss index, embeddings) must be"
	@echo "      mounted as a Render Disk, not committed to git."

# ── Clean ─────────────────────────────────────────────────────

clean:
	rm -f src/retrieval/query_vec.npy
	rm -f src/retrieval/results.json
	rm -f src/generation/caption_result.json
	rm -f src/generation/generation_result.json
	@echo "Intermediate pipeline files removed."

clean-reports:
	rm -f data/generated_reports/*.json
	rm -f data/pipeline_results/*.json
	@echo "Generated report files removed."

clean-eval:
	rm -rf evaluation/run_*/
	@echo "Evaluation run folders removed."

# ── Logs ─────────────────────────────────────────────────────

logs:
	tail -f logs/pipeline.log

logs-generation:
	tail -f logs/generation.log

logs-retrieval:
	tail -f logs/retrieval.log

logs-eval:
	tail -f logs/evaluation.log

# ── Help ─────────────────────────────────────────────────────

help:
	@echo ""
	@echo "RAG Radiology Report Generation — available commands:"
	@echo ""
	@echo "  make run              Run full pipeline on one test image"
	@echo "  make retrieve         Retrieval stage only"
	@echo "  make generate         Generation stage only"
	@echo "  make evaluate         Evaluate 100 test images"
	@echo "  make evaluate-full    Evaluate 200 test images"
	@echo "  make evaluate-smoke   Evaluate 10 images (quick check)"
	@echo "  make test             All tests"
	@echo "  make test-unit        Unit tests only (fast)"
	@echo "  make test-integration Integration tests"
	@echo "  make api              Start FastAPI server (dev mode)"
	@echo "  make install          Install dependencies"
	@echo "  make deploy-check     Verify Render deployment readiness"
	@echo "  make clean            Remove intermediate files"
	@echo "  make clean-eval       Remove evaluation run folders"
	@echo "  make clean-reports    Remove generated report JSONs"
	@echo "  make logs             Tail pipeline.log"
	@echo ""
