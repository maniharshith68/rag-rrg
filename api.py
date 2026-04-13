"""
api.py

FastAPI service wrapping the RAG radiology report generation pipeline.
Place at project root: rag-rrg/api.py

Endpoints:
    GET  /health       service health check — always fast, no pipeline call
    POST /generate     upload a chest X-ray, receive a radiology report JSON
    GET  /docs         auto-generated Swagger UI (FastAPI built-in)
    GET  /redoc        ReDoc documentation UI

Run locally (dev mode with auto-reload):
    make api
    or: uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Run locally (production mode, mirrors Render):
    make api-prod
    or: uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

On Render:
    Start command: uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1
    Environment variables: GROQ_API_KEY, GEMINI_API_KEY, GENERATION_BACKEND,
                           HUGGINGFACE_TOKEN, GEMINI_MODEL, DATA_MOUNT

Architecture note:
    The Pipeline class is instantiated ONCE at startup and reused for every
    request. This is intentional — the subprocess architecture means no model
    weights are held in the API process memory. Each request spawns subprocesses
    (step1_embed, step2_search, step1_caption, step2_generate) that load models,
    run inference, and exit. The API process stays lightweight between requests.

    Workers is set to 1 on Render because the subprocess architecture already
    handles memory isolation. Multiple uvicorn workers would multiply RAM usage
    without benefit since each worker would spawn its own subprocesses.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# ─────────────────────────────────────────────
# PATHS AND ENVIRONMENT
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

# DATA_MOUNT is set on Render to point at the persistent disk.
# Locally it is empty — all paths resolve relative to PROJECT_ROOT.
DATA_MOUNT = os.environ.get("DATA_MOUNT", "").strip()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Override embedding and index paths when running on Render disk
if DATA_MOUNT:
    mount = Path(DATA_MOUNT)
    os.environ.setdefault(
        "EMBEDDING_DIR",
        str(mount),
    )
    os.environ.setdefault(
        "FAISS_INDEX_FILE",
        str(mount / "faiss_image_index.bin"),
    )


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [api] %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "api.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STARTUP / SHUTDOWN (lifespan)
# ─────────────────────────────────────────────

_pipeline = None   # global — set once at startup, reused per request


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline at startup; nothing to clean up on shutdown."""
    global _pipeline
    log.info("=" * 60)
    log.info("API startup — loading pipeline …")
    log.info(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    log.info(f"  DATA_MOUNT   : {DATA_MOUNT or '(local dev — no mount)'}")
    try:
        from pipeline import Pipeline
        _pipeline = Pipeline(top_k=5)
        log.info("Pipeline loaded and ready.")
    except Exception as e:
        log.error(f"Pipeline failed to load: {e}")
        log.error(
            "The API will start but /generate will return 503 until the "
            "pipeline is fixed and the service is redeployed."
        )
        _pipeline = None
    log.info("=" * 60)
    yield
    log.info("API shutdown.")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="RAG Radiology Report Generation API",
    description=(
        "Accepts a chest X-ray image (JPG or PNG) and returns a generated "
        "radiology report using BioMedCLIP image retrieval and a vision LLM "
        "(Groq Llama 4 Scout / Gemini Flash).\n\n"
        "**Usage**: POST a chest X-ray to `/generate`. "
        "The response contains `findings` and `impression` sections.\n\n"
        "**Dataset**: MIMIC-CXR (199,214 image-report pairs as knowledge base)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins so the API can be called from any frontend.
# Tighten this in production if you want to restrict access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get(
    "/health",
    summary="Health check",
    description="Returns service status and pipeline readiness. Always fast.",
    tags=["monitoring"],
)
async def health():
    return {
        "status"          : "ok",
        "pipeline_ready"  : _pipeline is not None,
        "data_mount"      : DATA_MOUNT or "local",
        "timestamp"       : datetime.now().isoformat(),
    }


@app.post(
    "/generate",
    summary="Generate radiology report",
    description=(
        "Upload a chest X-ray image (JPG or PNG, max 20MB). "
        "Returns a structured radiology report with FINDINGS and IMPRESSION sections.\n\n"
        "**Processing time**: approximately 30–60 seconds per image "
        "(BioMedCLIP embedding + FAISS retrieval + LLM generation).\n\n"
        "**Note**: This endpoint is synchronous. For batch processing, "
        "call it sequentially with a delay between requests."
    ),
    tags=["generation"],
)
async def generate(
    request : Request,
    image   : UploadFile = File(
        ...,
        description="Chest X-ray image file (JPG or PNG, max 20MB)."
    ),
):
    # ── Pipeline availability check ────────────────────────────────────────
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Pipeline is not initialised. "
                "Check server logs for startup errors. "
                "Common causes: missing FAISS index or embedding files on the data mount."
            ),
        )

    # ── File type validation ───────────────────────────────────────────────
    filename = image.filename or "upload"
    suffix   = Path(filename).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                "Upload a JPEG (.jpg / .jpeg) or PNG (.png) chest X-ray."
            ),
        )

    # ── File size check (20MB limit, mirrors Groq vision API limit) ────────
    contents = await image.read()
    size_mb  = len(contents) / (1024 * 1024)
    if size_mb > 20:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Maximum allowed size is 20MB.",
        )

    log.info(
        f"Request received — file={filename}  "
        f"size={size_mb:.2f}MB  "
        f"client={request.client.host if request.client else 'unknown'}"
    )

    # ── Write upload to a temp file on disk ────────────────────────────────
    # The pipeline expects a real filesystem path, not an in-memory buffer,
    # because it spawns subprocesses that need to open the file independently.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, dir=tempfile.gettempdir()
        ) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
        log.info(f"Temp file written: {tmp_path}")
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write uploaded file to disk: {e}",
        )

    # ── Run pipeline ───────────────────────────────────────────────────────
    start = datetime.now()
    try:
        result = _pipeline.run(
            image_path   = str(tmp_path),
            ground_truth = "",   # no ground truth in API context
            save         = False, # do not write to data/pipeline_results/ on Render
        )
    except FileNotFoundError as e:
        log.error(f"Pipeline FileNotFoundError: {e}")
        raise HTTPException(
            status_code=500,
            detail=(
                f"Pipeline could not find a required file: {e}. "
                "Check that the FAISS index and embedding files are correctly "
                "mounted at DATA_MOUNT on this server."
            ),
        )
    except RuntimeError as e:
        log.error(f"Pipeline RuntimeError: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {e}",
        )
    except Exception as e:
        log.error(f"Pipeline unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected pipeline error: {e}",
        )
    finally:
        # Always clean up the temp file
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    elapsed_s = (datetime.now() - start).total_seconds()
    log.info(f"Pipeline complete — elapsed={elapsed_s:.1f}s")

    # ── Build and return response ──────────────────────────────────────────
    findings   = result.get("findings",    "").strip()
    impression = result.get("impression",  "").strip()

    if not findings and not impression:
        log.warning("Pipeline returned empty findings and impression.")
        raise HTTPException(
            status_code=500,
            detail=(
                "Pipeline ran successfully but returned empty report sections. "
                "This may indicate a problem with the LLM API response. "
                "Check the generation backend logs."
            ),
        )

    return JSONResponse(
        content={
            "findings"         : findings,
            "impression"       : impression,
            "caption"          : result.get("caption",      ""),
            "description"      : result.get("description",  ""),
            "model_used"       : result.get("model_used",   "unknown"),
            "n_retrieved"      : result.get("n_retrieved",   0),
            "elapsed_seconds"  : round(elapsed_s, 2),
            "generated_at"     : datetime.now().isoformat(),
        },
        status_code=200,
    )


# ─────────────────────────────────────────────
# LOCAL DEV ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host    = "0.0.0.0",
        port    = int(os.environ.get("PORT", 8000)),
        reload  = True,
        workers = 1,
    )
