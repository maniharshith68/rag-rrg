# RAG Pipeline for Radiology Report Generation

A Retrieval-Augmented Generation (RAG) system that takes a chest X-ray image as input, retrieves the most visually similar cases from a large clinical knowledge base, and generates a structured radiology report — complete with **FINDINGS** and **IMPRESSION** sections — using a vision-capable large language model.

Built and evaluated on **MIMIC-CXR**, a dataset of 227,827 chest radiographs paired with radiology reports from Beth Israel Deaconess Medical Center.

---

## How it works

A query chest X-ray enters the pipeline and passes through four sequential stages, each running in an isolated subprocess to manage memory on constrained hardware.

```
Query chest X-ray
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Stage 1 — Image Embedding                            │
│  BioMedCLIP (ViT-B/16) encodes the image into a      │
│  512-dimensional vector in a shared medical           │
│  vision-language space                                │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│  Stage 2 — Retrieval                                  │
│  FAISS IndexFlatIP searches 199,214 image embeddings  │
│  → top-5 most visually similar X-rays retrieved       │
│  → their radiology reports fetched from knowledge base│
│  → deduplicated by study ID                           │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│  Stage 3 — Image Captioning                           │
│  BioMedCLIP zero-shot similarity against a 53-phrase  │
│  clinical vocabulary produces an image caption and    │
│  structured description with confidence scores        │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│  Stage 4 — Report Generation                          │
│  Vision LLM (Groq Llama 4 Scout / Gemini Flash)       │
│  receives: image + caption + description +            │
│  top-3 retrieved reports → generates FINDINGS and     │
│  IMPRESSION as a board-certified radiologist          │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
         Structured radiology report
         (FINDINGS + IMPRESSION)
```

---

## Sample output

**Input:** Chest X-ray from MIMIC-CXR test set

**BioMedCLIP caption:**
> Mild pulmonary edema. Pacemaker leads present. Moderate pulmonary edema.

**Generated FINDINGS:**
> 1. **Cardiomediastinal contour**: The cardiac silhouette appears enlarged, suggesting possible cardiomegaly.
> 2. **Lungs**: The lung fields exhibit patchy, bilateral airspace opacities consistent with pulmonary edema. Increased vascular markings and interstitial markings further support the presence of pulmonary edema.
> 3. **Pleura**: Evidence of bilateral pleural effusions, more pronounced at the bases.
> 4. **Osseous structures**: The osseous structures appear intact without acute fractures.
> 5. **Tubes, lines, and devices**: Endotracheal tube in appropriate position. Pacemaker leads visible.

**Generated IMPRESSION:**
> Findings consistent with pulmonary edema as evidenced by patchy airspace opacities, increased vascular markings, and bilateral pleural effusions. Endotracheal tube and pacemaker leads noted.

---

## Key technical decisions

**Image-to-image retrieval, not cross-modal retrieval.** The FAISS index is built over image embeddings, not text embeddings. Querying a new image against image embeddings eliminates the cross-modal gap that would exist if comparing image vectors to text vectors. Retrieved reports come from the most visually similar X-rays, ensuring clinical relevance.

**Subprocess architecture for memory management.** The entire pipeline runs on a MacBook Air with Apple M3 and 8GB RAM — a severe constraint with macOS consuming ~3.5GB baseline. Each heavy operation (BioMedCLIP inference, FAISS search, LLM generation) runs in a separate Python subprocess. When a subprocess exits, the OS immediately reclaims all of its memory. This was the only reliable way to prevent OOM crashes; all single-process approaches caused segfaults.

**Zero-shot clinical captioning without a separate captioning model.** Rather than loading a second model to caption images, BioMedCLIP is reused in zero-shot mode: the query image is scored against a vocabulary of 53 clinical phrases (e.g. "bilateral pleural effusions", "mild cardiomegaly"). The top-scoring phrases become the caption, using the same model already loaded for embedding — no extra RAM cost.

**Free API backends with automatic fallback.** Generation uses Groq (free tier, Llama 4 Scout) as the primary backend and Gemini Flash (free tier) as automatic fallback. Both are vision-capable, cost $0, and require no credit card. The pipeline tries Groq first; if it fails (rate limit, network), it transparently retries with Gemini.

---

## Evaluation results

Evaluated on a filtered sample of 100 images from the MIMIC-CXR test split. Ground truth reports shorter than 30 words or containing purely comparative language ("unchanged", "similar to prior") were excluded — these reference prior studies the pipeline has no access to, making comparison meaningless. 22,034 of 49,804 test reports (44%) were filtered on this basis.

| Metric | Mean | Std |
|---|---|---|
| BLEU-1 | 0.187 | 0.042 |
| BLEU-4 | 0.028 | 0.035 |
| ROUGE-1 F1 | 0.271 | 0.067 |
| ROUGE-2 F1 | 0.080 | 0.059 |
| ROUGE-L F1 | 0.175 | 0.067 |
| BERTScore F1 | 0.841 | 0.026 |

The BERTScore F1 of **0.841** reflects strong semantic alignment between generated and reference reports. Lower BLEU/ROUGE scores are expected — radiology reports describe the same findings using varied formal phrasing, and exact n-gram overlap metrics systematically underestimate quality for this domain.

---

## Project structure

```
rag-rrg/
├── pipeline.py                    End-to-end orchestrator
├── evaluate.py                    Evaluation script (BLEU, ROUGE, BERTScore)
├── api.py                         FastAPI REST service
├── conftest.py                    Pytest auto-logging plugin
├── requirements.txt               Python dependencies
├── Makefile                       Workflow shortcuts
├── render.yaml                    Render deployment config
├── .env.example                   Environment variable template
│
├── config/
│   └── indexing.yml               FAISS index configuration
│
├── src/
│   ├── embedding/
│   │   ├── embedding.py           BioMedCLIP batch embedding with resume support
│   │   ├── image_embeddings.npy   (199,214 × 512, float32) — not in git
│   │   ├── text_embeddings.npy    (199,214 × 512, float32) — not in git
│   │   ├── metadata.csv           subject_id, study_id, image_path
│   │   └── checkpoint.txt         resume support for interrupted runs
│   │
│   ├── indexing/
│   │   ├── build_index.py         FAISS IndexFlatIP over image embeddings
│   │   ├── faiss_image_index.bin  (~800MB) — not in git
│   │   └── index_config.json      index metadata and modality record
│   │
│   ├── retrieval/
│   │   ├── retriever.py           Subprocess orchestrator
│   │   ├── step1_embed.py         Subprocess: embed query image, exit
│   │   └── step2_search.py        Subprocess: FAISS search, fetch reports, exit
│   │
│   └── generation/
│       ├── generator.py           Subprocess orchestrator
│       ├── step1_caption.py       Subprocess: zero-shot captioning, exit
│       └── step2_generate.py      Subprocess: LLM generation, exit
│
├── tests/
│   ├── unit/
│   │   ├── test_indexing.py       FAISS integrity, embedding alignment (28 tests)
│   │   ├── test_retrieval.py      results.json schema, deduplication (18 tests)
│   │   └── test_generation.py     caption/generation JSON schema (38 tests)
│   └── integration/
│       └── test_pipeline.py       End-to-end output validation (32 tests)
│
├── data/
│   ├── knowledge_base.csv         80% split (~199K rows) — not in git
│   ├── test_dataset.csv           20% split (~50K rows) — not in git
│   └── files/                     MIMIC-CXR images — not in git
│
├── evaluation/
│   └── run_YYYYMMDD_HHMMSS/       Per-run results (timestamped, never overwritten)
│       ├── sample_results/        Per-image JSON with metrics
│       ├── aggregate_metrics.json Mean ± std for all metrics
│       ├── evaluation_summary.txt Human-readable table
│       └── run_config.json        Run parameters snapshot
│
└── logs/                          All log files (retrieval, generation, tests)
```

---

## Setup

### Prerequisites

- Python 3.11 or higher
- Apple Silicon (MPS) or CUDA GPU recommended; CPU works but is slow for embedding
- MIMIC-CXR dataset access ([PhysioNet](https://physionet.org/content/mimic-cxr/)) — requires credentialed access
- HuggingFace account with BioMedCLIP access accepted

### Installation

```bash
git clone https://github.com/yourusername/rag-rrg.git
cd rag-rrg

pip install -r requirements.txt

# Download NLTK tokenizers required by BLEU metric
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Configuration

```bash
cp .env.example .env
# Edit .env and fill in:
#   PROJECT_ROOT      — absolute path to this directory
#   HUGGINGFACE_TOKEN — from https://huggingface.co/settings/tokens
#   GROQ_API_KEY      — from https://console.groq.com (free, no billing)
#   GEMINI_API_KEY    — from https://aistudio.google.com (free, no credit card)
```

### Dataset preparation

Place the MIMIC-CXR dataset under `data/files/` following the original directory structure (`files/p{XX}/p{subject_id}/s{study_id}/*.jpg`). The CSV split files (`knowledge_base.csv`, `test_dataset.csv`) are generated from the full dataset metadata.

### Build the knowledge base

```bash
# Step 1 — Generate embeddings for all 199,214 image-report pairs
# Takes several hours on first run; supports resume via checkpoint.txt
python3 src/embedding/embedding.py

# Step 2 — Build the FAISS index over image embeddings
python3 src/indexing/build_index.py
```

---

## Usage

### Run the full pipeline on a single image

```bash
make run
# or: python3 pipeline.py
```

### Run individual stages

```bash
make retrieve   # retrieval only
make generate   # generation only (requires retrieval results)
```

### Evaluate on the test set

```bash
make evaluate-smoke    # 10 images — quick sanity check (~6 minutes)
make evaluate          # 100 images — standard evaluation (~55 minutes)
make evaluate-full     # 200 images — more statistically robust

# Resume an interrupted evaluation run
python3 evaluate.py --n_samples 200 --resume evaluation/run_20260412_091233
```

### Run the API server locally

```bash
make api
# Server starts at http://localhost:8000
# Interactive docs at http://localhost:8000/docs

# Test with a real image
curl -X POST http://localhost:8000/generate \
  -F "image=@/path/to/chest_xray.jpg" | python3 -m json.tool
```

### Run tests

```bash
make test              # all 116 tests
make test-unit         # 84 unit tests (fast, no API calls, ~3 seconds)
make test-integration  # 32 integration tests (reads existing JSON files)
```

---

## API reference

### `POST /generate`

Accepts a chest X-ray image and returns a structured radiology report.

**Request:** `multipart/form-data` with field `image` (JPG or PNG, max 20MB)

**Response:**

```json
{
  "findings": "1. Cardiomediastinal contour: The cardiac silhouette appears enlarged...",
  "impression": "Findings consistent with pulmonary edema...",
  "caption": "Mild pulmonary edema. Pacemaker leads present.",
  "description": "Chest X-ray analysis identified: Mild pulmonary edema (0.40); ...",
  "model_used": "meta-llama/llama-4-scout-17b-16e-instruct",
  "n_retrieved": 5,
  "elapsed_seconds": 34.2,
  "generated_at": "2026-04-12T09:15:33.204"
}
```

### `GET /health`

Returns service status and pipeline readiness. Always fast.

---

## Technology stack

| Component | Technology | Notes |
|---|---|---|
| Embedding model | BioMedCLIP (ViT-B/16) | Microsoft, trained on 15M medical image-text pairs |
| Vector index | FAISS IndexFlatIP | Exact cosine similarity, 199,214 vectors |
| Primary LLM | Llama 4 Scout (Groq) | Free tier, vision-capable, fast inference |
| Fallback LLM | Gemini 2.0 Flash (Google) | Free tier, 1500 req/day |
| API framework | FastAPI + Uvicorn | REST service with auto-generated Swagger docs |
| Evaluation | BLEU, ROUGE, BERTScore | Standard NLP metrics for text generation |
| Dataset | MIMIC-CXR | 227,827 chest radiographs, BIDMC |

---

## Hardware constraints and engineering decisions

This project was built and runs entirely on a **MacBook Air (Apple M3, 8GB unified RAM)** — one of the most constrained environments for a production RAG pipeline of this scale. Every architectural decision was shaped by this constraint.

The 8GB limit creates an effective working budget of ~1.2–2.5GB after macOS baseline consumption. BioMedCLIP at fp16 precision consumes ~450MB. The FAISS index is memory-mapped from disk rather than fully loaded into RAM. All four subprocess scripts load their models, complete their work, and explicitly release memory before exiting — including `model.cpu()` before `del` to force Apple's unified memory system to release GPU allocations.

The result is a pipeline that peaks at ~2.15GB RAM usage and returns to ~2.1GB at rest, fitting comfortably within the hardware budget.

---

## Limitations

**Comparative reports excluded from evaluation.** ~44% of MIMIC-CXR reports are comparative ("unchanged from prior study") because radiologists routinely compare against historical imaging. The pipeline generates reports from a single image with no prior study access, making these ground truth reports invalid evaluation targets. This is a dataset characteristic rather than a pipeline limitation.

**Zero-shot captioning via phrase similarity.** BioMedCLIP is a CLIP-style model without a generative decoder. The captioning approach scores the image against a fixed vocabulary of 53 clinical phrases and uses the top-scoring phrases as a caption. This works well for common findings but may miss unusual pathologies not in the vocabulary.

**Single image per query.** The pipeline processes one image at a time. Multi-view studies (PA + lateral) are common in radiology — incorporating both views would improve report accuracy but would require changes to the embedding and retrieval stages.

---

BioMedCLIP is subject to Microsoft's model license. Groq and Gemini APIs are subject to their respective terms of service.
