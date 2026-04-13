"""
src/retrieval/step1_embed.py

Subprocess script — do NOT run directly.
Invoked exclusively by retriever.py via subprocess.run().

Responsibility:
    - Accept a single image path as CLI argument
    - Load BioMedCLIP fp16 on MPS (Apple Silicon)
    - Embed the image and L2-normalise the vector
    - Save result to src/retrieval/query_vec.npy
    - Explicit cleanup then exit — OS reclaims all ~450MB of model memory

Usage (called by retriever.py only):
    python3 step1_embed.py /absolute/path/to/image.jpg
"""

import sys
import gc
import logging
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS  (all derived from this file's location — no .env needed)
# ─────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent   # rag-rrg/src/retrieval/
SRC_DIR      = SCRIPT_DIR.parent                 # rag-rrg/src/
PROJECT_ROOT = SRC_DIR.parent                    # rag-rrg/

QUERY_VEC_FILE = SCRIPT_DIR / "query_vec.npy"   # output written here

# ─────────────────────────────────────────────
# MODEL CONFIG  (fixed constants — not environment-specific)
# ─────────────────────────────────────────────

BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [step1_embed] %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "retrieval.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HuggingFace token  (the ONE thing read from .env)
# ─────────────────────────────────────────────

def _load_hf_token() -> str:
    """Read only HUGGINGFACE_TOKEN from .env. Returns empty string if absent."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return ""
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("HUGGINGFACE_TOKEN"):
                _, _, val = line.partition("=")
                token = val.strip()
                return token if token != "hf_your_token_here" else ""
    return ""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Validate CLI argument ──
    if len(sys.argv) < 2:
        log.error("Usage: python3 step1_embed.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        log.error(f"Image not found: {image_path}")
        sys.exit(1)

    log.info(f"Query image  : {image_path}")
    log.info(f"Model        : {BIOMEDCLIP_MODEL}")
    log.info(f"Output       : {QUERY_VEC_FILE}")

    # ── HuggingFace login (required for gated BioMedCLIP) ──
    hf_token = _load_hf_token()
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            log.info("HuggingFace login successful.")
        except Exception as e:
            log.warning(f"HuggingFace login failed (may still work if cached): {e}")
    else:
        log.warning("No HuggingFace token found — assuming cached credentials.")

    # ── Detect device ──
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    # ── Load BioMedCLIP fp16 ──
    log.info("Loading BioMedCLIP fp16 …")
    try:
        import open_clip
        model, preprocess = open_clip.create_model_from_pretrained(
            BIOMEDCLIP_MODEL, precision="fp16"
        )
        model = model.to(device)
        model.eval()
        log.info("Model loaded.")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        sys.exit(1)

    # ── Load and preprocess image ──
    log.info("Preprocessing image …")
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)   # (1, C, H, W)
        if device.type != "cpu":
            img_tensor = img_tensor.half()                      # fp16 on MPS / CUDA
    except Exception as e:
        log.error(f"Failed to preprocess image: {e}")
        sys.exit(1)

    # ── Embed and L2-normalise ──
    log.info("Embedding image …")
    try:
        with torch.no_grad():
            image_features = model.encode_image(img_tensor)                        # (1, 512)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        query_vec = image_features.cpu().float().numpy()                            # (1, 512) float32
        log.info(f"Embedding shape: {query_vec.shape}  dtype: {query_vec.dtype}")
    except Exception as e:
        log.error(f"Embedding failed: {e}")
        sys.exit(1)

    # ── Save query vector ──
    try:
        np.save(QUERY_VEC_FILE, query_vec)
        log.info(f"Saved → {QUERY_VEC_FILE}")
    except Exception as e:
        log.error(f"Failed to save query_vec.npy: {e}")
        sys.exit(1)

    # ── Explicit cleanup — force MPS to release unified memory before process exits ──
    log.info("Releasing model memory …")
    model.cpu()
    del model, img_tensor, image_features
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass  # Not on MPS — safe to ignore

    log.info("step1_embed.py done. Exiting — OS reclaims ~450MB.")
    sys.exit(0)


if __name__ == "__main__":
    main()
