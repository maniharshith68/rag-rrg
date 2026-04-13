"""
src/generation/step1_caption.py

Subprocess script — do NOT run directly.
Invoked exclusively by generator.py via subprocess.run().

Responsibility:
    - Accept a single image path as CLI argument
    - Load BioMedCLIP fp16 on MPS (Apple Silicon)
    - Compute zero-shot image-text similarities against a clinical phrase vocabulary
    - Derive a caption (top-3 phrases joined) and a structured description
    - Save result to src/generation/caption_result.json
    - Explicit cleanup then exit — OS reclaims all ~450MB of model memory

Usage (called by generator.py only):
    python3 step1_caption.py /absolute/path/to/image.jpg
"""

import sys
import gc
import json
import logging
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# PATHS  (all derived from this file's location — no .env needed)
# ─────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent   # rag-rrg/src/generation/
SRC_DIR      = SCRIPT_DIR.parent                 # rag-rrg/src/
PROJECT_ROOT = SRC_DIR.parent                    # rag-rrg/

OUTPUT_FILE = SCRIPT_DIR / "caption_result.json"

# ─────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────

BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# Confidence threshold — phrases below this are excluded from the description
CONFIDENCE_THRESHOLD = 0.20

# Top-N phrases to include in the structured description
TOP_N_DESCRIPTION = 6

# ─────────────────────────────────────────────
# CLINICAL PHRASE VOCABULARY
# These are the candidate labels used for zero-shot classification.
# BioMedCLIP scores each phrase against the image; the highest-scoring
# phrases become the caption and description.
# ─────────────────────────────────────────────

CAPTION_PHRASES = [
    # Normal
    "No acute cardiopulmonary abnormality",
    "Normal chest X-ray",
    "No pneumothorax or pleural effusion",

    # Cardiac
    "Cardiomegaly",
    "Mild cardiomegaly",
    "Moderate cardiomegaly",
    "Severe cardiomegaly",
    "Enlarged cardiac silhouette",

    # Pleural
    "Bilateral pleural effusions",
    "Left pleural effusion",
    "Right pleural effusion",
    "Small pleural effusion",
    "Moderate pleural effusion",
    "Large pleural effusion",

    # Pulmonary vascular / edema
    "Pulmonary edema",
    "Mild pulmonary edema",
    "Moderate pulmonary edema",
    "Severe pulmonary edema",
    "Vascular congestion",
    "Interstitial edema",
    "Mild interstitial prominence",

    # Consolidation / infection
    "Pneumonia",
    "Left lower lobe pneumonia",
    "Right lower lobe pneumonia",
    "Bilateral pneumonia",
    "Airspace opacity",
    "Bilateral infiltrates consistent with pneumonia",
    "Lobar consolidation",
    "Patchy consolidation",

    # Atelectasis
    "Atelectasis",
    "Bibasilar atelectasis",
    "Left basilar atelectasis",
    "Right basilar atelectasis",
    "Linear atelectasis",
    "Subsegmental atelectasis",

    # Airways / lung
    "Lung hyperinflation consistent with COPD",
    "Flattened hemidiaphragms",
    "Hyperaeration",

    # Mediastinum
    "Mediastinal widening",
    "Prominent pulmonary vasculature",

    # Pneumothorax
    "Pneumothorax",
    "Left pneumothorax",
    "Right pneumothorax",
    "Small pneumothorax",

    # Osseous
    "Osseous structures are intact",
    "Rib fractures",
    "Degenerative changes of the spine",

    # Devices / lines
    "Endotracheal tube in appropriate position",
    "Central venous catheter",
    "Nasogastric tube",
    "Pacemaker leads present",
    "Implantable cardioverter defibrillator",
    "Chest tube present",
]


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [step1_caption] %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "generation.log"),
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
        log.error("Usage: python3 step1_caption.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        log.error(f"Image not found: {image_path}")
        sys.exit(1)

    log.info(f"Query image  : {image_path}")
    log.info(f"Model        : {BIOMEDCLIP_MODEL}")
    log.info(f"Phrases      : {len(CAPTION_PHRASES)}")
    log.info(f"Output       : {OUTPUT_FILE}")

    # ── HuggingFace login ──
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
        tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL)
        log.info("Model loaded.")
    except Exception as e:
        log.error(f"Failed to load BioMedCLIP: {e}")
        sys.exit(1)

    # ── Load and preprocess image ──
    log.info("Preprocessing image …")
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        if device.type != "cpu":
            img_tensor = img_tensor.half()
    except Exception as e:
        log.error(f"Failed to preprocess image: {e}")
        sys.exit(1)

    # ── Tokenize clinical phrases ──
    log.info("Tokenizing clinical phrases …")
    try:
        text_tokens = tokenizer(CAPTION_PHRASES).to(device)
    except Exception as e:
        log.error(f"Failed to tokenize phrases: {e}")
        sys.exit(1)

    # ── Compute image–text similarities ──
    log.info("Computing similarities …")
    try:
        with torch.no_grad():
            img_feat  = model.encode_image(img_tensor)
            txt_feat  = model.encode_text(text_tokens)
            img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ txt_feat.T).squeeze(0).cpu().float().numpy()
        log.info(f"Similarity range: min={sims.min():.4f}  max={sims.max():.4f}")
    except Exception as e:
        log.error(f"Similarity computation failed: {e}")
        sys.exit(1)

    # ── Rank phrases by similarity score ──
    ranked_idx   = np.argsort(sims)[::-1]
    ranked_pairs = [(CAPTION_PHRASES[i], float(sims[i])) for i in ranked_idx]

    # ── Caption: top-3 highest-scoring phrases joined as a sentence ──
    top3_phrases = [phrase for phrase, _ in ranked_pairs[:3]]
    caption = ". ".join(top3_phrases) + "."

    # ── Description: structured paragraph from phrases above threshold ──
    above_threshold = [
        (phrase, score)
        for phrase, score in ranked_pairs
        if score >= CONFIDENCE_THRESHOLD
    ][:TOP_N_DESCRIPTION]

    if above_threshold:
        parts = [f"{phrase} (confidence: {score:.2f})" for phrase, score in above_threshold]
        description = (
            "Chest X-ray analysis identified the following findings: "
            + "; ".join(parts) + "."
        )
    else:
        # Fallback — always produce something even if no phrase clears threshold
        top1_phrase, top1_score = ranked_pairs[0]
        description = (
            f"Chest X-ray. Top finding: {top1_phrase} (confidence: {top1_score:.2f}). "
            f"No findings exceeded the confidence threshold of {CONFIDENCE_THRESHOLD}."
        )
        log.warning(
            f"No phrases exceeded threshold {CONFIDENCE_THRESHOLD}. "
            f"Max score was {top1_score:.4f} ({top1_phrase}). "
            "Consider lowering CONFIDENCE_THRESHOLD if this recurs."
        )

    # ── Log top-5 for inspection ──
    log.info("Top-5 phrases:")
    for phrase, score in ranked_pairs[:5]:
        log.info(f"  {score:.4f}  {phrase}")

    # ── Build output dict ──
    result = {
        "image_path"   : str(image_path),
        "caption"      : caption,
        "description"  : description,
        "top_phrases"  : [
            {"phrase": phrase, "score": score}
            for phrase, score in ranked_pairs[:10]
        ],
        "model"        : BIOMEDCLIP_MODEL,
    }

    # ── Save to caption_result.json ──
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"Saved → {OUTPUT_FILE}")
    except Exception as e:
        log.error(f"Failed to save caption_result.json: {e}")
        sys.exit(1)

    # ── Explicit cleanup — force MPS to release unified memory ──
    log.info("Releasing model memory …")
    model.cpu()
    del model, img_tensor, img_feat, txt_feat, text_tokens
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass

    log.info("step1_caption.py done. Exiting — OS reclaims ~450MB.")
    sys.exit(0)


if __name__ == "__main__":
    main()
