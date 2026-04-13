"""
embedding.py
------------
Generates multimodal embeddings (image + text) using BioMedCLIP for the RAG pipeline.
Both image and text are embedded into the SAME shared vector space (512-dim),
enabling cross-modal retrieval: query image → retrieve similar report texts via FAISS.

Model     : microsoft/BiomedCLIP-PubMedBERT_256-vit_large_patch16_224
Hardware  : Optimized for Apple M3 (MPS backend) with 8GB unified RAM
Strategy  : Batched processing + incremental checkpointing to stay within ~70% RAM usage

Output files (saved to rag-rrg/embedding/):
    image_embeddings.npy  — shape (N, 512), float32
    text_embeddings.npy   — shape (N, 512), float32
    metadata.csv          — subject_id, study_id, image_path (row-aligned with embeddings)
    checkpoint.txt        — index of last successfully saved batch (for resume)
"""

import os
import gc
import csv
import time
import numpy as np
import pandas as pd
import torch
import psutil
from PIL import Image
from pathlib import Path
from open_clip import create_model_from_pretrained, get_tokenizer

# ─────────────────────────────────────────────
# 0. CONFIGURATION — adjust paths and limits here
# ─────────────────────────────────────────────

# Root of your project (one level above this script's embedding/ folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # rag-rrg/

# Base directory for resolving image paths from the CSV
# image_path in CSV starts with "files/..." which lives under rag-rrg/data/
IMAGE_BASE_DIR = PROJECT_ROOT / "data"

# Input knowledge base CSV
KNOWLEDGE_BASE_CSV = PROJECT_ROOT / "data" / "knowledge_base.csv"

# Output directory for all embedding artifacts
EMBEDDING_DIR = PROJECT_ROOT / "src" / "embedding"
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

# Output file paths
IMAGE_EMB_PATH  = EMBEDDING_DIR / "image_embeddings.npy"
TEXT_EMB_PATH   = EMBEDDING_DIR / "text_embeddings.npy"
METADATA_PATH   = EMBEDDING_DIR / "metadata.csv"
CHECKPOINT_PATH = EMBEDDING_DIR / "checkpoint.txt"

# Batch size — keep small to respect 8GB RAM limit
# 4 is safe; increase to 8 only if RAM usage stays under 70% during a test run
BATCH_SIZE = 4

# RAM safety threshold: pause/warn if system RAM usage exceeds this percentage
RAM_LIMIT_PERCENT = 70.0

# BioMedCLIP model identifier on HuggingFace / open_clip
BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# Image size expected by BioMedCLIP's ViT-Large backbone
IMAGE_SIZE = 224


# ─────────────────────────────────────────────
# 1. DEVICE SELECTION
# Prefer MPS (Apple Silicon GPU) for speed,
# fall back to CPU if MPS is unavailable.
# ─────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        print("[Device] MPS (Apple Silicon GPU) detected — using MPS for acceleration.")
        return torch.device("mps")
    else:
        print("[Device] MPS not available — falling back to CPU.")
        return torch.device("cpu")


# ─────────────────────────────────────────────
# 2. RAM MONITOR
# Checks current system RAM usage and warns
# if it exceeds the configured safety threshold.
# ─────────────────────────────────────────────

def check_ram(label=""):
    ram = psutil.virtual_memory()
    used_pct = ram.percent
    used_gb  = ram.used / (1024 ** 3)
    total_gb = ram.total / (1024 ** 3)
    print(f"[RAM {label}] {used_gb:.2f}GB / {total_gb:.2f}GB used ({used_pct:.1f}%)")

    if used_pct > RAM_LIMIT_PERCENT:
        print(f"[RAM WARNING] Usage {used_pct:.1f}% exceeds limit {RAM_LIMIT_PERCENT}%!")
        print("[RAM WARNING] Forcing garbage collection to free memory...")
        gc.collect()
        # On MPS, clearing cache helps reclaim GPU-mapped unified memory
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        time.sleep(1)  # Brief pause to let OS reclaim memory


# ─────────────────────────────────────────────
# 3. CHECKPOINT UTILITIES
# Saves and loads the index of the last
# successfully processed batch, so interrupted
# runs can resume without re-embedding from scratch.
# ─────────────────────────────────────────────

def load_checkpoint():
    """Returns the index of the next batch to process (0 if no checkpoint)."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            val = f.read().strip()
            if val.isdigit():
                batch_idx = int(val)
                print(f"[Checkpoint] Resuming from batch index {batch_idx}.")
                return batch_idx
    print("[Checkpoint] No checkpoint found — starting from batch 0.")
    return 0


def save_checkpoint(batch_idx):
    """Persists the index of the last successfully saved batch."""
    with open(CHECKPOINT_PATH, "w") as f:
        f.write(str(batch_idx))


# ─────────────────────────────────────────────
# 4. INCREMENTAL NUMPY SAVE UTILITIES
# Instead of accumulating all embeddings in RAM
# and saving once at the end (which risks OOM),
# we append each batch's embeddings to disk
# using numpy memory-mapped files or np.save
# with an accumulation list flushed periodically.
# ─────────────────────────────────────────────

def append_embeddings_to_disk(path, new_vectors):
    """
    Appends a (batch_size, 512) numpy array to an existing .npy file on disk,
    or creates it if it doesn't exist yet.
    Strategy: load existing → concatenate → save back.
    Safe for our batch sizes (512-dim × 4 rows = tiny per batch).
    """
    new_vectors = new_vectors.astype(np.float32)
    if path.exists():
        existing = np.load(str(path))
        combined = np.concatenate([existing, new_vectors], axis=0)
    else:
        combined = new_vectors
    np.save(str(path), combined)


def append_metadata_to_disk(rows):
    """
    Appends a list of metadata dicts to metadata.csv.
    Creates the file with header on first call.
    rows: list of dicts with keys subject_id, study_id, image_path
    """
    file_exists = METADATA_PATH.exists()
    with open(METADATA_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "study_id", "image_path"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────
# 5. IMAGE PREPROCESSING
# BioMedCLIP expects 224×224 RGB images.
# MIMIC-CXR images may be grayscale JPEGs —
# convert to RGB so the 3-channel ViT works correctly.
# ─────────────────────────────────────────────

def load_and_preprocess_image(image_path_str, preprocess_fn):
    """
    Loads a single image from disk, converts to RGB (handles grayscale MIMIC-CXR),
    applies BioMedCLIP's preprocessing transform, and returns a (1, C, H, W) tensor.
    Returns None if the image file is missing or corrupt.
    """
    full_path = IMAGE_BASE_DIR / image_path_str  # image_path in CSV is relative to project root
    if not full_path.exists():
        print(f"  [Warning] Image not found, skipping: {full_path}")
        return None
    try:
        img = Image.open(full_path).convert("RGB")  # force RGB even if grayscale
        tensor = preprocess_fn(img).unsqueeze(0)    # shape: (1, 3, 224, 224)
        return tensor
    except Exception as e:
        print(f"  [Warning] Failed to load image {full_path}: {e}")
        return None


# ─────────────────────────────────────────────
# 6. TEXT PREPROCESSING
# BioMedCLIP's text encoder (PubMedBERT) accepts
# tokenized text with a max context of 256 tokens.
# Reports longer than 256 tokens are truncated.
# ─────────────────────────────────────────────

def tokenize_texts(texts, tokenizer, device):
    """
    Tokenizes a list of report strings using BioMedCLIP's tokenizer.
    Returns token tensors moved to the target device.
    """
    # open_clip tokenizer returns a tensor of shape (batch, context_length)
    tokens = tokenizer(texts)  # handles truncation to 256 tokens internally
    return tokens.to(device)


# ─────────────────────────────────────────────
# 7. EMBEDDING EXTRACTION
# BioMedCLIP's encode_image and encode_text
# both project into the same 512-dim space.
# Normalization (L2) is applied so cosine similarity
# equals dot product — which FAISS IndexFlatIP uses.
# ─────────────────────────────────────────────

@torch.no_grad()  # disable gradient tracking to save memory during inference
def embed_batch(model, image_tensors, text_tokens, device):
    """
    Embeds a batch of (image_tensor, text_tokens) pairs.
    Both outputs are L2-normalized 512-dim vectors in the same shared space.

    Args:
        model           : BioMedCLIP model
        image_tensors   : (B, 3, 224, 224) tensor on device
        text_tokens     : (B, context_length) token tensor on device

    Returns:
        image_embs : (B, 512) numpy float32 array
        text_embs  : (B, 512) numpy float32 array
    """
    # Encode image through ViT-Large backbone → 512-dim projection
    image_embs = model.encode_image(image_tensors)

    # Encode text through PubMedBERT backbone → 512-dim projection
    text_embs  = model.encode_text(text_tokens)

    # L2 normalize so vectors lie on unit hypersphere
    # This makes cosine similarity = dot product (required for FAISS IndexFlatIP)
    image_embs = torch.nn.functional.normalize(image_embs, dim=-1)
    text_embs  = torch.nn.functional.normalize(text_embs,  dim=-1)

    # Move to CPU and convert to numpy before returning
    # (keeps GPU/MPS memory free for next batch)
    image_embs_np = image_embs.cpu().float().numpy()
    text_embs_np  = text_embs.cpu().float().numpy()

    return image_embs_np, text_embs_np


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BioMedCLIP Multimodal Embedding Pipeline")
    print("=" * 60)

    # ── 8a. Initial RAM check before loading anything ──
    check_ram("startup")

    # ── 8b. Select compute device ──
    device = get_device()

    # ── 8c. Load BioMedCLIP model and preprocessing ──
    print(f"\n[Model] Loading BioMedCLIP from HuggingFace hub...")
    print("[Model] This downloads ~1.7GB on first run and caches locally.")

    model, preprocess = create_model_from_pretrained(BIOMEDCLIP_MODEL)
    tokenizer = get_tokenizer(BIOMEDCLIP_MODEL)

    # Move model to MPS/CPU and set to eval mode (disables dropout, etc.)
    model = model.to(device)
    model.eval()

    print("[Model] BioMedCLIP loaded successfully.")
    check_ram("after model load")

    # ── 8d. Load knowledge base CSV ──
    print(f"\n[Data] Loading knowledge base from: {KNOWLEDGE_BASE_CSV}")
    df = pd.read_csv(KNOWLEDGE_BASE_CSV)

    # Drop rows with missing image_path or report — can't embed either without both
    df = df.dropna(subset=["image_path", "report"]).reset_index(drop=True)
    total_rows = len(df)
    print(f"[Data] Total valid rows to embed: {total_rows}")

    # ── 8e. Load checkpoint to support resume ──
    start_batch = load_checkpoint()
    start_row   = start_batch * BATCH_SIZE

    if start_row >= total_rows:
        print("[Info] All rows already embedded. Nothing to do.")
        return

    print(f"[Info] Starting from row {start_row} (batch {start_batch}).")

    # ── 8f. Batch processing loop ──
    # Processes BATCH_SIZE rows at a time to stay within RAM limits.
    # Each batch: preprocess images + tokenize texts → embed → save to disk → checkpoint.

    batch_num = start_batch

    for batch_start in range(start_row, total_rows, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_rows)
        batch_df  = df.iloc[batch_start:batch_end]

        print(f"\n[Batch {batch_num}] Rows {batch_start}–{batch_end-1} of {total_rows-1}")

        # ── Preprocess images in this batch ──
        image_tensors_list = []
        valid_indices      = []  # track which rows had valid images

        for local_idx, (_, row) in enumerate(batch_df.iterrows()):
            tensor = load_and_preprocess_image(row["image_path"], preprocess)
            if tensor is not None:
                image_tensors_list.append(tensor)
                valid_indices.append(local_idx)

        if len(image_tensors_list) == 0:
            print(f"  [Batch {batch_num}] All images missing — skipping batch.")
            save_checkpoint(batch_num + 1)
            batch_num += 1
            continue

        # Stack individual (1, 3, 224, 224) tensors → (B, 3, 224, 224)
        image_batch = torch.cat(image_tensors_list, dim=0).to(device)

        # ── Tokenize report texts for valid rows only ──
        valid_batch_df = batch_df.iloc[valid_indices]
        reports        = valid_batch_df["report"].tolist()
        text_tokens    = tokenize_texts(reports, tokenizer, device)

        # ── Embed both modalities in shared 512-dim space ──
        print(f"  [Embed] Embedding {len(valid_indices)} image-text pairs...")
        image_embs, text_embs = embed_batch(model, image_batch, text_tokens, device)

        # ── Verify both embeddings are in the same space (sanity check) ──
        # Cosine similarity between paired image and text should be > 0 for a good model
        cos_sims = (image_embs * text_embs).sum(axis=1)
        print(f"  [Sanity] Avg image-text cosine similarity: {cos_sims.mean():.4f} "
              f"(expect > 0 for aligned pairs)")

        # ── Save embeddings and metadata incrementally to disk ──
        append_embeddings_to_disk(IMAGE_EMB_PATH, image_embs)
        append_embeddings_to_disk(TEXT_EMB_PATH,  text_embs)

        # Save metadata rows aligned with embedding rows
        metadata_rows = [
            {
                "subject_id": row["subject_id"],
                "study_id":   row["study_id"],
                "image_path": row["image_path"],
            }
            for _, row in valid_batch_df.iterrows()
        ]
        append_metadata_to_disk(metadata_rows)

        print(f"  [Saved] Embeddings and metadata written to {EMBEDDING_DIR}")

        # ── Save checkpoint so we can resume if interrupted ──
        save_checkpoint(batch_num + 1)

        # ── Free batch tensors from memory explicitly ──
        del image_batch, text_tokens, image_embs, text_embs
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # ── RAM check after every batch ──
        check_ram(f"after batch {batch_num}")

        batch_num += 1

    # ── 8g. Final verification ──
    print("\n" + "=" * 60)
    print("[Done] Embedding pipeline complete.")

    image_embs_final = np.load(str(IMAGE_EMB_PATH))
    text_embs_final  = np.load(str(TEXT_EMB_PATH))
    metadata_final   = pd.read_csv(METADATA_PATH)

    print(f"[Verify] image_embeddings.npy shape : {image_embs_final.shape}")
    print(f"[Verify] text_embeddings.npy  shape : {text_embs_final.shape}")
    print(f"[Verify] metadata.csv rows           : {len(metadata_final)}")

    # All three should have the same number of rows
    assert image_embs_final.shape[0] == text_embs_final.shape[0] == len(metadata_final), \
        "[Error] Row count mismatch between embeddings and metadata!"

    print("[Verify] Row alignment check passed ✓")
    print(f"[Verify] Embedding dimension         : {image_embs_final.shape[1]} "
          f"(512 expected for BioMedCLIP)")
    print("=" * 60)
    print(f"\nNext step: Build FAISS index over text_embeddings.npy")
    print(f"           Query with image_embeddings at retrieval time.")


if __name__ == "__main__":
    main()
