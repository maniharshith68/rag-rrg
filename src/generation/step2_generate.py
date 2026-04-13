"""
src/generation/step2_generate.py

Subprocess script — do NOT run directly.
Invoked exclusively by generator.py via subprocess.run().

Responsibility:
    - Read caption_result.json produced by step1_caption.py
    - Read results.json produced by retriever.py (retrieved reports)
    - Read query image path from CLI argument
    - Encode image as base64 for Gemini multimodal input
    - Build a structured radiologist prompt:
          1. Image Caption
          2. Image Description
          3. Retrieved Similar Cases
          Task: Generate FINDINGS + IMPRESSION
    - Call Gemini Flash API (free tier)
    - Parse FINDINGS and IMPRESSION from the response
    - Save full structured result to src/generation/generation_result.json
    - Exit cleanly — OS reclaims memory

Usage (called by generator.py only):
    python3 step2_generate.py /absolute/path/to/image.jpg
"""

import sys
import json
import base64
import logging
import urllib.request
import urllib.error
import time
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# PATHS  (all derived from this file's location — no .env needed)
# ─────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).resolve().parent   # rag-rrg/src/generation/
SRC_DIR       = SCRIPT_DIR.parent                 # rag-rrg/src/
PROJECT_ROOT  = SRC_DIR.parent                    # rag-rrg/

CAPTION_FILE   = SCRIPT_DIR / "caption_result.json"
RETRIEVAL_FILE = SRC_DIR / "retrieval" / "results.json"
OUTPUT_FILE    = SCRIPT_DIR / "generation_result.json"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [step2_generate] %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "generation.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# GEMINI CONFIG
# ─────────────────────────────────────────────

# Gemini REST endpoint — no SDK needed, pure urllib
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Default model — can be overridden via GEMINI_MODEL in .env
DEFAULT_MODEL = "gemini-2.0-flash"

# Generation parameters
MAX_OUTPUT_TOKENS = 800    # enough for a thorough radiology report
TEMPERATURE       = 0.2    # low temperature = more deterministic, less hallucination
TOP_P             = 0.85

# Retry config — free tier occasionally returns 429 (rate limit)
MAX_RETRIES   = 3
RETRY_DELAY_S = 15   # seconds to wait before retrying on 429

# Context formatting config
MAX_CONTEXT_REPORTS = 3     # how many retrieved reports to include in the prompt
MAX_REPORT_CHARS    = 1500  # truncate each retrieved report to this length


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are an expert board-certified radiologist with 20 years of experience \
interpreting chest X-rays. You write formal, precise radiology reports.

Your reports follow this structure:
- FINDINGS: Systematic description of all visible structures. Always cover in order:
  1. Cardiomediastinal contour
  2. Lungs (each lobe, any opacities, consolidation, hyperinflation)
  3. Pleura (effusion, pneumothorax)
  4. Osseous structures and soft tissues
  5. Tubes, lines, and devices if present
- IMPRESSION: A concise 1–3 sentence clinical summary of the most important findings \
and their likely significance.

Rules you always follow:
- Use formal radiology language (e.g. "patchy airspace opacity", \
"blunting of the costophrenic angle", "mild cardiomegaly")
- Never fabricate findings — only report what the evidence supports
- If a finding is uncertain, qualify it ("cannot be excluded", "possible", "suggest")
- Base your reasoning on BOTH the image description AND the retrieved similar cases
- If the chest is normal, state "No acute cardiopulmonary abnormality" in IMPRESSION
- Always output FINDINGS: and IMPRESSION: as explicit section headers"""


def build_user_prompt(caption: str, description: str, context: str) -> str:
    return f"""Please generate a radiology report for this chest X-ray.

1. Image Caption: {caption}

2. Image Description: {description}

3. Retrieved Similar Cases (most visually similar X-rays from knowledge base):
{context}

Task: Generate a radiology report with the following sections:

FINDINGS:
[Your systematic findings here]

IMPRESSION:
[Your concise clinical summary here]

Guidelines:
- Be clinically precise
- Avoid hallucination — only report findings supported by the image description and similar cases
- Use formal radiology language throughout
- Base your reasoning on both the image description AND the retrieved similar cases above"""


# ─────────────────────────────────────────────
# .env READER
# ─────────────────────────────────────────────

def _read_env(key: str) -> str:
    """Read a single key from .env. Returns empty string if not found."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return ""
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(key + "="):
                _, _, val = line.partition("=")
                return val.strip()
    return ""


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def encode_image_base64(image_path: Path) -> tuple[str, str]:
    """
    Read image file and return (base64_string, mime_type).
    Supports JPG and PNG — the only formats in MIMIC-CXR.
    """
    suffix = image_path.suffix.lower()
    mime_map = {
        ".jpg"  : "image/jpeg",
        ".jpeg" : "image/jpeg",
        ".png"  : "image/png",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime_type


def format_retrieved_context(results: list) -> str:
    """
    Format top-K retrieved reports into a readable context block for the prompt.
    Deduplication by study_id was already done by step2_search.py — no need to repeat.
    """
    if not results:
        return "No similar cases retrieved."

    lines = []
    for r in results[:MAX_CONTEXT_REPORTS]:
        report = r.get("report", "").strip()
        if not report or report.lower() in ("nan", "none", ""):
            report = "[Report text not available for this case]"
        elif len(report) > MAX_REPORT_CHARS:
            report = report[:MAX_REPORT_CHARS] + "… [truncated]"

        lines.append(
            f"[Case {r['rank']}] "
            f"Similarity score: {r['score']:.4f} | "
            f"Study ID: {r['study_id']} | "
            f"Patient ID: {r['subject_id']}"
        )
        lines.append(report)
        lines.append("---")

    return "\n".join(lines)


def parse_findings_impression(raw_text: str) -> tuple[str, str]:
    """
    Extract FINDINGS and IMPRESSION sections from the raw LLM response.

    Handles variations like:
        FINDINGS:          FINDINGS\n       **FINDINGS:**
        IMPRESSION:        IMPRESSION\n     **IMPRESSION:**

    Returns (findings_text, impression_text).
    Both are stripped of their header labels and leading/trailing whitespace.
    If parsing fails, returns (full_raw_text, "") so nothing is silently lost.
    """
    import re

    # Normalise bold markdown headers that some models add
    text = re.sub(r"\*\*(FINDINGS|IMPRESSION)\*\*", r"\1", raw_text)

    findings   = ""
    impression = ""

    upper = text.upper()

    has_findings   = "FINDINGS"   in upper
    has_impression = "IMPRESSION" in upper

    if has_findings and has_impression:
        fi = upper.index("FINDINGS")
        ii = upper.index("IMPRESSION")

        if fi < ii:
            # Normal order: FINDINGS … IMPRESSION …
            findings_block   = text[fi:ii]
            impression_block = text[ii:]
        else:
            # Reversed order (rare but possible)
            impression_block = text[ii:fi]
            findings_block   = text[fi:]

        # Strip the section header itself ("FINDINGS:" or "FINDINGS\n")
        findings   = re.sub(r"(?i)^findings\s*:?\s*", "", findings_block).strip()
        impression = re.sub(r"(?i)^impression\s*:?\s*", "", impression_block).strip()

        # Remove any trailing section header that leaked in
        findings   = re.split(r"(?i)\nimpression", findings)[0].strip()
        impression = re.split(r"(?i)\nfindings",   impression)[0].strip()

    elif has_findings:
        # No IMPRESSION section — put everything in findings
        fi = upper.index("FINDINGS")
        findings = re.sub(r"(?i)^findings\s*:?\s*", "", text[fi:]).strip()
        log.warning("LLM response contained FINDINGS but no IMPRESSION section.")

    else:
        # No recognisable sections — return raw text as findings
        findings = raw_text.strip()
        log.warning(
            "Could not parse FINDINGS/IMPRESSION from LLM response. "
            "Storing full response in findings field. Raw output logged above."
        )

    return findings, impression


# ─────────────────────────────────────────────
# GEMINI API CALL
# ─────────────────────────────────────────────

def call_gemini(
    api_key   : str,
    model     : str,
    image_b64 : str,
    mime_type : str,
    prompt    : str,
) -> str:
    """
    Call Gemini via the REST API (no SDK — pure urllib).

    Uses the generateContent endpoint with:
      - system_instruction (radiologist persona)
      - user turn: [inline_data image] + [text prompt]

    Retries up to MAX_RETRIES times on HTTP 429 (rate limit).
    Raises RuntimeError on unrecoverable errors.

    Returns the raw text content from the model's response.
    """
    url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data"     : image_b64,
                        }
                    },
                    {
                        "text": prompt
                    },
                ],
            }
        ],
        "generationConfig": {
            "maxOutputTokens" : MAX_OUTPUT_TOKENS,
            "temperature"     : TEMPERATURE,
            "topP"            : TOP_P,
        },
        "safetySettings": [
            # Relax safety filters that sometimes block medical content
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        ],
    }

    body = json.dumps(payload).encode("utf-8")

    for attempt in range(1, MAX_RETRIES + 1):
        log.info(f"Gemini API call — attempt {attempt}/{MAX_RETRIES} (model: {model})")
        try:
            req = urllib.request.Request(
                url,
                data    = body,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")

            if e.code == 429:
                if attempt < MAX_RETRIES:
                    log.warning(
                        f"HTTP 429 rate limit hit. "
                        f"Waiting {RETRY_DELAY_S}s before retry {attempt + 1} …"
                    )
                    time.sleep(RETRY_DELAY_S)
                    continue
                else:
                    raise RuntimeError(
                        f"Gemini API rate limit (429) after {MAX_RETRIES} attempts. "
                        f"Wait a minute and retry. Body: {error_body[:300]}"
                    )

            elif e.code == 400:
                raise RuntimeError(
                    f"Gemini API bad request (400). "
                    f"Check API key, model name, and payload. Body: {error_body[:500]}"
                )

            elif e.code == 403:
                raise RuntimeError(
                    f"Gemini API forbidden (403). "
                    f"Check that your API key is valid and the model is available "
                    f"on the free tier. Body: {error_body[:300]}"
                )

            else:
                raise RuntimeError(
                    f"Gemini API HTTP {e.code}. Body: {error_body[:300]}"
                )

        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Network error calling Gemini API: {e.reason}. "
                f"Check your internet connection."
            )

        # ── Parse the response ──
        # Check for prompt feedback block (safety filter triggered before generation)
        if "promptFeedback" in raw:
            block_reason = raw["promptFeedback"].get("blockReason", "")
            if block_reason:
                raise RuntimeError(
                    f"Gemini blocked the prompt. Reason: {block_reason}. "
                    f"This can happen if safety filters flag medical content. "
                    f"The safetySettings in this script are already relaxed — "
                    f"try rephrasing the retrieved reports if this persists."
                )

        # Check candidates exist
        candidates = raw.get("candidates", [])
        if not candidates:
            raise RuntimeError(
                f"Gemini returned no candidates. Full response: {json.dumps(raw)[:500]}"
            )

        candidate = candidates[0]

        # Check finish reason
        finish_reason = candidate.get("finishReason", "")
        if finish_reason == "SAFETY":
            raise RuntimeError(
                "Gemini stopped generation due to safety filters (finishReason=SAFETY). "
                "Consider adjusting safetySettings or rephrasing the prompt."
            )
        if finish_reason == "MAX_TOKENS":
            log.warning(
                f"Gemini hit MAX_TOKENS ({MAX_OUTPUT_TOKENS}). "
                "Report may be truncated. Consider increasing MAX_OUTPUT_TOKENS."
            )

        # Extract text
        try:
            text = candidate["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected Gemini response structure — could not extract text. "
                f"Candidate: {json.dumps(candidate)[:400]}"
            )

        log.info(f"Gemini response received ({len(text)} chars, finishReason={finish_reason}).")
        return text

    # Should never reach here
    raise RuntimeError("call_gemini exhausted all retries without returning.")


# ─────────────────────────────────────────────
# GROQ API CALL  (free-tier fallback)
# ─────────────────────────────────────────────

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def call_groq(
    api_key   : str,
    image_b64 : str,
    mime_type : str,
    prompt    : str,
) -> str:
    """
    Call Groq API with Llama 4 Scout vision (free tier, no billing needed).

    Uses Groq's OpenAI-compatible /v1/chat/completions endpoint.
    Accepts base64-encoded image via the data URI format.
    Raises RuntimeError on any failure.

    Returns the raw text content from the model's response.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = json.dumps({
        "model"      : GROQ_MODEL,
        "max_tokens" : MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "messages"   : [
            {
                "role"   : "system",
                "content": SYSTEM_INSTRUCTION,
            },
            {
                "role"   : "user",
                "content": [
                    {
                        "type"     : "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data    = payload,
        headers = {
            "Content-Type"  : "application/json",
            "Authorization" : f"Bearer {api_key}",
            "User-Agent"    : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        method = "POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Groq API HTTP {e.code}: {error_body[:400]}"
        )
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error calling Groq API: {e.reason}")

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(
            f"Unexpected Groq response structure: {json.dumps(data)[:400]}"
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Validate CLI argument ──
    if len(sys.argv) < 2:
        log.error("Usage: python3 step2_generate.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1]).resolve()
    if not image_path.exists():
        log.error(f"Image not found: {image_path}")
        sys.exit(1)

    # ── Validate required input files ──
    for label, path in [
        ("caption_result.json", CAPTION_FILE),
        ("results.json",        RETRIEVAL_FILE),
    ]:
        if not path.exists():
            log.error(f"Required input not found — {label}: {path}")
            if label == "results.json":
                log.error("Run src/retrieval/retriever.py first to generate results.json.")
            if label == "caption_result.json":
                log.error("Run step1_caption.py first (generator.py does this automatically).")
            sys.exit(1)

    # ── Load caption result ──
    log.info(f"Loading {CAPTION_FILE.name} …")
    try:
        with open(CAPTION_FILE) as f:
            caption_data = json.load(f)
    except Exception as e:
        log.error(f"Failed to read caption_result.json: {e}")
        sys.exit(1)

    caption     = caption_data.get("caption", "").strip()
    description = caption_data.get("description", "").strip()

    if not caption:
        log.error("caption_result.json has an empty caption field. Re-run step1_caption.py.")
        sys.exit(1)

    log.info(f"Caption     : {caption[:120]}")
    log.info(f"Description : {description[:120]}")

    # ── Load retrieval results ──
    log.info(f"Loading {RETRIEVAL_FILE.name} …")
    try:
        with open(RETRIEVAL_FILE) as f:
            retrieval_data = json.load(f)
    except Exception as e:
        log.error(f"Failed to read results.json: {e}")
        sys.exit(1)

    if not isinstance(retrieval_data, list) or not retrieval_data:
        log.error("results.json is empty or malformed. Re-run retriever.py.")
        sys.exit(1)

    log.info(f"Retrieved cases : {len(retrieval_data)}")

    # ── Format context from retrieved reports ──
    context = format_retrieved_context(retrieval_data)
    log.info(f"Context block   : {len(context)} chars")

    # ── Build the radiologist prompt ──
    user_prompt = build_user_prompt(caption, description, context)
    log.info(f"Prompt built    : {len(user_prompt)} chars")

    # ── Encode image ──
    log.info(f"Encoding image  : {image_path.name}")
    try:
        image_b64, mime_type = encode_image_base64(image_path)
        log.info(f"Image encoded   : {len(image_b64)} base64 chars  mime={mime_type}")
    except Exception as e:
        log.error(f"Failed to encode image: {e}")
        sys.exit(1)

    # ── Read all API keys and config from .env ──
    gemini_key  = _read_env("GEMINI_API_KEY")
    groq_key    = _read_env("GROQ_API_KEY")
    gemini_model = _read_env("GEMINI_MODEL") or DEFAULT_MODEL
    backend     = _read_env("GENERATION_BACKEND") or "gemini"

    # Normalise placeholder values written by users who forgot to fill .env
    if gemini_key in ("your_gemini_api_key_here", ""):
        gemini_key = ""
    if groq_key in ("your_groq_api_key_here", ""):
        groq_key = ""

    # Must have at least one key configured
    if not gemini_key and not groq_key:
        log.error(
            "No API keys found in .env. At least one of the following is required:\n"
            "  GEMINI_API_KEY  — get free key at https://aistudio.google.com\n"
            "  GROQ_API_KEY    — get free key at https://console.groq.com"
        )
        sys.exit(1)

    # ── Try backends in order, with automatic fallback ──
    raw_output = None
    model_used = None
    errors     = []

    # ── Backend 1: Gemini ──
    # Skipped if: key absent, or GENERATION_BACKEND explicitly set to "groq"
    if gemini_key and backend != "groq":
        log.info(f"Trying Gemini ({gemini_model}) …")
        try:
            raw_output = call_gemini(
                api_key   = gemini_key,
                model     = gemini_model,
                image_b64 = image_b64,
                mime_type = mime_type,
                prompt    = user_prompt,
            )
            model_used = gemini_model
            log.info("Gemini succeeded ✓")
        except RuntimeError as e:
            log.warning(f"Gemini failed — {e}")
            errors.append(f"gemini: {e}")

    # ── Backend 2: Groq (automatic fallback or explicit) ──
    # Runs if: Gemini failed, key absent, or GENERATION_BACKEND set to "groq"
    if raw_output is None and groq_key:
        log.info(f"Trying Groq ({GROQ_MODEL}) …")
        try:
            raw_output = call_groq(
                api_key   = groq_key,
                image_b64 = image_b64,
                mime_type = mime_type,
                prompt    = user_prompt,
            )
            model_used = GROQ_MODEL
            log.info("Groq succeeded ✓")
        except RuntimeError as e:
            log.warning(f"Groq failed — {e}")
            errors.append(f"groq: {e}")

    # ── All backends exhausted ──
    if raw_output is None:
        log.error(
            "All backends failed. Errors:\n"
            + "\n".join(f"  {e}" for e in errors)
            + "\n\nTroubleshooting:\n"
            "  Gemini 429 → wait 60s, or regenerate key at aistudio.google.com/app/apikey\n"
            "  Groq error  → check key at console.groq.com, model may have changed\n"
            "  Both fail   → check internet connection and logs/generation.log"
        )
        sys.exit(1)

    log.info(f"Model used      : {model_used}")
    log.info("Raw LLM output:")
    log.info("-" * 40)
    log.info(raw_output[:800])
    log.info("-" * 40)

    # ── Parse FINDINGS and IMPRESSION ──
    findings, impression = parse_findings_impression(raw_output)

    log.info(f"Findings parsed   : {len(findings)} chars")
    log.info(f"Impression parsed : {len(impression)} chars")

    if not findings:
        log.warning(
            "Findings field is empty after parsing. "
            "The raw output will be stored in raw_llm_output for inspection."
        )

    # ── Build output dict ──
    result = {
        "image_path"        : str(image_path),
        "generated_at"      : datetime.now().isoformat(),
        "model_used"        : model_used,
        "caption"           : caption,
        "description"       : description,
        "retrieved_cases"   : len(retrieval_data),
        "context_used"      : context,
        "findings"          : findings,
        "impression"        : impression,
        "raw_llm_output"    : raw_output,
        "top_phrases"       : caption_data.get("top_phrases", []),
        "backends_attempted": errors,
    }

    # ── Save to generation_result.json ──
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"Saved → {OUTPUT_FILE}")
    except Exception as e:
        log.error(f"Failed to save generation_result.json: {e}")
        sys.exit(1)

    log.info("step2_generate.py done. Exiting.")
    sys.exit(0)


if __name__ == "__main__":
    main()
