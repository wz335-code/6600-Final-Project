import json
import random
from typing import List, Dict
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TORCH"] = "1"

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from df_cleaning import clean_text

# Sentence segmentation
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

# DeepSeek uses an OpenAI-compatible API. Set DEEPSEEK_API_KEY in your environment.
from openai import OpenAI

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise EnvironmentError(
        "Missing DEEPSEEK_API_KEY. Set it like: export DEEPSEEK_API_KEY='...'(or in your shell/IDE env)."
    )

# Use /v1 for OpenAI-SDK compatibility.
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# --------------------------------
# Perf knobs
# --------------------------------
LLM_MAX_WORKERS = 16          # thread pool for OpenAI calls
LLM_GROUP_SIZE = 20           # how many sentences to rewrite per OpenAI call
LLM_MAX_RETRIES_ON_PARSE = 2  # local parse retries per response

def _chunks(idxs: List[int], size: int) -> List[List[int]]:
    return [idxs[i:i + size] for i in range(0, len(idxs), size)]

def _extract_json_array(text: str):
    """Best-effort extraction of a JSON array from a model response."""
    if not text:
        return None
    t = text.strip()
    # Fast path
    if t.startswith("[") and t.endswith("]"):
        try:
            return json.loads(t)
        except Exception:
            pass
    # Try to find the first [...] span
    l = t.find("[")
    r = t.rfind("]")
    if l != -1 and r != -1 and r > l:
        span = t[l:r + 1]
        try:
            return json.loads(span)
        except Exception:
            return None
    return None

# ========================================================
# 0. CONFIG
# ========================================================
def call_llm(prompt: str, max_tokens: int = 1024) -> str:
    """Thin wrapper around DeepSeek (OpenAI-compatible) Chat Completions API."""
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()

def call_llm_batch(prompts: List[str], max_tokens: int = 1024, max_workers: int = 8) -> List[str]:
    """
    Issue multiple LLM calls in parallel using a thread pool.
    This does not use DeepSeek's async batch API; it parallelizes regular synchronous calls to `call_llm`.
    """
    if not prompts:
        return []

    results: List[str] = [""] * len(prompts)

    def _worker(idx: int, prompt: str):
        try:
            results[idx] = call_llm(prompt, max_tokens=max_tokens)
        except Exception as e:
            # On any failure, fall back to empty string so downstream logic
            # can treat it as a rejected candidate.
            print(f"LLM call failed for idx {idx}: {e}")
            results[idx] = ""

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, i, p) for i, p in enumerate(prompts)]
        for _ in as_completed(futures):
            # We don't need per-future results here; results[] is filled in place.
            pass

    return results


# All sources will be converted into sentence rows.
# Put both sentence-level and document-level CSVs here; document-level ones will be sentence-split.
classification_files = [
    "cleaned_readme.csv",
    "cleaned_elg.csv",
]

OUTPUT_JSONL = "universalcefr_pseudopairs_8.jsonl"

# CEFR ordering
CEFR = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR2IDX = {c: i for i, c in enumerate(CEFR)}

LEVEL_DESCRIPTORS = {
    "A2": "elementary learners who can understand simple sentences about everyday topics.",
    "B1": "lower-intermediate learners who can understand straightforward connected text on familiar topics.",
    "B2": "upper-intermediate learners who can follow the main ideas of complex text on both concrete and abstract topics.",
    "C1": "advanced learners who can understand demanding, longer texts and recognise implicit meaning.",
    "C2": "near-native learners who can understand virtually everything read or heard.",
}

# Input prefix builder for Flan-T5 training format.
# Your fine-tuning examples use the simple prefix style:
#   "simplify to A2: <text>"
def make_input_prefix(target_cefr: str) -> str:
    target_cefr = (target_cefr or "").strip()
    return f"simplify to {target_cefr}: "

# Prefer Apple Silicon MPS when available (local Mac runs), otherwise fall back to CUDA, then CPU.
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Classifier (from TSAR paper families)
CEFR_MODEL_NAME = "AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr"


def to_sentences(text: str) -> List[str]:
    """Split a document into clean sentences for sentence-level fine-tuning."""
    if not text:
        return []
    # normalize whitespace
    t = str(text).replace("\n", " ").replace("\r", " ")
    t = " ".join(t.split())
    sents = [s.strip() for s in sent_tokenize(t) if s.strip()]
    return sents

# ========================================================
# 1. LOAD DATA
# ========================================================

def load_universal_data() -> pd.DataFrame:
    """Load all sources and return a single sentence-level dataframe with columns: text, cefr."""
    all_rows: List[Dict[str, str]] = []

    for f in classification_files:
        df = pd.read_csv(f)
        df["cefr"] = df["cefr_level"]
        df = df.dropna(subset=["cefr"])
        df = df[["text", "cefr"]]

        # Clean as a document first, then sentence-split so we always output sentence rows.
        df["text"] = df["text"].apply(lambda t: clean_text(t, sentence_tokenize=False))

        for _, r in df.iterrows():
            cefr = r["cefr"]
            for s in to_sentences(r["text"]):
                all_rows.append({"text": s, "cefr": cefr})

    out_df = pd.DataFrame(all_rows)
    return out_df

# ========================================================
# 2. CEFR EVALUATOR
# ========================================================

def load_cefr_model():
    tokenizer = AutoTokenizer.from_pretrained(CEFR_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(CEFR_MODEL_NAME)
    if DEVICE == "cuda":
        model = model.half()
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def predict_cefr(text, tok, model):
    with torch.no_grad():
        inp = tok(text, truncation=True, return_tensors="pt").to(DEVICE)
        logits = model(**inp).logits
        pr = torch.softmax(logits, dim=-1)[0]
        idx = int(pr.argmax().item())
        return CEFR[idx], float(pr[idx].item())

# --------------------------------------------------------
# Batched CEFR prediction helper
from typing import List, Tuple
def predict_cefr_batch(
    texts: List[str],
    tok,
    model,
    batch_size: int = 64,
) -> Tuple[List[str], List[float]]:
    """
    Batched CEFR prediction for a list of texts.

    Returns:
        labels: list of CEFR labels (e.g. ["B1", "A2", ...])
        scores: list of probabilities for the predicted label.
    """
    labels: List[str] = []
    scores: List[float] = []

    model.eval()
    # Sentences are short; cap to reduce compute.
    max_length = 128

    use_cuda_amp = (DEVICE == "cuda")
    amp_dtype = torch.float16

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            if not batch:
                continue

            enc = tok(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(DEVICE)

            if use_cuda_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits

            probs = torch.softmax(logits, dim=-1)
            idxs = probs.argmax(dim=-1)

            for i in range(len(batch)):
                idx = int(idxs[i].item())
                labels.append(CEFR[idx])
                scores.append(float(probs[i, idx].item()))

    return labels, scores

# ========================================================
# 3. NOISE / COMPLEXIFY (LLM stub)
# ========================================================

def pick_harder_level(cefr_label: str) -> str:
    """Pick a CEFR level that is >= original."""
    idx = CEFR2IDX.get(cefr_label, None)
    if idx is None:
        return cefr_label
    harder_idx = min(idx + 1, len(CEFR) - 1)
    return CEFR[harder_idx]


def build_noisy_prompt(
    text: str,
    original_level: str,
    target_level: str,
) -> str:
    """Sentence-only complexification prompt."""
    return f"""You are an expert in CEFR-graded English sentences.
Rewrite the text so it is CEFR {target_level}, exactly one level harder than {original_level}.
Keep ALL facts, entities, and events identical.
Do NOT add new information, reasons, or descriptions.
Increase difficulty ONLY via:
- slightly more advanced vocabulary,
- more complex syntax (subordinate clauses, nominalizations).

### Example (B1 → B2)
Original: "Faults in buried transmission lines take longer to locate and repair."
Rewritten: "Faults within underground transmission lines require a lengthier period for identification and restoration."

### Now rewrite:
Original:
{text}
Return ONLY the rewritten sentence."""


# --------------------------------------------------------
# Batch prompt for multi-sentence rewriting
def build_noisy_prompt_multi(texts: List[str], original_levels: List[str], target_levels: List[str]) -> str:
    """Batch complexification prompt.

    We require explicit indices in the response so we can map outputs back safely even if the
    model reorders items.
    """
    items = []
    for i, (t, o, tgt) in enumerate(zip(texts, original_levels, target_levels)):
        items.append(f"{i}. (CEFR {o} → {tgt}) {t}")

    items_block = "\n".join(items)

    return f"""You are an expert in CEFR-graded English sentences.
Rewrite EACH item to match its target CEFR level (follow the arrow shown in each item).
Keep ALL facts, entities, and events identical.
Do NOT add new information, reasons, or descriptions.
Increase difficulty ONLY via:
- slightly more advanced vocabulary
- more complex syntax (subordinate clauses, nominalizations)

Return ONLY valid JSON as a list of objects.
Each object MUST have:
- "i": the input item number (integer)
- "rewrite": the rewritten sentence (string)

Example output:
[{{"i":0,"rewrite":"..."}},{{"i":1,"rewrite":"..."}}]

Items:
{items_block}
"""


def make_noisy_versions_batch(
    texts: List[str],
    cefr_labels: List[str],
    tokenizer=None,
    cefr_model=None,
    max_retries: int = 3,
    max_workers: int = LLM_MAX_WORKERS,
) -> Tuple[List[str], List[str]]:
    """
    Batched version of `make_noisy_version`.

    Parameters
    ----------
    texts : List[str]
        Original texts (approx. at the given CEFR labels).
    cefr_labels : List[str]
        CEFR labels of the original texts.
    granularity : str
        'sentence' or 'document'.
    tokenizer, cefr_model :
        Optional CEFR evaluator. If provided, we enforce that the generated
        text is exactly one level harder.
    max_retries : int
        How many attempts to sample a suitable candidate from the LLM.
    max_workers : int
        Thread-pool size for parallel LLM calls.

    Returns
    -------
    noisy_texts : List[str]
        More complex versions of the texts (or fall back to originals if constraints fail).
    target_levels : List[str]
        CEFR level of the noisy texts (one level higher than the originals).
    """
    assert len(texts) == len(cefr_labels)
    n = len(texts)
    if n == 0:
        return [], []

    target_levels = [pick_harder_level(lbl) for lbl in cefr_labels]

    # We will build prompts per retry in grouped batches to reduce API calls.

    noisy = list(texts)  # fallbacks
    accepted = [False] * n

    for retry_idx in range(max_retries):
        pending = [i for i, ok in enumerate(accepted) if not ok]
        print(f"[Retry {retry_idx}] Pending: {len(pending)} remaining")
        if not pending:
            break

        # Group multiple sentences into one OpenAI call to cut round-trips.
        groups = _chunks(pending, LLM_GROUP_SIZE)
        group_prompts: List[str] = []
        for g in groups:
            group_prompts.append(
                build_noisy_prompt_multi(
                    [texts[i] for i in g],
                    [cefr_labels[i] for i in g],
                    [target_levels[i] for i in g],
                )
            )

        # Heuristic output budget: ~80 tokens per item (usually plenty for sentences)
        max_out = min(4000, 80 * LLM_GROUP_SIZE)
        group_outputs = call_llm_batch(group_prompts, max_tokens=max_out, max_workers=max_workers)

        # 1) Clean and gather candidates that are non-empty and pass length check
        candidates_to_check: List[str] = []
        candidate_indices: List[int] = []  # map back to global indices

        for g, raw in zip(groups, group_outputs):
            parsed = None
            for _ in range(LLM_MAX_RETRIES_ON_PARSE):
                parsed = _extract_json_array(raw)
                if parsed is not None:
                    break

            if not isinstance(parsed, list):
                continue

            # Prefer the indexed-object format: [{"i":0,"rewrite":"..."}, ...]
            mapped_by_i: Dict[int, str] = {}
            if parsed and all(isinstance(x, dict) for x in parsed):
                for obj in parsed:
                    try:
                        i = int(obj.get("i"))
                        rewrite = (obj.get("rewrite") or "").strip()
                        if rewrite:
                            mapped_by_i[i] = rewrite
                    except Exception:
                        continue

                for local_i, global_idx in enumerate(g):
                    candidate = mapped_by_i.get(local_i, "").strip()
                    if not candidate:
                        continue
                    if len(candidate) < len(texts[global_idx]) * 0.4:
                        continue
                    candidates_to_check.append(candidate)
                    candidate_indices.append(global_idx)
                continue

            # Backward-compatible fallback: list of strings assumed to be in order.
            if parsed and all(isinstance(x, str) for x in parsed):
                if len(parsed) < len(g):
                    parsed = parsed + [""] * (len(g) - len(parsed))
                if len(parsed) > len(g):
                    parsed = parsed[:len(g)]

                for global_idx, cand in zip(g, parsed):
                    candidate = (cand or "").strip()
                    if not candidate:
                        continue
                    if len(candidate) < len(texts[global_idx]) * 0.4:
                        continue
                    candidates_to_check.append(candidate)
                    candidate_indices.append(global_idx)

        if not candidates_to_check:
            continue

        pred_labels, _ = predict_cefr_batch(
            candidates_to_check,
            tokenizer,
            cefr_model,
            batch_size=64,
        )

        # 2) Use batch predictions to accept or reject
        # Accept any candidate that is strictly more difficult than the original,
        # and set the target level to the actual predicted level.
        for cand, global_idx, pred_label in zip(candidates_to_check, candidate_indices, pred_labels):
            orig_level = cefr_labels[global_idx]
            orig_idx = CEFR2IDX.get(orig_level, -1)
            pred_idx = CEFR2IDX.get(pred_label, -1)

            # Require strictly harder (pred_idx > orig_idx). If pred is unknown, skip.
            if pred_idx <= orig_idx or pred_idx == -1 or orig_idx == -1:
                continue

            noisy[global_idx] = cand
            accepted[global_idx] = True
            # Update target level to the actual predicted level rather than the precomputed one.
            target_levels[global_idx] = pred_label

    # Discard any examples still not accepted: reset them (both noisy text and target level) to empty strings
    for i, ok in enumerate(accepted):
        if not ok:
            noisy[i] = ""          # remove fallback originals
            target_levels[i] = ""  # mark target level as unused

    return noisy, target_levels


def make_noisy_version(
    text: str,
    cefr_label: str,
    tokenizer=None,
    cefr_model=None,
    max_retries: int = 3,
):
    """Single-text wrapper around `make_noisy_versions_batch`."""
    noisy_list, target_list = make_noisy_versions_batch(
        texts=[text],
        cefr_labels=[cefr_label],
        tokenizer=tokenizer,
        cefr_model=cefr_model,
        max_retries=max_retries,
    )
    return noisy_list[0], target_list[0]


# ========================================================
# 5. BUILD PSEUDOPAIRS
# ========================================================

def build_pairs(df, tok, cefr_model):
    pairs = []

    # First pass: keep only rows whose CEFR label matches the classifier,
    # using a batched CEFR prediction to reduce overhead.
    texts_all: List[str] = df["text"].tolist()
    levels_all: List[str] = df["cefr"].tolist()

    # For now, treat all rows as valid. If you later want to filter by
    # the CEFR classifier, you can update `valid_indices`, `texts`, and
    # `levels` accordingly.
    valid_indices = df.index.tolist()

    # Batched noisy generation for all valid rows
    noisy_texts, src_harder_levels = make_noisy_versions_batch(
        texts=texts_all,
        cefr_labels=levels_all,
        tokenizer=tok,
        cefr_model=cefr_model,
        max_retries=3,
        max_workers=LLM_MAX_WORKERS,
    )

    # Only build noisy pairs; remove identity/topic options.
    for k, i in enumerate(valid_indices):
        y = texts_all[k]
        c = levels_all[k]

        # -------- Noisy version (complexified input) --------
        y_noisy = noisy_texts[k]
        src_harder_level = src_harder_levels[k]

        # If noisy generation failed for this example, y_noisy (and src_harder_level) may be empty;
        # in that case, skip creating a noisy pair.
        if not y_noisy or not src_harder_level:
            continue

        input_text = make_input_prefix(c) + y_noisy
        pairs.append({
            "id": f"sent_{k}_{c}",
            "input_text": input_text,
            "target_text": y,
            "type": "noisy",
            "granularity": "sentence",
            "source_cefr": src_harder_level,
            "target_cefr": c,
        })

    return pairs

def main():
    print("Loading datasets...")
    df = load_universal_data()

    # ---- quick sanity prints ----
    print(f"df (sentence rows): {len(df)} rows | cols={list(df.columns)}")
    if len(df) == 0:
        raise ValueError("Loaded dataframe is empty. Check your CSVs / cleaning / sentence splitting.")

    # OPTIONAL: for fast debugging, only generate a small number of pairs
    DEBUG_N = None
    if DEBUG_N is not None:
        df = df.head(DEBUG_N).copy()

    # OPTIONAL: skip first N rows (make sure to reset index)
    SKIP_N = 1
    if SKIP_N:
        df = df.iloc[18000:].reset_index(drop=True)

    print("Loading CEFR evaluator...")
    tok, cefr_model = load_cefr_model()

    print("Constructing pseudo-pairs...")
    pairs = build_pairs(df, tok, cefr_model)

    print(f"Total pairs constructed: {len(pairs)}")

    with open(OUTPUT_JSONL, "w", encoding="utf8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("Saved to", OUTPUT_JSONL)


if __name__ == "__main__":
    main()