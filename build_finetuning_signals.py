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

from openai import OpenAI

client = OpenAI()

# ========================================================
# 0. CONFIG
# ========================================================
def call_llm(prompt: str, max_tokens: int = 1024) -> str:
    """
    Thin wrapper around OpenAI Responses API.
    Adjust model name as you like (e.g. gpt-4.1, gpt-4.1-mini, gpt-4o-mini).
    """
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=max_tokens,
    )
    # Responses API: output[0].content[0].text
    return resp.output[0].content[0].text.strip()

def call_llm_batch(prompts: List[str], max_tokens: int = 1024, max_workers: int = 8) -> List[str]:
    """
    Issue multiple LLM calls in parallel using a thread pool.

    This does not use the asynchronous Batch API; instead it parallelizes
    regular synchronous calls to `call_llm`, which is usually enough to
    greatly speed up large-scale data generation.
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

sentence_classification_files = [
    "cleaned_readme.csv",
]

document_classification_files = [
    "cleaned_cambridge.csv",
    "cleaned_elg.csv",
]

OUTPUT_JSONL = "document4_universalcefr_pseudopairs.jsonl"

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

# Input prefix builder for tagged tasks
def make_input_prefix(granularity: str, pair_type: str, source_cefr: str, target_cefr: str) -> str:
    """Build a compact prefix tag describing the task configuration.

    granularity: 'sentence' or 'document'
    pair_type:  'identity' | 'noisy' | 'topic' | ...
    source_cefr: CEFR level of the input text
    target_cefr: CEFR level we want to simplify to
    """
    return (
        f"<task=simplify> "
        f"<granularity={granularity}> "
        f"<pair={pair_type}> "
        f"<source_cefr={source_cefr}> "
        f"<target_cefr={target_cefr}>"
    )

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Classifier (from TSAR paper families)
CEFR_MODEL_NAME = "AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr"

# Sentence embedding model for topic match
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_TOPIC_MATCHES = 1  # how many harder texts to pair each y with each easier text

# ========================================================
# 1. LOAD DATA
# ========================================================

def load_universal_data() -> pd.DataFrame:
    all_sentence_df = []
    all_document_df = []
    for f in sentence_classification_files:
        sentence_df = pd.read_csv(f)
        sentence_df['cefr'] = sentence_df['cefr_level']
        sentence_df = sentence_df.dropna(subset=["cefr"])
        sentence_df = sentence_df[["text", "cefr"]]
        all_sentence_df.append(sentence_df)
    sentence_df = pd.concat(all_sentence_df, ignore_index=True)

    for f in document_classification_files:
        document_df = pd.read_csv(f)
        document_df['cefr'] = document_df['cefr_level']
        document_df = document_df.dropna(subset=["cefr"])
        document_df = document_df[["text", "cefr"]]
        all_document_df.append(document_df)
    document_df = pd.concat(all_document_df, ignore_index=True)
    document_df['text'] = document_df['text'].apply(lambda t: clean_text(t, sentence_tokenize=True))

    return sentence_df, document_df

# ========================================================
# 2. CEFR EVALUATOR
# ========================================================

def load_cefr_model():
    tokenizer = AutoTokenizer.from_pretrained(CEFR_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(CEFR_MODEL_NAME)
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
    batch_size: int = 16,
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
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            if not batch:
                continue

            enc = tok(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)

            logits = model(**enc).logits  # [B, num_labels]
            probs = torch.softmax(logits, dim=-1)  # [B, num_labels]
            idxs = probs.argmax(dim=-1)            # [B]

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
    granularity: str,
) -> str:
    """
    Construct the user prompt for making the text one CEFR level harder.
    Shared between single-example and batched generation.
    """
    orig_desc = LEVEL_DESCRIPTORS.get(original_level, "")
    target_desc = LEVEL_DESCRIPTORS.get(target_level, "")

    if granularity == "sentence":
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
    else:  # "document"
        return f"""You are an expert in CEFR-graded English documents.
Rewrite the document so it is CEFR {target_level}, exactly one level different from {original_level}.
Keep all facts, entities, events, and paragraph boundaries identical.
Do NOT add new information, explanations, or opinions.
Increase difficulty ONLY through:
- more advanced vocabulary,
- more complex syntax,
- light discourse markers that restate existing relations.

### Example (B1 → A2)
Original:
"Many individuals in Britain have embraced a new pastime—line dancing. In nearly every town, one can discover clubs and classes dedicated to this engaging activity."

Rewritten:
"Thousands of people in Britain have a new hobby -- line dancing. In almost every town, you will find clubs and classes for this new activity."

### Now rewrite:
Original:
{text}

Return ONLY the rewritten document."""


def make_noisy_versions_batch(
    texts: List[str],
    cefr_labels: List[str],
    granularity: str,
    tokenizer=None,
    cefr_model=None,
    max_retries: int = 3,
    max_workers: int = 8,
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

    # Pre-build prompts once; they stay the same across retries.
    prompts = [
        build_noisy_prompt(texts[i], cefr_labels[i], target_levels[i], granularity)
        for i in range(n)
    ]

    noisy = list(texts)  # fallbacks
    accepted = [False] * n

    for retry_idx in range(max_retries):
        pending = [i for i, ok in enumerate(accepted) if not ok]
        print(f"[Retry {retry_idx}] Pending: {len(pending)} remaining")
        if not pending:
            break

        batch_prompts = [prompts[i] for i in pending]
        batch_outputs = call_llm_batch(batch_prompts, max_tokens=1024, max_workers=max_workers)

        # 1) Clean and gather candidates that are non-empty and pass length check
        candidates_to_check: List[str] = []
        candidate_indices: List[int] = []  # map back to global indices

        for local_idx, global_idx in enumerate(pending):
            candidate = batch_outputs[local_idx].strip()
            if not candidate:
                continue

            # Basic sanity: avoid very short / clearly truncated outputs
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
            batch_size=16,
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
    granularity: str,
    tokenizer=None,
    cefr_model=None,
    max_retries: int = 3,
):
    """
    Single-text wrapper around `make_noisy_versions_batch` to preserve
    the previous interface.
    """
    noisy_list, target_list = make_noisy_versions_batch(
        texts=[text],
        cefr_labels=[cefr_label],
        granularity=granularity,
        tokenizer=tokenizer,
        cefr_model=cefr_model,
        max_retries=max_retries,
    )
    return noisy_list[0], target_list[0]

# ========================================================
# 4. TOPIC MATCHING USING EMBEDDINGS
# ========================================================

def build_topic_matches(df):
    model = SentenceTransformer(EMB_MODEL_NAME)
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True, convert_to_tensor=True)

    # Pre-bucket texts by CEFR for fast retrieval
    buckets = {c: df[df["cefr"] == c].index.tolist() for c in CEFR}
    bucket_embs = {c: embeddings[buckets[c]] for c in CEFR}

    print("Bucket sizes:")
    for c in CEFR:
        print(c, "→", len(buckets[c]))
    topic_map = {}

    for i, row in tqdm(df.iterrows(), total=len(df)):
        y = row["text"]
        c = row["cefr"]
        y_emb = embeddings[i]

        # find higher levels
        higher_levels = CEFR[CEFR2IDX[c]+1:]
        candidates = []
        for hl in higher_levels:
            if len(buckets[hl]) == 0:
                continue
            sims = util.cos_sim(y_emb, bucket_embs[hl])[0]
            topk = torch.topk(sims, k=min(MAX_TOPIC_MATCHES, len(sims)))
            for score, idx in zip(topk.values, topk.indices):
                j = buckets[hl][int(idx)]
                candidates.append((j, float(score)))

        topic_map[i] = candidates

    return topic_map

def build_balanced_pairs_for_level(df, topic_map, src_level, pairs_per_source=3):
    """
    For all texts with CEFR == src_level (e.g. 'A1'),
    build (complex_index, simple_index) pairs so that
    higher CEFR levels are used roughly evenly.
    """
    src_indices = df.index[df["cefr"] == src_level].tolist()
    higher_levels = CEFR[CEFR2IDX[src_level] + 1:]

    # track how many times we've used each higher level
    level_counts = {hl: 0 for hl in higher_levels}
    pairs = []

    for i in src_indices:
        # all candidates for this source i
        cand_list = topic_map[i]

        # group candidates by their CEFR level
        cands_by_level = {hl: [] for hl in higher_levels}
        for j, score in cand_list:
            hl = df.loc[j, "cefr"]
            if hl in cands_by_level:
                cands_by_level[hl].append((j, score))

        # sort within each level by similarity (highest first)
        for hl in higher_levels:
            cands_by_level[hl].sort(key=lambda x: x[1], reverse=True)

        # choose up to pairs_per_source matches, preferring levels that are under-used
        for _ in range(pairs_per_source):
            # among levels that still have candidates, pick the one with smallest global count
            available_levels = [hl for hl in higher_levels if cands_by_level[hl]]
            if not available_levels:
                break

            hl_chosen = min(available_levels, key=lambda hl: level_counts[hl])

            # take the best remaining candidate from that level
            j, score = cands_by_level[hl_chosen].pop(0)
            pairs.append((j, i))  # (complex_idx, simple_idx)
            level_counts[hl_chosen] += 1

    return pairs, level_counts

# ========================================================
# 5. BUILD PSEUDOPAIRS
# ========================================================

def build_pairs(df, tok, cefr_model, granularity, topic_based: bool = False):
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
        granularity=granularity,
        tokenizer=tok,
        cefr_model=cefr_model,
        max_retries=3,
        max_workers=8,
    )

    # Second pass: construct identity, noisy, and (optionally) topic-based pairs
    for k, i in enumerate(valid_indices):
        y = texts_all[k]
        c = levels_all[k]

        # -------- (1) Identity rewriting --------
        prefix_identity = make_input_prefix(
            granularity=granularity,
            pair_type="identity",
            source_cefr=c,
            target_cefr=c,
        )
        x = (
            prefix_identity
            + "\nRewrite the following text at CEFR "
            + c
            + " level:\n\n"
            + y
        )
        pairs.append({"x": x, "y": y, "target_cefr": c, "type": "identity"})

        # -------- (2) Noisy version (complexified input) --------
        y_noisy = noisy_texts[k]
        src_harder_level = src_harder_levels[k]

        # If noisy generation failed for this example, y_noisy (and src_harder_level) may be empty;
        # in that case, skip creating a noisy pair.
        if not y_noisy or not src_harder_level:
            continue

        prefix_noisy = make_input_prefix(
            granularity=granularity,
            pair_type="noisy",
            source_cefr=src_harder_level,
            target_cefr=c,
        )
        x2 = (
            prefix_noisy
            + "\nRewrite the following text to CEFR "
            + c
            + " level:\n\n"
            + y_noisy
        )
        pairs.append({"x": x2, "y": y, "target_cefr": c, "type": "noisy"})

        if not topic_based:
            continue
        # -------- (3) Topic-based harder → y (optional) --------
        if topic_based:
            for j, _ in topic_map.get(i, []):
                z = df.iloc[j]["text"]
                cz = df.iloc[j]["cefr"]
                prefix_topic = make_input_prefix(
                    granularity=granularity,
                    pair_type="topic",
                    source_cefr=cz,
                    target_cefr=c,
                )
                x3 = (
                    prefix_topic
                    + "\nSimplify the following "
                    + cz
                    + "-level text to CEFR "
                    + c
                    + ":\n\n"
                    + z
                )
                pairs.append({"x": x3, "y": y, "target_cefr": c, "type": "topic"})

    return pairs

def main():
    print("Loading datasets...")
    sentence_df, document_df = load_universal_data()
    #sentence_df = sentence_df.iloc[250:]  
    document_df = document_df.iloc[650:]  

    print("Loading CEFR evaluator...")
    tok, cefr_model = load_cefr_model()
    """
    print("Building topic matches...")
    sentence_topic_map = build_topic_matches(sentence_df)
    document_topic_map = build_topic_matches(document_df)
    """
    print("Constructing pseudo-pairs...")
    
    document_pairs = build_pairs(document_df, tok, cefr_model, granularity="document")
    #sentence_pairs = build_pairs(sentence_df, tok, cefr_model, granularity="sentence")
    pairs = document_pairs
    print(f"Total pairs constructed: {len(pairs)}")

    with open(OUTPUT_JSONL, "w", encoding="utf8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("Saved to", OUTPUT_JSONL)

if __name__ == "__main__":
    main()