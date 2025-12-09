import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TORCH"] = "1"

import json
from tqdm import tqdm
from build_finetuning_signals import load_cefr_model, predict_cefr_batch
import torch


# Classifier (from TSAR paper families)
CEFR_MODEL_NAME = "AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr"

# ========================================================
# CEFR EVALUATOR
# ========================================================


INPUT = "tsar2025.jsonl"
OUTPUT = "tsar2025_formatted.jsonl"
BATCH_SIZE = 32

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    records = list(iter_jsonl(INPUT))
    originals = [r["original"] for r in records]
    tok, cefr_model = load_cefr_model()

    # Run evaluator in batches
    pred_levels = []
    for i in tqdm(range(0, len(originals), BATCH_SIZE)):
        batch = originals[i:i+BATCH_SIZE]
        pred_levels.extend(predict_cefr_batch(batch, tok, cefr_model, len(batch))[0])
        print(f"Processed {len(pred_levels)}/{len(originals)}")
    print(len(pred_levels), len(originals))
    assert len(pred_levels) == len(originals)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec, src_cefr in zip(records, pred_levels):
            tgt_cefr = rec["target_cefr"]
            reference = rec["reference"]
            original = rec["original"]

            # identity or noisy?
            pair_type = "identity" if src_cefr == tgt_cefr else "noisy"

            # build the x prompt
            x_text = (
                f"<task=simplify> <granularity=paragraph> <pair={pair_type}> "
                f"<source_cefr={src_cefr}> <target_cefr={tgt_cefr}>\n"
                f"Rewrite the following text to CEFR {tgt_cefr} level:\n\n"
                f"{original}"
            )

            new_obj = {
                "x": x_text,
                "y": reference,
                "target_cefr": tgt_cefr,
                "type": pair_type
            }

            f.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    print("Done →", OUTPUT)


if __name__ == "__main__":
    main()    
