import json
from pathlib import Path

IN_PATH  = Path("train_weighted.jsonl")
OUT_PATH = Path("train_weighted_with_source.jsonl")

def to_source(weight) -> str:
    w = float(weight)
    if abs(w - 5.0) < 1e-9:
        return "gold"
    if abs(w - 1.0) < 1e-9:
        return "pseudo"
    return "other"  # for any unexpected weights

with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        ex = json.loads(line)
        if "weight" not in ex:
            raise ValueError("Missing 'weight' in a line")
        ex["source"] = to_source(ex["weight"])
        fout.write(json.dumps(ex, ensure_ascii=False) + "\n")