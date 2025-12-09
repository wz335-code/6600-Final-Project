import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TORCH"] = "1"
from dataclasses import dataclass
from typing import Optional
import inspect

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn.functional as F

DATA_PATH = "finetuning_signals.jsonl"   # your JSONL file

# Mixture training weights
GOLD_WEIGHT = 5.0
PSEUDO_WEIGHT = 1.0

def infer_example_weight(ex: dict) -> float:
    """Heuristic: treat TSAR (human gold) as higher weight; otherwise pseudo.

    Supports a few common fields you may have in JSONL:
    - ex['is_gold'] == True
    - ex['type'] in {'gold', 'human', 'tsar'}
    - ex['dataset_id'] contains 'tsar'
    """
    if bool(ex.get("is_gold", False)):
        return GOLD_WEIGHT
    t = str(ex.get("type", "")).lower()
    if t in {"gold", "human", "tsar", "tsar2025test", "tsar2025_test"}:
        return GOLD_WEIGHT
    dsid = str(ex.get("dataset_id", "")).lower()
    if "tsar" in dsid:
        return GOLD_WEIGHT
    return PSEUDO_WEIGHT

#
# Flan-T5 is instruction-tuned; use it as the base for CEFR simplification.
MODEL_NAME = "google/flan-t5-base"  # try "google/flan-t5-small" for very quick debugging

# Your examples are often paragraph/document-like; 256 can truncate too aggressively.
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256

# Turn this on for quick, cheap debugging (CPU / tiny subset / few steps)
DEBUG = False

# =====================================
# Helper: training config depending on DEBUG
# =====================================

@dataclass
class TrainConfig:
    num_train_epochs: int
    max_steps: Optional[int]
    train_subset: Optional[int]
    per_device_batch_size: int
    grad_accum_steps: int
    fp16: bool
    logging_steps: int
    save_steps: int
    eval_strategy: str

if DEBUG:
    train_cfg = TrainConfig(
        num_train_epochs=1,
        max_steps=150,             # hard cap: only 30 optimizer steps
        train_subset=64,          # small subset for quick overfit / sanity checks
        per_device_batch_size=1,
        grad_accum_steps=4,
        fp16=False,               # keep it simple when debugging
        logging_steps=1,
        save_steps=10,            # save a few checkpoints so load_best_model_at_end works
        eval_strategy="steps",
    )
else:
    train_cfg = TrainConfig(
        num_train_epochs=8,
        max_steps=None,          # train for full epochs
        train_subset=None,       # use all data
        per_device_batch_size=8,
        grad_accum_steps=2,      # global batch = 32
        fp16=False,               # good for T4
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",
    )

# =====================================
# Main
# =====================================

class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that supports per-example weights via an input key 'weight'."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight", None)  # [B]
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]

        # Token-level CE with ignore_index, then normalize per-example by #valid tokens.
        loss_per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.size(0), -1)  # [B, T]

        valid = (labels != -100).float()
        loss_per_ex = (loss_per_token * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)  # [B]

        if weights is not None:
            weights = weights.to(loss_per_ex.device).float()
            loss = (loss_per_ex * weights).sum() / weights.sum().clamp_min(1e-8)
        else:
            loss = loss_per_ex.mean()

        return (loss, outputs) if return_outputs else loss

def main():
    # Optional: uncomment this to force CPU while debugging on VM
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print("Loading dataset from:", DATA_PATH)
    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    print("Total examples in file:", len(ds))

    # Use only a subset when debugging
    if train_cfg.train_subset is not None and len(ds) > train_cfg.train_subset:
        ds = ds.select(range(train_cfg.train_subset))
        print(f"DEBUG mode: using only first {len(ds)} examples")

    # Keep just input/output text; support only {input_text,target_text} schema.
    # If the JSONL already provides a numeric `weight`, use it; otherwise infer via heuristics.
    def keep_fields(ex):
        src = ex.get("input_text", None)
        tgt = ex.get("target_text", None)
        if src is None or tgt is None:
            raise KeyError("Example must contain either (input_text,target_text)")

        w = ex.get("weight", None)
        if w is None:
            w = infer_example_weight(ex)

        return {
            "input_text": src,
            "target_text": tgt,
            "weight": float(w),
        }

    ds = ds.map(keep_fields)

    # Simple train/validation split
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]

    print("Train size:", len(train_ds), "Val size:", len(val_ds))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        # Encode input (x)
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding=False,  # dynamic padding via DataCollator
        )

        # Encode target (y)
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]

        # Carry per-example weights through to the Trainer.
        model_inputs["weight"] = batch["weight"]
        return model_inputs

    train_tok = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in train_ds.column_names if c != "weight"],
    )
    val_tok = val_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in val_ds.column_names if c != "weight"],
    )

    # Quick sanity check: show one example length (helps catch accidental truncation)
    ex0 = train_tok[0]
    print("[sanity] src tokens:", len(ex0["input_ids"]), "tgt tokens:", len(ex0["labels"]))

    print("Loading model:", MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # LoRA configuration for T5
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # T5 attention uses projections named q/k/v/o; adapting q,v,o often helps more than q,v alone.
        target_modules=["q", "v", "o"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    num_trainable = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            num_trainable += 1
    print("Number of trainable tensors:", num_trainable)
    model.config.use_cache = False

    # Training arguments (make version-compatible with Transformers)
    args_dict = dict(
        output_dir="t5-cefr",
        num_train_epochs=train_cfg.num_train_epochs,
        max_steps=train_cfg.max_steps if train_cfg.max_steps is not None else -1,
        per_device_train_batch_size=train_cfg.per_device_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_batch_size,
        gradient_accumulation_steps=train_cfg.grad_accum_steps,
        learning_rate=5e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        fp16=train_cfg.fp16,
        logging_steps=train_cfg.logging_steps,
        eval_steps=train_cfg.save_steps if train_cfg.eval_strategy == "steps" else None,
        save_steps=train_cfg.save_steps,
        save_total_limit=2,
        predict_with_generate=False,
        report_to="none",  # disable wandb etc.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        remove_unused_columns =False,
    )

    # Handle naming differences across Transformers versions.
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    accepted = set(sig.parameters.keys())
    if "eval_strategy" in accepted:
        args_dict["eval_strategy"] = train_cfg.eval_strategy
    elif "evaluation_strategy" in accepted:
        args_dict["evaluation_strategy"] = train_cfg.eval_strategy

    # Only include eval_steps when doing step-based evaluation.
    if train_cfg.eval_strategy == "steps" and "eval_steps" in accepted:
        args_dict["eval_steps"] = train_cfg.save_steps

    training_args = Seq2SeqTrainingArguments(**args_dict)

    base_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def data_collator(features):
        # Pop weights off each feature dict before passing to HF collator.
        weights = torch.tensor([f.pop("weight") for f in features], dtype=torch.float)
        batch = base_collator(features)
        batch["weight"] = weights
        return batch

    trainer = WeightedSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok if train_cfg.eval_strategy != "no" else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)]
        if train_cfg.eval_strategy != "no" else [],
    )

    print("Starting training...")
    trainer.train()
    print("Training finished, saving model...")
    trainer.save_model("flan-t5-cefr")
    print("Saved to ./flan-t5-cefr")

if __name__ == "__main__":
    main()