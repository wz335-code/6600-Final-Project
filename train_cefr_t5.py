import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TORCH"] = "1"
from dataclasses import dataclass
from typing import Optional

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

DATA_PATH = "pseudopairs.jsonl"   # your JSONL file
MODEL_NAME = "t5-base"           # use "t5-small" for very quick debugging

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
        num_train_epochs=50,
        max_steps=None,            # hard cap: only 30 optimizer steps
        train_subset=8,        # only use first 256 examples
        per_device_batch_size=1,
        grad_accum_steps=1,
        fp16=False,              # keep it simple when debugging
        logging_steps=1,
        save_steps=1000000,      # effectively "don't save during debug"
        eval_strategy="steps",
    )
else:
    train_cfg = TrainConfig(
        num_train_epochs=8,
        max_steps=None,          # train for full epochs
        train_subset=None,       # use all data
        per_device_batch_size=8,
        grad_accum_steps=4,      # global batch = 32
        fp16=False,               # good for T4
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",
    )

# =====================================
# Main
# =====================================

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

    # Keep just input/output text; your JSONL has keys: x, y, target_cefr, type
    def keep_fields(ex):
        return {
            "input_text": ex["x"],
            "target_text": ex["y"],
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
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_tok = val_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=val_ds.column_names,
    )

    print("Loading model:", MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # LoRA configuration for T5
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
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

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="t5-cefr",
        num_train_epochs=train_cfg.num_train_epochs,
        max_steps=train_cfg.max_steps if train_cfg.max_steps is not None else -1,
        per_device_train_batch_size=train_cfg.per_device_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_batch_size,
        gradient_accumulation_steps=train_cfg.grad_accum_steps,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        fp16=train_cfg.fp16,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        save_total_limit=2,
        predict_with_generate=False,
        report_to="none",  # disable wandb etc.
        eval_strategy=train_cfg.eval_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok if train_cfg.eval_strategy != "no" else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)],
    )

    print("Starting training...")
    trainer.train()
    print("Training finished, saving model...")
    trainer.save_model("t5-cefr")
    print("Saved to ./t5-cefr")

if __name__ == "__main__":
    main()