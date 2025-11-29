import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# =====================================
# 1. Mac M4 专属配置
# =====================================
# 禁用并行，防止 Mac 出现 "Leaked semaphore" 报错
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 检查 MPS 加速
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🚀 正在使用 Apple MPS (GPU) 加速！")
else:
    DEVICE = "cpu"
    print("⚠️ 未检测到 MPS，将使用 CPU (会比较慢)")

# 文件配置
DATA_PATH = "cambridge_flan_t5_512.jsonl" 
MODEL_NAME = "google/flan-t5-base"            

MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 512

# =====================================
# 2. 训练逻辑
# =====================================

def main():
    print(f"Loading dataset from: {DATA_PATH}")
    
    # 1. 读取数据
    df = pd.read_json(DATA_PATH, lines=True)
    print(f"Total examples: {len(df)}")

    # 2. 按文章 ID 切分 (严防数据泄漏)
    # 提取基础ID (0_A1 -> 0)
    df['base_id'] = df['id'].astype(str).apply(lambda x: x.split('_')[0])
    
    unique_ids = df['base_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_ids)
    
    # 90% 训练，10% 验证
    split_idx = int(len(unique_ids) * 0.9)
    train_ids = unique_ids[:split_idx]
    val_ids = unique_ids[split_idx:]
    
    train_df = df[df['base_id'].isin(train_ids)]
    val_df = df[df['base_id'].isin(val_ids)]
    
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")
    
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding=False, 
        )
        labels = tokenizer(
            text_target=batch["target_text"], 
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing datasets...")
    # remove_columns 很重要，防止格式冲突
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)

    # 4. 加载模型
    print(f"Loading Model: {MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE) # 移动到 GPU

    # 5. 配置 LoRA (轻量化微调)
    # 这让你的 Mac 即使跑大一点的模型也不会发烫太严重
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"], # T5 的注意力层
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# 6. 训练参数 (Mac 优化最终版)
    training_args = Seq2SeqTrainingArguments(
        output_dir="mac_flan_t5_finetuned",
        num_train_epochs=5,             
        per_device_train_batch_size=8,  
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  
        learning_rate=3e-4,
        warmup_ratio=0.05,
        
        # Mac 关键设置
        fp16=False, 
        bf16=False, 
        
        logging_steps=10,
        
        # 🔴 关键修改点：让保存和评估频率一致 🔴
        eval_strategy="epoch",  # 每跑完一轮，评估一次
        save_strategy="epoch",  # 每跑完一轮，保存一次
        
        save_total_limit=2,     # 只保留最好的2个模型，防止硬盘塞满
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        dataloader_num_workers=0, 
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[],
    )

    print("🚀 Starting local training on Mac M4 Pro...")
    trainer.train()
    
    print("✅ Training finished.")
    # 保存模型
    save_path = "mac_final_model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to ./{save_path}")

if __name__ == "__main__":
    main()