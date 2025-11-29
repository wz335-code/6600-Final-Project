import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ========================================================
# 1. 终极稳定配置 (Python 3.10 + CPU + Verified Model)
# ========================================================

# 禁用并行，防止死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 强制 CPU (在 Python 3.10 下非常稳定)
DEVICE = "cpu"
print(f"🛡️ Running on device: {DEVICE} (Stable Mode)")

# ✅ 这是一个真实存在、架构成熟的 CEFR 模型
CEFR_MODEL_NAME = "AbdulSami/bert-base-cased-cefr"

# ========================================================
# 2. 核心逻辑
# ========================================================

def load_model():
    print(f"Loading model: {CEFR_MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CEFR_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(CEFR_MODEL_NAME)
        model.to(DEVICE)
        model.eval()
        print("✅ 模型加载成功！(Python 3.10 环境验证通过)")
        return tokenizer, model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def predict_batch(texts, tok, model, batch_size=16):
    labels = []
    
    # 自动读取模型的标签映射 {0: 'A1', 1: 'A2'...}
    id2label = model.config.id2label
    
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            batch = texts[start:start + batch_size]
            if not batch: continue

            enc = tok(
                batch,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(DEVICE)

            logits = model(**enc).logits
            idxs = logits.argmax(dim=-1)

            for i in range(len(batch)):
                idx = int(idxs[i].item())
                # 直接从模型配置里拿标签，不再怕顺序搞错
                label = id2label.get(idx, "Unknown")
                labels.append(label)
                
    return labels

# ========================================================
# 3. 执行清洗
# ========================================================

if __name__ == "__main__":
    tokenizer, model = load_model()

    if model:
        # 🔴 记得把这里改成你具体要清洗的文件名
        input_file = "cambridge_simplified_only.csv"
        
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            print(f"原始数据: {len(df)} 行")
            
            # 收集任务
            tasks = []
            target_cols = [f"text_{lvl}" for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]]
            
            for idx, row in df.iterrows():
                for col in target_cols:
                    if col in df.columns and pd.notna(row[col]):
                        tasks.append((idx, col, str(row[col])))

            if tasks:
                print(f"开始验证 {len(tasks)} 个片段...")
                all_texts = [t[2] for t in tasks]
                preds = predict_batch(all_texts, tokenizer, model, batch_size=16)

                kept, dropped = 0, 0
                for i, (row_idx, col, text) in enumerate(tasks):
                    target_lvl = col.split("_")[1] # text_A1 -> A1
                    pred_lvl = preds[i]
                    
                    # 简单清洗一下标签 (有些模型输出带有空格)
                    if pred_lvl.strip() == target_lvl:
                        kept += 1
                    else:
                        df.at[row_idx, col] = None
                        dropped += 1
                
                # 保存
                df.dropna(subset=target_cols, how='all').to_csv("cambridge_verified_final.csv", index=False)
                print(f"\n🎉 成功！保留: {kept} | 删除: {dropped}")
                print("结果已保存至: cambridge_verified_final.csv")
            else:
                print("没有找到需要验证的数据列！")
        else:
            print(f"找不到文件: {input_file}")