import json
import torch
import csv
import os
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ================= 配置区域 =================
# 输入文件
TEST_FILE = "tsar2025_simplified.jsonl" 

# 输出的 CSV 文件名
OUTPUT_CSV = "model_evaluation_results.csv"

# 模型路径
ADAPTER_PATH = "./flan-t5-cefr"
BASE_MODEL_NAME = "google/flan-t5-base"

# 测试样本数量 (-1 代表测试文件里的所有数据)
NUM_SAMPLES = 20
# ===========================================

def load_model():
    print(f"正在加载基础模型 {BASE_MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.float32
    )
    
    if not os.path.exists(ADAPTER_PATH):
        print(f"错误：找不到模型路径 {ADAPTER_PATH}")
        return None, None

    print(f"正在加载 LoRA 权重 {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return tokenizer, model

def main():
    # 1. 加载模型
    tokenizer, model = load_model()
    if model is None: return

    # 2. 读取数据
    if not os.path.exists(TEST_FILE):
        print(f"错误：找不到文件 {TEST_FILE}")
        return

    print(f"正在读取数据文件: {TEST_FILE}")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 如果 NUM_SAMPLES 是 -1，则测试所有数据
    total_lines = len(lines)
    limit = NUM_SAMPLES if NUM_SAMPLES > 0 else total_lines
    print(f"即将处理 {limit} 条数据，并保存到 {OUTPUT_CSV} ...")

    # 3. 准备写入 CSV
    # encoding='utf-8-sig' 是为了让 Excel 能正确识别中文/特殊字符
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        
        # 写入表头 (Header)
        header = ["ID", "Input (Instruction+Source)", "Target (Human Reference)", "Prediction (Model Output)"]
        writer.writerow(header)

        count = 0
        for i, line in enumerate(lines):
            if count >= limit:
                break
                
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            input_text = data.get("input_text", "")
            reference_text = data.get("target_text", "")
            
            if not input_text:
                continue

            # --- 模型生成 ---
            input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(model.device)
            
# --- 修改后的模型生成参数 ---
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=512,
                    
                    # 1. 开启采样，增加创造性
                    do_sample=True, 
                    temperature=0.7,  # 温度越高，改写幅度越大 (0.7-0.9 比较合适)
                    top_p=0.9,        # 核采样
                    
                    # 2. 关键：增加重复惩罚 (防止直接照抄)
                    # 1.0 代表没惩罚，建议设置在 1.1 到 1.3 之间
                    repetition_penalty=1.2, 
                    
                    # 3. 保持 Beam Search (可选，如果开启 sample，num_beams 可以设为 1 或者保持)
                    num_beams=1, 
                    
                    early_stopping=True
                )
            # ---------------------------
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # ----------------

            # 写入一行数据
            writer.writerow([i+1, input_text, reference_text, prediction])
            
            count += 1
            
            # 每处理 5 条在终端打印一下进度，避免以为程序卡死
            if count % 5 == 0:
                print(f"已处理 {count}/{limit} 条...")

    print("\n" + "="*40)
    print(f"测试完成！结果已保存至: {OUTPUT_CSV}")
    print("="*40)

if __name__ == "__main__":
    main()