import pandas as pd
import json
import re
import os

# ==========================================
# 1. 配置文件路径
# ==========================================
CSV_FILE_PATH = "/workspace/my_data/wikihowAll.csv" # 替换成你实际的文件路径
CHUNK_SIZE = 5000  # 每次只读 5000 行，完全不占用内存

print(f"正在分块加载数据集 (流式处理防 OOM): {CSV_FILE_PATH}...")

# 计数器
q_count = 0
a_count = 0

# ==========================================
# 2. 以追加模式打开写入文件，边处理边写
# ==========================================
# 先清空旧文件（如果存在的话）
open("raw_questions.jsonl", "w").close()
open("raw_answers.jsonl", "w").close()

with open("raw_questions.jsonl", "a", encoding="utf-8") as f_q, \
     open("raw_answers.jsonl", "a", encoding="utf-8") as f_a:

    # chunksize 参数是解决内存溢出的核心
    for chunk_idx, chunk in enumerate(pd.read_csv(CSV_FILE_PATH, chunksize=CHUNK_SIZE, on_bad_lines='skip')):
        print(f"正在处理第 {chunk_idx + 1} 块数据...")
        
        # 丢弃没有正文的行
        chunk = chunk.dropna(subset=['text'])
        
        for text in chunk['text']:
            # 按换行符切分为段落
            paragraphs = re.split(r'\n+', str(text))
            
            for p in paragraphs:
                p = p.strip()
                
                # 过滤太短的段落
                if len(p) < 10:
                    continue
                    
                # 按照CYCLE-INSTRUCT规则分类并直接写入硬盘
                if '?' in p:
                    f_q.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
                    q_count += 1
                else:
                    f_a.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
                    a_count += 1

print("\n--- 切分统计 ---")
print(f"提取到的潜在问题 (Raw Questions): {q_count} 条")
print(f"提取到的潜在答案 (Raw Answers):   {a_count} 条")
print("处理完成！结果已安全保存。")