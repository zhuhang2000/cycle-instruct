from datasets import Dataset

# 1. 加载本地的 test 测试集
# 请将路径替换为你截图中 mmlu-test.arrow 的实际绝对或相对路径
file_path = "/workspace/my_data/cais___mmlu/abstract_algebra/0.0.0/c30699e8356da336a370243923dbaf21066bb9fe/mmlu-test.arrow"

try:
    # 直接从 arrow 文件加载数据集
    test_dataset = Dataset.from_file(file_path)
except Exception as e:
    print(f"加载数据集失败: {e}")
    exit()

# 用于将数字索引映射为选项字母
idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

correct_count = 0
total_count = len(test_dataset)

print(f"成功加载抽象代数测试集，共 {total_count} 道题目。开始测试...\n")

# 2. 遍历测试集进行评估
for i, item in enumerate(test_dataset):
    question = item['question']
    choices = item['choices']
    answer_idx = item['answer']
    correct_answer = idx_to_letter[answer_idx]
    
    # 构造 Zero-shot (零样本) Prompt
    prompt = f"Question: {question}\n"
    prompt += f"A. {choices[0]}\n"
    prompt += f"B. {choices[1]}\n"
    prompt += f"C. {choices[2]}\n"
    prompt += f"D. {choices[3]}\n"
    prompt += "Answer:"
    
    # -------------------------------------------------------------
    # 3. 调用你的大语言模型 (此处需要你根据实际使用的模型进行替换)
    # -------------------------------------------------------------
    # 例如：如果你使用本地部署的 transformers 或 vLLM，在这里传入 prompt
    # predicted_text = your_llm_generate(prompt)
    
    # 这里我们用一个模拟输出代替，假设模型总是回答 "A"
    predicted_text = "A" 
    
    # 提取模型输出的第一个有效字母作为答案
    predicted_letter = predicted_text.strip().upper()[0] if predicted_text else ""
    
    # 对比正确答案
    if predicted_letter == correct_answer:
        correct_count += 1
        
    if i < 2: # 打印前两题的示例供你调试
        print(f"--- 题目 {i+1} ---")
        print(prompt)
        print(f"模型预测: {predicted_letter}, 正确答案: {correct_answer}\n")

# 4. 计算并输出总准确率
accuracy = (correct_count / total_count) * 100
print("-" * 30)
print(f"测试完成！")
print(f"总题数: {total_count}")
print(f"答对题数: {correct_count}")
print(f"MMLU 抽象代数准确率 (Accuracy): {accuracy:.2f}%")