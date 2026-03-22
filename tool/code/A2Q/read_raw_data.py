import json, re

# 句号/分号/问号/感叹号/换行作为软边界
SENT_SPLIT = re.compile(r'(?<=[。！？；\?\!])\s*|\n+')

def read_raw_data(file,target_min=300, target_max=600, overlap=0):
  # 读取txt文件并按空行切分
  with open(file, 'r', encoding='utf-8') as f:
      text = f.read()

  sentences = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
  chunks, cur = [], []
  cur_len = 0
  for s in sentences:
      if cur_len + len(s) <= target_max or cur_len < target_min:
          cur.append(s); cur_len += len(s)
      else:
          chunk = " ".join(cur).strip()
          if chunk:
              chunks.append(chunk)
          # 构造重叠：从末尾回取若干字符的句子

          overlap_sents, ol_len = [], 0
          for t in reversed(cur):
              if ol_len >= overlap: break
              overlap_sents.insert(0, t); ol_len += len(t)
          cur = overlap_sents + [s]
          cur_len = sum(len(x) for x in cur)
  if cur:
      chunks.append(" ".join(cur).strip())
  # 合理性微调：过短合并
  cleaned = []
  for ch in chunks:
      if cleaned and len(ch) < 200 and len(cleaned[-1]) + len(ch) <= target_max + 200:
          cleaned[-1] = cleaned[-1] + " " + ch
      else:
          cleaned.append(ch)
  return cleaned

if __name__ == "__main__":
    A = read_raw_data("../data/train_data.txt")

    output_path = "../out/out_QA1.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(A, f, ensure_ascii=False, indent=2)