# cycle-instruct

一个用于构建 Cycle-Instruct 数据流程的小型项目，包含：

- 原始数据预处理（CSV/Arrow/JSON）
- A2Q 伪数据生成（Question -> Answer，再转为 A2Q 训练样本）
- MMLU Arrow 数据的本地模型评测

---

## 1. 目录结构（核心）

- [tool/prepare_cycle_data.py](tool/prepare_cycle_data.py)：从 CSV 流式提取问句/答句到 JSONL。
- [tool/convert_to_llamafactory_json.py](tool/convert_to_llamafactory_json.py)：把通用 JSON 转成 LlamaFactory alpaca 格式。
- [tool/arrow_to_json.py](tool/arrow_to_json.py)：把 `.arrow` 转换为 JSON/JSONL。
- [tool/model_loader.py](tool/model_loader.py)：统一模型加载（模型路径、量化、设备配置集中在这里）。
- [code/A2Q/generate_pseudo_a.py](code/A2Q/generate_pseudo_a.py)：生成 A2Q 伪数据（输出 LlamaFactory 可用 JSON）。
- [test_model/test.py](test_model/test.py)：本地模型评测 MMLU `.arrow`。

> 说明：当前仓库内的 [readme.md](readme.md) 即本文件；重点介绍 Python 运行方式。

---

## 2. Python 环境准备

建议 Python 3.10+。

### 2.1 创建虚拟环境（Windows PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 2.2 安装依赖

按当前脚本实际 import，至少需要：

```powershell
pip install torch transformers datasets bitsandbytes pandas
```

如果在 Windows 上 `bitsandbytes` 不可用，可先安装其余包；量化相关功能主要面向 Linux + NVIDIA CUDA 环境。

---

## 3. 运行前的关键配置

统一模型配置在 [tool/model_loader.py](tool/model_loader.py) 中维护：

- `MODEL_PATH`
- `QUANTIZATION`（`none` / `8bit` / `4bit`）
- `DEVICE_GPU`、`DTYPE_GPU` 等

当前脚本调用 `load_causal_lm()` 为**无参**模式，所以你只需先改好 [tool/model_loader.py](tool/model_loader.py) 里的这些常量。

---

## 4. Python 执行方式（重点）

以下命令均在仓库根目录执行（当前项目根：`cycle-instruct`）。

### 4.1 生成 A2Q 伪数据

脚本：[code/A2Q/generate_pseudo_a.py](code/A2Q/generate_pseudo_a.py)

输入 JSON 每条建议包含 `input` 字段（脚本当前按 `item.get("input")` 读取问题）。

```powershell
python code/A2Q/generate_pseudo_a.py `
	--input data/input.json `
	--output data/a2q_pseudo.json `
	--max-new-tokens 256 `
	--temperature 0.0 `
	--top-p 0.9 `
	--save-every 100
```

运行中会周期性落盘并打印进度，结束后输出 LlamaFactory 可训练格式（含 `instruction/input/output`）。

---

### 4.2 本地模型评测 MMLU Arrow

脚本：[test_model/test.py](test_model/test.py)

```powershell
python test_model/test.py `
	--file-path data/mmlu-test.arrow `
	--max-new-tokens 16 `
	--temperature 0.0 `
	--top-p 0.9 `
	--print-samples 2
```

#### 日志路径格式说明（重点）

- 不传 `--log-file`：
	- 自动生成到 `test_model/logs/` 目录
	- 文件名格式：`mmlu_eval_YYYYMMDD_HHMMSS.log`
	- 示例：`test_model/logs/mmlu_eval_20260322_153045.log`
- 传相对路径：
	- 相对“当前执行目录”解析
	- 示例：`--log-file logs/run1.log`
- 传绝对路径：
	- 直接写入指定位置（目录不存在会自动创建）
	- 示例：`--log-file /workspace/logs/mmlu_eval.log`

示例（指定日志文件）：

```powershell
python test_model/test.py --file-path data/mmlu-test.arrow --log-file test_model/logs/manual_run.log
```

脚本会输出：

- 样例题目预测
- 总题数 / 答对题数 / 准确率

---

### 4.3 通用 JSON 转 LlamaFactory 格式

脚本：[tool/convert_to_llamafactory_json.py](tool/convert_to_llamafactory_json.py)

#### 基础

```powershell
python tool/convert_to_llamafactory_json.py data/src.json data/out_lf.json
```

#### 自定义字段名

```powershell
python tool/convert_to_llamafactory_json.py data/src.json data/out_lf.json --input-key question --output-key answer
```

#### 保留原始 `instruction`

```powershell
python tool/convert_to_llamafactory_json.py data/src.json data/out_lf.json --keep-original-instruction
```

---

### 4.4 Arrow 转 JSON/JSONL

脚本：[tool/arrow_to_json.py](tool/arrow_to_json.py)

```powershell
python tool/arrow_to_json.py data/train.arrow -o data/train.json
python tool/arrow_to_json.py data/train.arrow -o data/train.jsonl --jsonl
```

---

### 4.5 CSV 流式预处理（大文件防 OOM）

脚本：[tool/prepare_cycle_data.py](tool/prepare_cycle_data.py)

该脚本内部使用常量 `CSV_FILE_PATH`，请先在文件中修改路径后再运行：

```powershell
python tool/prepare_cycle_data.py
```

默认会在当前目录产出：

- `raw_questions.jsonl`
- `raw_answers.jsonl`

---

## 5. 常用排查

### 5.1 生成结果为空

优先检查输入字段是否匹配：

- [code/A2Q/generate_pseudo_a.py](code/A2Q/generate_pseudo_a.py) 当前读取 `input`。
- 如果源数据字段不是 `input`，先用 [tool/convert_to_llamafactory_json.py](tool/convert_to_llamafactory_json.py) 转换。

### 5.2 GPU 未生效 / 很慢

- 确认 CUDA 可用：`torch.cuda.is_available()`
- 检查 [tool/model_loader.py](tool/model_loader.py) 中 `MODEL_PATH` 与量化配置
- Linux CUDA 环境下优先 `4bit` 量化以降低显存压力

### 5.3 语法快速检查

```powershell
python -m py_compile tool/model_loader.py
python -m py_compile code/A2Q/generate_pseudo_a.py
python -m py_compile test_model/test.py
```

---

## 6. 备注

- 目前模型加载参数是“集中固定配置”模式，适合单机/单模型流程。
- 如果后续要支持多模型切换，可在 [tool/model_loader.py](tool/model_loader.py) 增加环境变量覆盖逻辑。

