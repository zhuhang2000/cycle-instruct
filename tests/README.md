# tests 运行说明

这份文档说明 `tests/` 目录下的测试文件什么时候运行。它们主要用于在修改代码后快速检查参数、数据格式、路径注册、mock 流水线和评测逻辑是否被改坏。除非测试名或说明特别指出，这些测试都是轻量测试，不会真正启动大模型推理、GPU LoRA 微调或完整评测。

## 快速命令

在仓库根目录 `D:\code\VSCode\cycle-instruct` 执行：

```powershell
python -m pytest tests -q
```

按模块运行：

```powershell
python -m pytest tests/test_i2qa -q
python -m pytest tests/test_iterative -q
python -m pytest tests/test_experiments -q
```

按单个文件运行：

```powershell
python -m pytest tests/test_i2qa/test_verify_cycle_consistency.py -q
```

## 推荐运行时机

### 1. 小改动后

只运行最相关的测试文件。这样最快，适合刚改完一个函数或一个参数名时检查是否有明显错误。

示例：只改了 `code/I2QA/verify_cycle_consistency.py`：

```powershell
python -m pytest tests/test_i2qa/test_verify_cycle_consistency.py -q
```

### 2. 跨模块改动后

运行对应目录下所有测试。

示例：同时改了 `code/I2QA/` 和 `tool/multimodal_types.py`：

```powershell
python -m pytest tests/test_i2qa -q
```

示例：改了循环训练、数据混合、LLaMAFactory 参数：

```powershell
python -m pytest tests/test_iterative -q
```

### 3. 准备跑真实生成或 LoRA 微调前

建议至少运行：

```powershell
python -m pytest tests/test_i2qa tests/test_iterative -q
```

原因是这两组测试覆盖多模态 QA 生成/过滤、Image+A -> Q' 重构、循环轮次、LLaMAFactory `dataset_dir`、`dataset_info.json`、`template` 和模型路径参数。真实训练成本高，先用测试排除参数和路径错误更划算。

### 4. 提交代码或长时间实验前

运行全量测试：

```powershell
python -m pytest tests -q
```

全量测试用于确认 I2QA、iterative pipeline、实验评测、baseline、intrinsic analysis 都没有明显回归。

## 测试文件对应关系

| 修改内容 | 建议运行 |
| --- | --- |
| `code/I2QA/filter_and_export.py`、`tool/multimodal_types.py`、过滤导出格式、`cycle_scores` 保留逻辑 | `python -m pytest tests/test_i2qa/test_filter_and_export.py -q` |
| `code/I2QA/verify_cycle_consistency.py`、Image+Q -> A'、Image+A -> Q'、cycle consistency verifier | `python -m pytest tests/test_i2qa/test_verify_cycle_consistency.py -q` |
| `code/iterative/data_mixer.py`、训练数据混合、去重、历史池、`dataset_info.json` 注册 | `python -m pytest tests/test_iterative/test_data_mixer.py -q` |
| `code/iterative/iterative_trainer.py`、生成/过滤子进程、LoRA 参数、`--model_path`、`--template`、`--dataset_dir` | `python -m pytest tests/test_iterative/test_iterative_trainer.py -q` |
| 循环训练整体逻辑、每轮输入输出、历史池、上一轮 merged model 作为下一轮 generator | `python -m pytest tests/test_iterative/test_iterative_smoke.py -q` |
| `code/iterative/metrics.py`、每轮指标保存/加载、early stop 规则 | `python -m pytest tests/test_iterative/test_metrics.py -q` |
| `code/iterative/qa_templates.py`、QA 类型模板、类型分类、多样性分布、重采样 | `python -m pytest tests/test_iterative/test_qa_templates.py -q` |
| `code/iterative/round_config.py`、每轮学习率、epoch、LoRA rank、CLI override | `python -m pytest tests/test_iterative/test_round_config.py -q` |
| `experiments/baselines/`、baseline 数据准备、baseline runner | `python -m pytest tests/test_experiments/test_baseline_runner_smoke.py tests/test_experiments/test_ours_baseline.py -q` |
| baseline 训练 hook 调用脚本参数 | `python -m pytest tests/test_experiments/test_train_hook.py -q` |
| `experiments/eval/runner.py`、默认推理 hook、推理结果落盘 | `python -m pytest tests/test_experiments/test_eval_runner.py -q` |
| POPE 评分、yes/no 解析、F1/recall/yes_ratio | `python -m pytest tests/test_experiments/test_pope_scoring.py -q` |
| MMBench 选择题解析和按类别评分 | `python -m pytest tests/test_experiments/test_mmbench_choice_extract.py -q` |
| DocVQA ANLS 评分 | `python -m pytest tests/test_experiments/test_docvqa_anls.py -q` |
| benchmark/baseline 注册表 | `python -m pytest tests/test_experiments/test_registry.py -q` |
| 实验结果聚合表、ablation 表格 | `python -m pytest tests/test_experiments/test_aggregator.py -q` |
| 人工评测 CSV 分析 | `python -m pytest tests/test_experiments/test_human_eval.py -q` |
| intrinsic analysis 全部模块 | `python -m pytest tests/test_experiments/test_intrinsic -q` |

## 目录说明

### `tests/test_i2qa`

检查多模态 QA 数据生成链条中的关键接口。

- `test_filter_and_export.py`：确认过滤后的 ShareGPT 数据仍保留 `cycle_scores`、`cycle_score`、`image_id`、`generation_model` 等元信息。
- `test_verify_cycle_consistency.py`：确认问题重构使用的是 `Image + A -> Q'`，而不是纯文本 `A -> Q'`，并且 verifier 模型路径传递正确。

### `tests/test_iterative`

检查循环训练控制器。重点不是训练效果，而是每轮数据、路径、参数和 hook 是否正确。

- `test_data_mixer.py`：测试 seed/new/historical 数据混合、去重、历史池质量过滤、LLaMAFactory 数据注册。
- `test_iterative_trainer.py`：测试生成过滤三阶段 CLI、LoRA 训练参数、统一模型路径 alias、template 和 dataset_dir。
- `test_iterative_smoke.py`：用 fake hooks 跑 2 轮循环，验证 round 目录、metrics、mixed dataset、historical pool、上一轮 merged model 传递。
- `test_metrics.py`：测试指标读写和 early stop 条件。
- `test_qa_templates.py`：测试 QA 类型模板、分类、多样性和类型重平衡。
- `test_round_config.py`：测试每轮训练超参数 schedule。

### `tests/test_experiments`

检查实验、评测、baseline 和分析脚本。

- `test_baseline_runner_smoke.py`：mock train/infer，检查 baseline runner 输出目录和 `run.json`。
- `test_train_hook.py`：检查 baseline 默认训练 hook 调用 `bash/run_multimodal_cycle.sh` 的参数。
- `test_eval_runner.py`：检查默认推理函数是否写入真实 artifact 路径。
- `test_registry.py`：检查内置 benchmark 和 baseline 是否都注册成功，且重复注册会报错。
- `test_pope_scoring.py`、`test_mmbench_choice_extract.py`、`test_docvqa_anls.py`：检查具体 benchmark 的评分边界条件。
- `test_aggregator.py`：检查实验结果表格和 ablation delta。
- `test_human_eval.py`：检查人工评测 CSV 分析逻辑。
- `test_ours_baseline.py`：检查 ours baseline 从 iterative run 目录读取 round 数据。
- `test_intrinsic/`：检查 QA 类型分布、多样性、cycle score 统计、语言质量、幻觉指标和 report 输出。

### `tests/fixtures`

存放小型测试数据，例如 `tiny_vqa.jsonl`。这些数据用于 intrinsic report 等轻量测试，不是正式训练数据。

## 这些测试不能替代什么

这些测试不能保证：

- 大模型真实生成质量好。
- CUDA、显存、LLaMAFactory 安装环境一定正确。
- 真实图片路径和大规模数据集一定完整。
- LoRA 微调一定能在当前机器上跑完。
- 训练后的模型在真实 benchmark 上一定提升。

因此，测试通过后，正式实验前仍建议先做一次小规模真实 smoke run，例如只用少量图片生成 QA，再用很小的训练步数验证 LLaMAFactory 能启动。

## Windows/PowerShell 注意事项

本仓库含中文和 Markdown 文本。Windows 下如果读取文件出现中文乱码、NUL 字符或类似 `W�i�n�d�o�w�s�` 的输出，优先按编码问题处理，不要直接判断代码失败。

运行 pytest 时通常不需要额外设置编码；如果你在 PowerShell 里查看中文文件，建议显式使用 UTF-8：

```powershell
Get-Content -Encoding UTF8 tests/README.md
```
