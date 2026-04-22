# Intrinsic Evaluation Summary

## 概述

这一组代码实现了 Section 11 所需的 **intrinsic evaluation** 能力，用来从数据层面对生成后的 VQA 样本池做质量分析，而不是只看外部 benchmark 分数。

实现的目标可以概括为四件事：

1. 用统一接口组织多种 intrinsic metric，并支持按名字注册和调度。
2. 对 VQA JSONL 数据做共享读取、字段抽取和格式兼容。
3. 从多个维度评估样本池，包括题型分布、多样性、幻觉、cycle score 统计、语言质量和图文对齐。
4. 用一份 YAML 配置驱动整套评估，最终产出 JSON、Markdown 和图表。

当前实现主要位于 `experiments/intrinsic/`，默认配置在 `experiments/configs/intrinsic_default.yaml`，脚本入口是 `bash/run_intrinsic.sh`。

## 整体结构

### 1. 统一指标接口与注册机制

- `base.py`
  定义 `IntrinsicMetric` 抽象基类，约定所有模块都实现 `compute(samples, **ctx)`，并可选实现 `plots(result, out_dir)`。
- `METRIC_REGISTRY`
  所有指标模块通过 `@register_metric("name")` 注册到全局表中，聚合器可以按名字动态发现并运行模块。
- 指标元信息
  基类上声明了 `requires_images`、`requires_cycle_scores`、`requires_gpu` 等标记，便于说明模块依赖。

### 2. 共享 I/O 与格式兼容

- `_io.py`
  提供 JSONL 读取与通用字段抽取：
  - `load_vqa_jsonl`
  - `extract_question`
  - `extract_answer`
  - `extract_image_path`
  - `save_json`
  - `save_md_table`
- 兼容两类样本格式：
  - 直接包含 `question` / `answer` / `image_path` 的字典
  - ShareGPT 风格 `messages` / `images` 结构

### 3. 自动导入内置模块

- `__init__.py`
  导入包时会尝试安全加载各个内置指标模块，从而完成注册。即使某个模块因为可选依赖缺失而导入失败，也不会让整个包直接崩掉。

## 已实现的指标能力

### 1. `qa_type_stats.py`

用于分析 **问题类型分布是否健康**，核心能力包括：

- 将问题划分为 6 类：
  - `objects`
  - `spatial`
  - `actions`
  - `counting`
  - `text_ocr`
  - `reasoning`
- 复用 `code.iterative.qa_templates` 中的分类与分布逻辑，保证和生成端的类型重平衡逻辑一致。
- 统计：
  - 各题型占比
  - `diversity_score`
  - 低于阈值的欠代表类型
  - 各题型下的 `cycle_scores.composite` 均值和通过率
- 如果提供 `seed_ref`，还会计算：
  - seed 集的题型分布
  - 当前样本池与 seed 分布的 `JS divergence`
- 可选输出柱状图 `qa_type_distribution.png`

这一模块回答的是：**样本池的题型是否均衡，是否偏离 seed 分布，以及哪些题型更容易拿到高 cycle score。**

### 2. `diversity.py`

用于分析 **文本和图像层面的多样性**，核心能力包括：

- 文本多样性：
  - `distinct-1/2/3`，分别对问题和答案统计
  - `type_token_ratio`
  - `MTLD`
  - 问题和答案长度标准差
- 文本重复度：
  - `Self-BLEU-4`
  - 通过随机采样文本对，避免全量两两比较的平方复杂度
- 可选语义多样性：
  - 如果提供 `embedding_model`，使用 `sentence-transformers` 计算 embedding spread
- 可选图像多样性：
  - 如果启用 `with_images`，计算基于 8x8 均值哈希的 `pHash` 唯一率

这一模块回答的是：**问题和答案是否过于模板化，是否有大量语义近似或重复样本，图像侧是否也存在明显重复。**

### 3. `hallucination.py`

用于分析 **答案里的实体幻觉**，核心能力包括：

- 从答案中抽取候选名词：
  - 优先用 spaCy noun chunks
  - 缺依赖时回退到基于字母 token 的轻量启发式
- 使用可插拔检测器验证图像中是否真的存在这些实体：
  - 默认提供 OWL-ViT v2 检测器构造器
  - 测试中也可以直接传入 mock detector
- 计算经典 CHAIR 指标：
  - `CHAIRi` = 幻觉 mention 数 / 总 mention 数
  - `CHAIRs` = 出现过至少一个幻觉 mention 的答案数 / 总答案数
- 输出粒度：
  - 总体 CHAIRi / CHAIRs
  - 每个样本的幻觉情况
  - 每个类别词的幻觉统计
- 可选 CLIP 交叉检查：
  - 计算低图文对齐率样本占比，辅助发现 answer-level mismatch

这一模块回答的是：**模型是否在答案中凭空“看见”了图里没有的物体或概念。**

### 4. `cycle_score_stats.py`

用于分析 **已有 cycle_scores 字段的统计特征**，核心能力包括：

- 支持统计以下分量：
  - `ar`
  - `clip`
  - `qr`
  - `ppl`
  - `composite`
- 对每个分量输出：
  - 样本数
  - 均值 / 标准差
  - 最小值 / 最大值
  - percentiles：`p25/p50/p75/p90/p95/p99`
  - 直方图
- 对 composite 输出：
  - 不同阈值下的通过率曲线
- 计算组件之间的 Pearson 相关矩阵
- 可选输出图表：
  - `composite_score_hist.png`
  - `pass_rate_vs_threshold.png`

这一模块回答的是：**cycle score 的整体分布怎样，不同阈值下会筛掉多少样本，各子分量之间是否强相关。**

### 5. `linguistic_quality.py`

用于分析 **语言层面的表面质量与模板化程度**，核心能力包括：

- 问题和答案长度统计：
  - 均值
  - 标准差
  - 最小值 / 最大值
- 模板重复率：
  - 以前 3 个 token 的共享前缀衡量模板化程度
- 答案形态统计：
  - `yes_no_answer_rate`
  - `sentence_shape_rate`
- 可选语法检查：
  - 如果启用 `run_grammar_check` 且环境具备 `language_tool_python`，会输出语法错误率

这一模块回答的是：**答案是否过短、过于模板化、是否大量退化成 yes/no、句子表面形态是否正常。**

### 6. `alignment.py`

用于分析 **图像与问题/答案的语义对齐**，核心能力包括：

- 调用 `tool.cycle_scorer.clip_similarity_batch` 计算：
  - `CLIP(image, answer)`
  - `CLIP(image, question)`
- 计算 `blind_caption_rate`
  - 当 `CLIP(image, answer)` 低于阈值时，认为是可能“盲答”或弱对齐
- 通过打乱答案重新配对，估计：
  - `mi_shuffle_estimate`
  - 近似反映真实配对比随机配对多带来的图文互信息

这一模块回答的是：**答案是否真的依赖图像内容，而不是只靠语言先验在“盲写”。**

## 聚合与报告

### `report.py`

这是整套 intrinsic evaluation 的总入口，负责：

- 读取 YAML 配置
- 加载样本与可选的 `seed_ref`
- 解析要启用的模块
- 逐个实例化并运行注册表中的 metric
- 捕获单模块异常，避免一个模块失败导致整套评估中断
- 统一写出结果

输出产物包括：

- `intrinsic_report.json`
  保存完整机器可读结果
- `intrinsic_report.md`
  保存适合人工阅读的汇总
- `plots/*.png`
  保存各模块生成的图表

Markdown 报告里目前会汇总：

- QA type distribution
- Diversity
- Hallucination
- Cycle score percentiles
- Linguistic quality
- Image-text alignment

### `experiments/configs/intrinsic_default.yaml`

默认配置定义了：

- 输入样本路径
- 图像目录
- seed 参考集路径
- 输出目录
- 运行设备
- 每个模块的启用状态和参数

当前默认配置中：

- `qa_types`、`diversity`、`hallucination`、`cycle_stats`、`linguistic` 默认启用
- `alignment` 默认关闭

### `bash/run_intrinsic.sh`

提供最薄的一层 shell 包装，最终执行：

```bash
python -m experiments.intrinsic.report "$@"
```

这样可以直接把命令行参数透传给 `report.py`。

## 运行方式

### 完整运行

```bash
bash bash/run_intrinsic.sh \
  --config experiments/configs/intrinsic_default.yaml
```

### 直接指定输入输出

```bash
python -m experiments.intrinsic.report \
  --input runs/round_3/filtered.jsonl \
  --image-dir data/raw/cc3m/images \
  --seed-ref data/seed/llava_instruct.jsonl \
  --out report/round_3
```

### Smoke 模式

```bash
python -m experiments.intrinsic.report \
  --smoke \
  --input tests/fixtures/tiny_vqa.jsonl \
  --out .tmp/intrinsic_smoke
```

`--smoke` 模式会：

- 只取前 100 条样本
- 自动跳过 `hallucination` 和 `alignment`
- 适合无 GPU 或只想快速检查流水线时使用

## 可选依赖与降级行为

这套实现不是“所有依赖都装好才能跑”，而是尽量做了 best-effort 降级：

- `matplotlib`
  缺失时不生成图，但 JSON/Markdown 仍可产出
- `sentence-transformers`
  缺失时无法算 embedding diversity
- `PIL`
  缺失时无法算图像 pHash 多样性
- `spaCy` / `en_core_web_sm`
  缺失时回退到简化名词抽取
- `transformers` + `torch`
  缺失时 OWL-ViT 检测器不可用
- `tool.cycle_scorer.clip_similarity_batch`
  不可用时 alignment 或 CLIP cross-check 会返回错误信息而非直接崩溃
- `language_tool_python`
  缺失时不输出 grammar error rate

因此它既支持在轻量环境下跑核心统计，也支持在依赖齐全时启用更重的图像与语义评估。

## 测试与验证

测试覆盖位于 `tests/test_experiments/test_intrinsic/`，共 31 个测试，覆盖：

- QA type 统计
- diversity 计算
- hallucination 逻辑
- cycle score 统计
- linguistic quality
- report 聚合与集成流程

辅助测试数据在 `tests/fixtures/tiny_vqa.jsonl`，包含 8 条样本，并覆盖全部 6 类 QA type。

本地验证结果：

- `pytest -q tests/test_experiments/test_intrinsic`
  - 31 passed
- `python -m experiments.intrinsic.report --smoke --input tests/fixtures/tiny_vqa.jsonl --out .tmp/intrinsic_smoke`
  - 成功生成 `intrinsic_report.json` 与 `intrinsic_report.md`

## 总结

这部分实现本质上把“生成数据的内在质量分析”补齐成了一条可复用流水线：

- 有统一接口和注册机制，方便继续扩展新指标
- 有 YAML 驱动的批量运行方式，便于实验复现
- 有 JSON / Markdown / 图表三类输出，兼顾程序消费和人工阅读
- 有轻重两档运行模式，既能快速 smoke，也能在依赖齐全时做更完整的图文质量分析

如果后续要继续扩展，最直接的方式是新增一个继承 `IntrinsicMetric` 的模块，注册到 `METRIC_REGISTRY`，然后在 YAML 里打开对应配置即可。
