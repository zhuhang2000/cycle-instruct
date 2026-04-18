# Experiments

Independent scaffolding for running **6 downstream benchmarks × N data-construction baselines**, then aggregating everything into paper-ready tables and figures. The package depends on `tool/` and `code/iterative/` but is **not** depended on by them — delete the directory and the training pipeline keeps working.

## Layout

```
experiments/
  types.py                # shared dataclasses + JSON helpers
  eval/
    runner.py             # CLI: run one method across N benchmarks
    benchmarks/           # 6 plugins: vqav2/gqa/mmbench/pope/docvqa/hallusion
  baselines/
    runner.py             # CLI: prepare → train → evaluate → write MethodRun
    *.py                  # 6 preparers: no_filter/random_k/clip_only/...
  intrinsic/              # data-pool diagnostics (diversity / hallucination / cycle-score stats)
  analysis/               # aggregators: main_table / ablation / iteration_curves / efficiency / threshold sweep / human eval
  configs/                # YAML specs for main table / ablations / iteration eval / θ sweep
```

## Quickstart

### 1. Smoke-test on CPU (no GPU, no real model)

Both CLIs honour `--smoke` (and `CI_SKIP_HEAVY=1`), which swaps in stub
inference + stub training hooks:

```bash
bash bash/run_eval.sh --smoke \
    --benchmark pope \
    --data-path tests/fixtures/pope_tiny.jsonl \
    --image-dir tests/fixtures/images \
    --output-dir /tmp/eval_smoke

bash bash/run_baselines.sh --smoke \
    --spec experiments/configs/main_table.yaml \
    --only no_filter,clip_only_0p25
```

### 2. Full run (GPU required)

```bash
# One method across the main table's 6 benchmarks:
bash bash/run_eval.sh \
    --spec experiments/configs/main_table.yaml \
    --method ours_round5 \
    --model-path /path/to/merged_model \
    --output-dir runs/experiments/main_table_v1/ours_round5

# Entire main table (prepares data, trains each baseline via LlamaFactory,
# then evaluates on all 6 benchmarks):
bash bash/run_baselines.sh --spec experiments/configs/main_table.yaml
```

### 3. Aggregate into paper tables

```bash
bash bash/run_analysis.sh runs/experiments/main_table_v1
# → runs/experiments/main_table_v1/paper_tables/{main_table.md,csv,tex, summary.json}

python -m experiments.analysis.iteration_curves  runs/iterative/latest
python -m experiments.analysis.threshold_sweep   runs/experiments/thresholds_sweep_v1
```

## Extending

Adding a new benchmark:

1. Create `experiments/eval/benchmarks/<name>.py`, subclass `BaseEvaluator`, implement `load_examples / build_prompt / score`.
2. Decorate with `@register_benchmark("<name>")`.
3. Add it to `_safe_import(...)` in `experiments/eval/benchmarks/__init__.py`.
4. Reference it in any YAML spec.

Adding a new baseline: same pattern with `BaseDataPreparer` / `@register_baseline`.

## Design invariants

- **Registry first**: no `if name == ...` branches in runner code.
- **Mock-friendly hooks**: the training and MLLM inference callables are injectable, so orchestration is unit-testable on CPU.
- **JSON-in / JSON-out**: each stage writes self-contained JSON artifacts; aggregators never call runners.
- **No reverse deps**: `experiments/` imports from `tool/` and `code/iterative/`, not the other way around.

## Tests

```bash
pytest tests/test_experiments/ -v
# or with coverage:
pytest tests/test_experiments/ --cov=experiments --cov-report=term-missing
```
