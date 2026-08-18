# Optimization

Judge signature optimization for maximizing human-LLM alignment.

This package uses DSPy's built-in optimizers to improve the evaluation judge signatures by incorporating human annotations as ground truth. It supports three optimization strategies and tracks version history for reproducible comparison.

## Analysis Notebook

Optimization is integrated into the end-to-end alignment analysis notebook:

**[`human_annotation/human_alignment_analysis_multimodels.ipynb`](../human_annotation/human_alignment_analysis_multimodels.ipynb)**

The notebook covers training set construction, running optimizers, evaluating the optimized signatures against human annotations, and comparing versions.

## Modules

### `alignment_metric`

Provides a DSPy-compatible metric function that scores a judge's predictions against human ground truth.

**Key function:**

| Function | Description |
|----------|-------------|
| `make_alignment_metric(target_metrics, weights)` | Returns a function with signature `metric(example, prediction, trace=None) -> float`. Compares predicted categorical labels to human labels with partial credit: 1.0 for exact match, 0.5 for adjacent category, 0.0 for far apart. NOT APPLICABLE handled separately. |

### `signature_optimizer`

Orchestrates the full optimization pipeline: training set construction, optimizer execution, evaluation, and model persistence.

**Dataclasses:**

| Class | Fields |
|-------|--------|
| `OptimizationConfig` | `optimizer_type`, `judge_type`, `model_name`, `max_bootstrapped_demos`, `max_labeled_demos`, `temperature`, `num_trials`, `seed` |

**Key functions and classes:**

| Name | Description |
|------|-------------|
| `build_trainset(annotation_df, judge_type)` | Convert a long-format annotation DataFrame to a list of `dspy.Example` objects with categorical target labels. Uses `denormalize_score()` to convert numeric scores back to labels. |
| `build_trainset_from_paths(annotator_csv, llm_scores_csv)` | Convenience wrapper that loads annotation data and builds the trainset in one call. |
| `SignatureOptimizer` | Main class. Configures DSPy LM, creates the student module (ChainOfThought judge), and exposes `optimize()`, `evaluate()`, `save_optimized()`, and `load_optimized()` methods. |
| `JUDGE_METRICS` | Dict mapping judge type to its available metrics (e.g., `"comprehensive"` includes all seven metrics). |

**Supported optimizers:**

| Optimizer | Description | Recommended data size |
|-----------|-------------|----------------------|
| `labeled_fewshot` | Selects k human-annotated examples as few-shot demonstrations. Safe for small datasets. | 5+ examples |
| `bootstrap_fewshot` | Generates Chain-of-Thought demonstrations via a teacher model. More powerful but may overfit on small data. | 10+ examples |
| `miprov2` | Multi-stage instruction + demonstration optimization. Most powerful, but data-hungry. | 50+ examples |

### `versioning`

Tracks optimization history for reproducible comparisons across judge signature versions.

**Dataclass:**

| Class | Fields |
|-------|--------|
| `SignatureVersion` | `version_id`, `timestamp`, `judge_type`, `optimizer_type`, `n_training_examples`, `per_metric_kappa`, `overall_kappa`, `model_path`, `notes`, `config` |

**Key functions:**

| Function | Description |
|----------|-------------|
| `SignatureVersion.from_alignment_results(...)` | Create a version record from alignment results with automatic kappa extraction. |
| `save_version_history(versions, path)` | Persist version list to JSON. |
| `load_version_history(path)` | Reload version list from JSON. |
| `version_comparison_table(versions)` | Create a comparison DataFrame (Table 6) with per-metric and overall kappa. |

## Usage

```python
from langdmta_eval.alignment import load_multiple_annotators
from langdmta_eval.optimization import (
    OptimizationConfig,
    SignatureOptimizer,
    SignatureVersion,
    build_trainset,
    version_comparison_table,
)

# Load annotation data
df = load_multiple_annotators(annotator_csvs, llm_scores_csv)

# Build training set from one annotator (DSPy needs one label per example)
trainset = build_trainset(df[df["annotator_id"] == "LS"], judge_type="comprehensive")

# LabeledFewShot optimization
config = OptimizationConfig(
    optimizer_type="labeled_fewshot",
    judge_type="comprehensive",
    max_labeled_demos=4,
)
optimizer = SignatureOptimizer(config, trainset)
optimized = optimizer.optimize()

# Evaluate and save
score = optimizer.evaluate(optimized)
optimizer.save_optimized(optimized, "optimized_judge.json")
# Saves a JSON file containing the optimized signature (few-shot demos / instructions).
# This file can be passed directly to run_judge.py via --optimized-path:
#
#   python run_judge.py results.csv /path/to/model \
#       --optimized-path optimized_judge.json

# Track versions
version = SignatureVersion.from_alignment_results(
    version_id="v2_labeled_fewshot",
    results=post_optimization_results,
    judge_type="comprehensive",
    optimizer_type="labeled_fewshot",
    n_training_examples=len(trainset),
)
print(version_comparison_table([baseline_version, version]).round(3))
```

## Dependencies

- `dspy-ai` -- LabeledFewShot, BootstrapFewShot, MIPROv2 optimizers
- `langdmta_eval.alignment` -- data loading and agreement metrics
- `langdmta_eval.judges` -- judge signatures and score mappings
