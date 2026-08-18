# Alignment

Human-LLM alignment analysis for evaluation judge signatures.

This package computes formal agreement metrics between human annotators and LLM judges. It supports multiple LLM judges simultaneously, enabling cross-model comparisons and inter-LLM reliability analysis alongside the standard per-annotator and majority-vote human alignment metrics. It also measures inter-annotator reliability and self-consistency to contextualize human-LLM disagreements.

## Analysis Notebook

All end-to-end analysis — loading data, computing alignment and inter-annotator agreement, multi-model comparisons, inter-LLM reliability, and generating publication figures — is implemented in:

[`human_annotation/human_alignment_analysis_multimodels.ipynb`](../human_annotation/human_alignment_analysis_multimodels.ipynb)

## Modules

### `data_loader`

Loads and merges human annotation CSVs with LLM judge scores. Supports multiple annotators per case, multiple judge models, and produces long-format DataFrames suitable for downstream analysis.

**Key functions:**

| Function | Description |
|----------|-------------|
| `load_annotation_pair(annotator_csv, llm_scores_csv)` | Load one annotator sheet and merge with LLM scores on `annotation_id` (CSV) or `trace_id` (JSON). Filters incomplete rows and normalizes categorical labels to numeric scores. |
| `load_multiple_annotators(annotator_csvs, llm_scores_csv)` | Load annotations from multiple annotators (e.g., `{"LS": "path/to/ls.csv", "RMO": "path/to/rmo.csv"}`) into a single long-format DataFrame. |
| `load_multiple_models(annotator_csvs, model_llm_scores)` | Load annotations paired with LLM scores from multiple judge models. Adds a `model_id` column and warns about non-shared annotation IDs across models. |
| `load_all_annotations(annotation_pairs)` | Load and concatenate multiple annotation pairs from different datasets. |
| `get_paired_scores(df, metric)` | Extract paired `(human, llm)` score arrays for a single metric. Optionally filters NOT APPLICABLE (`-1.0`) scores and specific annotators. |
| `get_annotator_scores_wide(df, metric)` | Pivot to wide format (annotators as columns) for inter-annotator agreement computation. |
| `get_majority_human_scores(df, metric)` | Compute majority-vote human scores across annotators and pair with LLM scores. |
| `get_duplicate_pairs(df, metric)` | Extract paired scores for duplicate questions (original vs duplicate) for measuring self-consistency. |
| `drop_duplicates(df)` | Remove duplicate rows (those with a non-null `duplicate_of` value). |

**Constants:**

- `ANNOTATION_METRICS_V1`: Six V1 metrics -- `correctness`, `output_relevancy`, `output_clarity`, `overall_solvedness`, `capability_awareness`, `boldness`.
- `ANNOTATION_METRICS_V2`: Four V2 metrics -- `relevancy`, `completeness`, `structural_clarity`, `scope_adherence`.
- `ANNOTATION_METRICS_BY_VERSION`: Dict mapping `"v1"` / `"v2"` to their respective metric lists.
- `ANNOTATION_METRICS`: Defaults to V2 metrics.

### `metrics`

Computes agreement statistics between human and LLM scores, inter-annotator reliability, and self-consistency.

**Dataclasses:**

| Class | Fields |
|-------|--------|
| `AlignmentResult` | `metric_name`, `n_samples`, `exact_match_rate`, `cohens_kappa_unweighted`, `cohens_kappa_weighted`, `spearman_correlation`, `spearman_pvalue`, `confusion_matrix`, `labels` |
| `InterAnnotatorResult` | `metric_name`, `n_samples`, `n_annotators`, `fleiss_kappa`, `mean_pairwise_kappa`, `pairwise_kappas` |
| `SelfConsistencyResult` | `annotator_id`, `metric_name`, `n_pairs`, `exact_match_rate`, `mean_absolute_difference`, `cohens_kappa_weighted` |

**Key functions:**

| Function | Description |
|----------|-------------|
| `compute_alignment(human, llm, metric_name)` | Compute all agreement metrics for one evaluation metric. Returns `AlignmentResult`. |
| `compute_all_alignments(df)` | Compute alignment for all metrics. Supports `annotator_id` filtering, `use_majority_vote`, and optional `overall` pooled entry. |
| `compute_all_alignments_by_model(df)` | Compute alignment for each model in a multi-model DataFrame. Returns `{model_id: {metric: AlignmentResult}}`. |
| `multi_model_summary_table(all_results)` | Format multi-model results as a single summary DataFrame with a `model` column. |
| `alignment_summary_table(results)` | Format results as a publication-ready DataFrame (Table 5). |
| `compute_inter_annotator_agreement(df, metric)` | Fleiss' kappa and mean pairwise weighted Cohen's kappa for one metric. |
| `compute_all_inter_annotator(df)` | Inter-annotator agreement for all metrics. |
| `inter_annotator_summary_table(results)` | Format inter-annotator results as a DataFrame. |

**Edge case handling:** Returns `np.nan` for undefined cases (n < 2, constant arrays, zero-variance). Converts float scores to string labels internally for sklearn compatibility.

### `visualization`

Publication-quality plots for alignment analysis, inter-annotator agreement, and self-consistency.

| Function | Description |
|----------|-------------|
| `plot_confusion_matrix(result)` | Single confusion matrix heatmap (rows = human, columns = LLM). Supports row-normalization. |
| `plot_all_confusion_matrices(results)` | Grid of per-metric confusion matrices (Figure 4). |
| `plot_alignment_bar_chart(results)` | Bar chart of alignment scores per metric with threshold lines (moderate=0.4, substantial=0.6). |
| `plot_alignment_comparison(before, after)` | Grouped bar chart comparing two conditions (e.g., baseline vs. optimized, Figure 5). |
| `plot_multi_alignment_comparison(results_dict)` | Multi-version grouped bar chart (Table 6 visual). |
| `plot_self_consistency_barplot(data, metric)` | Bar plot of self-consistency scores grouped by evaluation metric with 95% CIs. |
| `plot_inter_annotator_barplot(results)` | Bar plot of inter-annotator agreement (Fleiss' kappa and mean pairwise kappa). |

All plot functions accept an optional `save_path` argument and return a `matplotlib.Figure`.

## Usage

```python
from langdmta_eval.alignment import (
    load_multiple_annotators,
    load_multiple_models,
    compute_all_alignments,
    compute_all_alignments_by_model,
    compute_all_inter_annotator,
    alignment_summary_table,
    multi_model_summary_table,
    inter_annotator_summary_table,
    plot_all_confusion_matrices,
    plot_multi_alignment_comparison,
)

# Load three annotators
annotators = {
    "LS": "path/to/annotated_LS.csv",
    "RMO": "path/to/annotated_RMO.csv",
    "SG": "path/to/annotated_SG.csv",
}
df = load_multiple_annotators(annotators, "path/to/llm_scores.json")

# Inter-annotator agreement
iaa = compute_all_inter_annotator(df)
print(inter_annotator_summary_table(iaa).round(3))

# Human-LLM alignment (majority vote)
results = compute_all_alignments(df, use_majority_vote=True)
print(alignment_summary_table(results).round(3))

# Confusion matrices
fig = plot_all_confusion_matrices(results, save_path="figure4.pdf")

# Multi-model comparison
model_scores = {"gpt-5": "scores_gpt5.json", "claude-4": "scores_claude4.json"}
multi_df = load_multiple_models(annotators, model_scores)
all_results = compute_all_alignments_by_model(multi_df)
print(multi_model_summary_table(all_results).round(3))
fig = plot_multi_alignment_comparison(all_results, save_path="model_comparison.pdf")
```

## Dependencies

- `scikit-learn` -- Cohen's kappa, confusion matrix
- `scipy` -- Spearman rank correlation
- `seaborn` / `matplotlib` -- visualization
- `pandas` / `numpy` -- data manipulation
