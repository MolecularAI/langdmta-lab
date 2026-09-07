"""Extract a stratified sample of evaluation results for human annotation.

Produces an annotation-ready CSV that mirrors the context provided to the LLM
judge (minus question_variation), with columns for annotators to fill in scores.

Sampling strategy:
- Stratified by LLM score bucket (low / middle / high) to capture the full range
- Biased toward complex, multi-tool workflows where LLM judges are least reliable
- Includes calibration anchors (simple, unambiguous cases) for annotator QC
- Ensures diversity across test_name, category, and question_variation
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langdmta_eval import EvaluationResult, TestCase, load_results_from_json

# ---------------------------------------------------------------------------
# Metrics the annotators will score (same labels as the LLM judge)
# ---------------------------------------------------------------------------

ANNOTATION_METRICS = [
    "correctness",
    "relevancy",
    "completeness",
    "structural_clarity",
    "ethical_awareness",
    "scope_adherence",
]


# Categories considered "complex" (multi-tool workflows)
COMPLEX_TEST_NAMES = {
    "DesignPropertySynthesis",
    "DesignProperty",
    "DesignSynthesis",
}

# Score bucket boundaries (on the 0-1 normalized scale)
SCORE_BUCKETS = {
    "low": (None, 0.25),  # 0.0
    "middle": (0.25, 0.75),  # 0.5
    "high": (0.75, None),  # 1.0
}


def _bucket(score: float) -> str:
    """Assign a score to a bucket."""
    if score <= 0.25:
        return "low"
    elif score <= 0.75:
        return "middle"
    else:
        return "high"


def _metric_variance(scores: dict, annotation_metrics: list) -> float:
    """Compute variance across metrics for a single result (higher = more disagreement between dimensions)."""
    values = [
        v
        for k, v in scores.items()
        if k in annotation_metrics and v is not None and v >= 0
    ]
    if len(values) < 2:
        return 0.0
    return float(np.var(values))


def results_to_records(
    results: List[EvaluationResult],
    annotation_metrics: list,
) -> pd.DataFrame:
    """Convert evaluation results into a flat DataFrame for sampling."""
    metrics_set = set(annotation_metrics)
    records = []
    for i, r in enumerate(results):
        tc = r.test_case
        avg_score = np.mean(
            [
                v
                for k, v in r.scores.items()
                if k in metrics_set and v is not None and v >= 0
            ]
        )
        record = {
            "original_index": i,
            "question": tc.question,
            "agent_output": tc.agent_output,
            "evaluation_context": tc.note or "",
            "tool_calls": str(tc.tool_calls),
            "expected_tool_calls": str(tc.expected_tool_calls.get("complete", [])),
            "accepted_tool_calls": str(tc.expected_tool_calls.get("accepted", [])),
            "category": tc.category,
            "test_name": tc.test_name,
            "question_variation": tc.question_variation,
            "trace_id": tc.trace_id or "",
            "session": tc.session,
            **{f"llm_{m}": r.scores.get(m) for m in annotation_metrics},
            "llm_avg_score": avg_score,
            "score_bucket": _bucket(avg_score),
            "metric_variance": _metric_variance(r.scores, annotation_metrics),
            "is_complex": tc.test_name in COMPLEX_TEST_NAMES,
            "n_tools_expected": (
                len(tc.expected_tool_calls.get("complete", [[]])[0])
                if tc.expected_tool_calls.get("complete")
                else 0
            ),
        }
        records.append(record)
    return pd.DataFrame(records)


def sample_stratified(
    df: pd.DataFrame,
    n_total: int = 30,
    complex_ratio: float = 0.6,
    calibration_ratio: float = 0.20,
    bucket_weights: dict = None,
    seed: int = 10,
) -> pd.DataFrame:
    """Draw a stratified sample biased toward complex and middle-score cases.

    Parameters
    ----------
    df : DataFrame from results_to_records
    n_total : target sample size
    complex_ratio : fraction of sample from complex multi-tool workflows
    calibration_ratio : fraction of sample reserved for easy calibration anchors
    bucket_weights : weight per score bucket (low/middle/high). Defaults to {low: 0.25, middle: 0.50, high: 0.25}
    seed : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    if bucket_weights is None:
        bucket_weights = {"low": 0.25, "middle": 0.50, "high": 0.25}

    n_calibration = max(1, int(n_total * calibration_ratio))
    n_complex = int((n_total - n_calibration) * complex_ratio)
    n_simple = n_total - n_calibration - n_complex

    sampled_indices = set()

    # --- 1. Calibration anchors: simple cases with high LLM scores ---
    calibration_pool = df[
        (~df["is_complex"])
        & (df["score_bucket"] == "high")
        & (df["metric_variance"] < 0.05)
    ]
    if len(calibration_pool) > 0:
        cal_idx = rng.choice(
            calibration_pool.index,
            size=min(n_calibration, len(calibration_pool)),
            replace=False,
        )
        sampled_indices.update(cal_idx)

    # --- 2. Complex cases, stratified by score bucket ---
    complex_pool = df[(df["is_complex"]) & (~df.index.isin(sampled_indices))]
    for bucket, weight in bucket_weights.items():
        bucket_pool = complex_pool[complex_pool["score_bucket"] == bucket]
        n_bucket = max(1, int(n_complex * weight))
        if len(bucket_pool) > 0:
            # Prefer cases with high metric variance (more interesting)
            probs = bucket_pool["metric_variance"].values + 1e-6
            probs = probs / probs.sum()
            chosen = rng.choice(
                bucket_pool.index,
                size=min(n_bucket, len(bucket_pool)),
                replace=False,
                p=probs,
            )
            sampled_indices.update(chosen)

    # --- 3. Simpler (non-complex) cases, stratified by score bucket ---
    simple_pool = df[(~df["is_complex"]) & (~df.index.isin(sampled_indices))]
    for bucket, weight in bucket_weights.items():
        bucket_pool = simple_pool[simple_pool["score_bucket"] == bucket]
        n_bucket = max(1, int(n_simple * weight))
        if len(bucket_pool) > 0:
            chosen = rng.choice(
                bucket_pool.index,
                size=min(n_bucket, len(bucket_pool)),
                replace=False,
            )
            sampled_indices.update(chosen)

    # --- 4. Top up if under target (greedily from remaining, prefer high variance) ---
    remaining = df[~df.index.isin(sampled_indices)]
    if len(sampled_indices) < n_total and len(remaining) > 0:
        n_more = n_total - len(sampled_indices)
        probs = remaining["metric_variance"].values + 1e-6
        probs = probs / probs.sum()
        extra = rng.choice(
            remaining.index,
            size=min(n_more, len(remaining)),
            replace=False,
            p=probs,
        )
        sampled_indices.update(extra)

    sample = df.loc[sorted(sampled_indices)].copy()

    # Shuffle so annotators don't see grouped patterns
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)

    return sample


def add_consistency_duplicates(
    sample: pd.DataFrame,
    n_duplicates: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Add duplicate rows from complex cases for intra-annotator consistency checks.

    Duplicates are selected from complex questions, weighted by metric variance.
    A `duplicate_of` column tracks which original each clone maps to.
    """
    if n_duplicates <= 0:
        sample["duplicate_of"] = np.nan
        return sample

    rng = np.random.default_rng(seed)
    complex_pool = sample[sample["is_complex"]]

    if len(complex_pool) == 0:
        sample["duplicate_of"] = np.nan
        return sample

    n_pick = min(n_duplicates, len(complex_pool))
    probs = complex_pool["metric_variance"].values + 1e-6
    probs = probs / probs.sum()

    chosen_idx = rng.choice(complex_pool.index, size=n_pick, replace=False, p=probs)
    duplicates = sample.loc[chosen_idx].copy()
    duplicates["duplicate_of"] = duplicates["original_index"]

    sample["duplicate_of"] = np.nan
    combined = pd.concat([sample, duplicates], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    return combined


def format_for_annotation(
    sample: pd.DataFrame, annotation_metrics: list
) -> pd.DataFrame:
    """Format the sample into the annotation-ready CSV.

    The output has:
    - Context columns (what annotators read)
    - Empty scoring columns (what annotators fill in)
    - LLM scores in separate columns (for later alignment analysis, hidden from annotators)
    """
    # Columns the annotator sees and evaluates
    context_cols = [
        "annotation_id",
        "question",
        "agent_output",
        "evaluation_context",
        "tool_calls",
        "expected_tool_calls",
        "accepted_tool_calls",
    ]

    # Empty columns for annotator to fill in
    for metric in annotation_metrics:
        sample[f"human_{metric}"] = ""

    sample["human_justification"] = ""

    # Metadata columns (for analysis, not shown to annotators)
    meta_cols = [
        "category",
        "test_name",
        "session",
        "trace_id",
    ]
    if "duplicate_of" in sample.columns:
        meta_cols.append("duplicate_of")

    llm_cols = [f"llm_{m}" for m in annotation_metrics] + [
        "llm_avg_score",
        "score_bucket",
        "metric_variance",
    ]

    human_cols = [f"human_{m}" for m in annotation_metrics] + ["human_justification"]

    sample["annotation_id"] = range(1, len(sample) + 1)

    output = sample[context_cols + human_cols + meta_cols + llm_cols].copy()
    return output


def print_sample_summary(sample: pd.DataFrame) -> None:
    """Print a summary of the extracted sample."""
    print("=" * 60)
    print("ANNOTATION SAMPLE SUMMARY")
    print("=" * 60)
    n_duplicates = (
        int(sample["duplicate_of"].notna().sum())
        if "duplicate_of" in sample.columns
        else 0
    )
    print(
        f"Total rows: {len(sample)} ({len(sample) - n_duplicates} unique + {n_duplicates} duplicates)"
    )
    print()
    print("By score bucket:")
    print(sample["score_bucket"].value_counts().to_string())
    print()
    print("By test_name:")
    print(sample["test_name"].value_counts().to_string())
    print()
    print("By category:")
    print(sample["category"].value_counts().to_string())
    print()
    if "question_variation" in sample.columns:
        print("By question_variation:")
        print(sample["question_variation"].value_counts().sort_index().to_string())
        print()
    print("Complex vs simple:")
    print(
        sample["is_complex"]
        .value_counts()
        .rename({True: "complex", False: "simple"})
        .to_string()
    )
    print()
    print("LLM avg score stats:")
    print(sample["llm_avg_score"].describe().to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Extract a stratified annotation sample from LLM evaluation results."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Path(s) to LLM evaluation result JSON files",
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=60,
        help="Target number of samples to extract (default: 60)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="annotation_sample.csv",
        help="Output CSV path (default: annotation_sample.csv)",
    )
    parser.add_argument(
        "--complex-ratio",
        type=float,
        default=0.6,
        help="Fraction of non-calibration samples from complex workflows (default: 0.6)",
    )
    parser.add_argument(
        "--calibration-ratio",
        type=float,
        default=0.15,
        help="Fraction of samples reserved for calibration anchors (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed (default: 10)",
    )
    parser.add_argument(
        "--n-duplicates",
        type=int,
        default=0,
        help="Number of duplicate complex questions for self-consistency checks, added on top of --n-samples (default: 0)",
    )
    
    args = parser.parse_args()

    annotation_metrics = ANNOTATION_METRICS

    # Load all results
    all_results = []
    for path in args.input_files:
        print(f"Loading {path}...")
        results = load_results_from_json(path)
        all_results.extend(results)
    print(f"Loaded {len(all_results)} total evaluation results")

    # Convert to DataFrame
    df = results_to_records(all_results, annotation_metrics)

    # Filter out error outputs (agent crashes, not real responses)
    error_prefixes = (
        "BadRequestError:",
        "GraphRecursionError:",
        "Recursion limit",
        "CSVAnalyzerException:",
    )
    error_mask = df["agent_output"].str.startswith(error_prefixes)
    n_errors = error_mask.sum()
    if n_errors > 0:
        print(f"Filtered out {n_errors} error responses")
        df = df[~error_mask].reset_index(drop=True)

    # Clamp sample size to available data
    n = min(args.n_samples, len(df))
    print(f"Sampling {n} cases from {len(df)} available...")

    # Sample
    sample = sample_stratified(
        df,
        n_total=n,
        complex_ratio=args.complex_ratio,
        calibration_ratio=args.calibration_ratio,
        seed=args.seed,
    )

    # Add consistency duplicates
    if args.n_duplicates > 0:
        sample = add_consistency_duplicates(sample, args.n_duplicates, seed=args.seed)

    print_sample_summary(sample)

    # Format and save
    output = format_for_annotation(sample, annotation_metrics)
    output.to_csv(args.output, index=False)
    print(f"\nSaved annotation sheet to: {args.output}")

    # Also save a version without LLM scores (for annotators)
    annotator_path = args.output.replace(".csv", "_annotator_sheet.csv")
    llm_cols = [
        c
        for c in output.columns
        if c.startswith("llm_")
        or c in ("score_bucket", "metric_variance", "duplicate_of")
    ]
    annotator_sheet = output.drop(columns=llm_cols)
    annotator_sheet.to_csv(annotator_path, index=False)
    print(f"Saved annotator sheet (no LLM scores) to: {annotator_path}")


if __name__ == "__main__":
    main()
