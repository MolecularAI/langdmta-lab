"""Load and merge human annotation data with LLM judge scores.

Supports multiple annotators per case and multiple annotation datasets.
Produces paired (human, LLM) score arrays for alignment analysis.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from langdmta_eval.judges.score_mappings import get_metric_mapping, normalize_score
from langdmta_eval.schemas import load_results_from_json

# Metrics that annotators score (same as in extract_annotation_sample.py)
ANNOTATION_METRICS = [
    "relevancy",
    "completeness",
    "structural_clarity",
    # "ethical_awareness",
    "scope_adherence",
]


def _load_llm_scores_from_json(
    json_path: str,
    metrics: List[str],
) -> pd.DataFrame:
    """Load LLM judge scores from a JSON results file.

    Returns a DataFrame with ``trace_id`` and ``llm_<metric>`` columns,
    suitable for merging with annotator CSVs on ``trace_id``.
    """
    results = load_results_from_json(json_path)
    records = []
    for r in results:
        tc = r.test_case
        row = {"trace_id": tc.trace_id}
        for m in metrics:
            row[f"llm_{m}"] = r.scores.get(m)
        if r.reasoning:
            row["llm_reasoning"] = r.reasoning
        records.append(row)
    return pd.DataFrame(records)


def load_annotation_pair(
    annotator_csv: str,
    llm_scores_csv: str,
    annotator_id: Optional[str] = None,
) -> pd.DataFrame:
    """Load a single annotator sheet and merge with LLM scores.

    Args:
        annotator_csv: Path to the annotated CSV (contains human_* columns)
        llm_scores_csv: Path to the companion CSV or JSON with LLM scores.
            CSV files are merged on ``annotation_id``; JSON files (from
            ``save_results_to_json``) are merged on ``trace_id``.
        annotator_id: Optional label for this annotator (e.g., 'LS', 'RMO')

    Returns:
        Long-format DataFrame with columns: annotation_id, metric, human_score,
        human_label, llm_score, annotator_id, category, test_name, session
    """
    ann = pd.read_csv(annotator_csv)
    print(f"Loaded {len(ann)} annotations from: {Path(annotator_csv.split('/')[-1])}")

    is_json = str(llm_scores_csv).endswith(".json")

    # Filter to only fully annotated rows
    human_cols = [f"human_{m}" for m in ANNOTATION_METRICS]
    for col in human_cols:
        ann[col] = ann[col].fillna("").astype(str).str.strip()
    complete_mask = ann[human_cols].replace("", pd.NA).notna().all(axis=1)
    ann = ann[complete_mask].copy()

    if len(ann) == 0:
        return pd.DataFrame()

    # Drop any llm_* columns already present in the annotator sheet so the
    # merge doesn't create suffixed duplicates (llm_relevancy_x / _y).
    llm_metric_cols = [f"llm_{m}" for m in ANNOTATION_METRICS]
    ann = ann.drop(
        columns=[c for c in llm_metric_cols if c in ann.columns], errors="ignore"
    )

    if is_json:
        llm = _load_llm_scores_from_json(llm_scores_csv, ANNOTATION_METRICS)
        merge_key = "trace_id"
        llm_cols = [merge_key] + [f"llm_{m}" for m in ANNOTATION_METRICS]
        if "llm_reasoning" in llm.columns:
            llm_cols.append("llm_reasoning")
        available_llm_cols = [c for c in llm_cols if c in llm.columns]
        n_json = len(llm)
        merged = ann.merge(llm[available_llm_cols], on=merge_key, how="inner")
        n_unmatched = len(ann) - len(merged)
        if n_unmatched > 0:
            warnings.warn(
                f"{n_unmatched} annotated case(s) had no matching trace_id "
                f"in {llm_scores_csv}"
            )
        print(
            f"Matched {len(merged)} of {len(ann)} annotated cases "
            f"from {n_json} JSON entries"
        )
    else:
        llm = pd.read_csv(llm_scores_csv)
        merge_key = "annotation_id"
        llm_cols = [merge_key] + [f"llm_{m}" for m in ANNOTATION_METRICS]
        available_llm_cols = [c for c in llm_cols if c in llm.columns]
        merged = ann.merge(llm[available_llm_cols], on=merge_key, how="inner")
        print(f"Merged {len(merged)} annotations with LLM scores")

    # Melt to long format: one row per (case, metric)
    records = []
    for _, row in merged.iterrows():
        for metric in ANNOTATION_METRICS:
            human_col = f"human_{metric}"
            llm_col = f"llm_{metric}"

            human_label = row.get(human_col, "")
            if not human_label or human_label == "":
                continue

            human_score = normalize_score(human_label)
            llm_score = row.get(llm_col, None)
            if llm_score is not None:
                llm_score = float(llm_score)

            records.append(
                {
                    "annotation_id": row["annotation_id"],
                    "metric": metric,
                    "human_label": human_label,
                    "human_score": human_score,
                    "llm_score": llm_score,
                    "annotator_id": annotator_id or "unknown",
                    "category": row.get("category", ""),
                    "test_name": row.get("test_name", ""),
                    "session": row.get("session", ""),
                    "question": row.get("question", ""),
                    "agent_output": row.get("agent_output", ""),
                    "evaluation_context": row.get("evaluation_context", ""),
                    "human_justification": row.get("human_justification", ""),
                    "llm_reasoning": row.get("llm_reasoning", ""),
                    "duplicate_of": row.get("duplicate_of", ""),
                }
            )

    return pd.DataFrame(records)


def load_multiple_annotators(
    annotator_csvs: Dict[str, str],
    llm_scores_csv: str,
) -> pd.DataFrame:
    """Load annotations from multiple annotators for the same cases.

    Args:
        annotator_csvs: Mapping of annotator_id -> annotator CSV path
        llm_scores_csv: Path to the companion CSV with LLM scores

    Returns:
        Long-format DataFrame with all annotators' scores
    """
    frames = []
    for annotator_id, csv_path in annotator_csvs.items():
        df = load_annotation_pair(csv_path, llm_scores_csv, annotator_id=annotator_id)
        if len(df) > 0:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_multiple_models(
    annotator_csvs: Dict[str, str],
    model_llm_scores: Dict[str, "str | List[str]"],
) -> pd.DataFrame:
    """Load annotations paired with LLM scores from multiple judge models.

    For each model, calls :func:`load_multiple_annotators` with that model's
    LLM scores CSV, adds a ``model_id`` column, and concatenates.

    Args:
        annotator_csvs: Mapping of annotator_id -> annotator CSV path
        model_llm_scores: Mapping of model_id -> LLM scores path (str) or
            list of paths (one per run).  When a list is given, the first
            path is used to populate ``llm_score`` in the returned DataFrame;
            call :func:`get_majority_llm_vote_df` afterwards to replace it
            with the majority vote across all runs.

    Returns:
        Long-format DataFrame with all annotators and models, including
        a ``model_id`` column.
    """
    frames = []
    for model_id, llm_csv in model_llm_scores.items():
        # Accept a list of run paths — use the first run for the initial load
        if isinstance(llm_csv, list):
            llm_csv = llm_csv[0]
        model_df = load_multiple_annotators(annotator_csvs, llm_csv)
        if len(model_df) > 0:
            model_df["model_id"] = model_id
            frames.append(model_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    ids_per_model = combined.groupby("model_id")["annotation_id"].apply(set)
    if len(ids_per_model) > 1:
        common = set.intersection(*ids_per_model)
        for model_id, ids in ids_per_model.items():
            diff = ids - common
            if diff:
                warnings.warn(
                    f"Model '{model_id}' has {len(diff)} annotation_ids "
                    f"not shared by all models"
                )

    return combined


def load_all_annotations(
    annotation_pairs: List[Tuple[str, str, Optional[str]]],
) -> pd.DataFrame:
    """Load and concatenate multiple annotation pairs.

    Args:
        annotation_pairs: List of (annotator_csv, llm_scores_csv, annotator_id)
            tuples. annotator_id is optional (defaults to 'unknown').

    Returns:
        Long-format DataFrame with a 'source_dataset' / 'annotator_id' column
    """
    frames = []
    for pair in annotation_pairs:
        annotator_csv, llm_csv = pair[0], pair[1]
        annotator_id = pair[2] if len(pair) > 2 else None
        df = load_annotation_pair(annotator_csv, llm_csv, annotator_id=annotator_id)
        if len(df) > 0:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def get_paired_scores(
    df: pd.DataFrame,
    metric: str,
    exclude_not_applicable: bool = True,
    annotator_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract paired (human, LLM) score arrays for a single metric.

    Args:
        df: Long-format DataFrame from load_* functions
        metric: Metric name (e.g., 'correctness')
        exclude_not_applicable: If True, remove rows where either score is -1.0
        annotator_id: If set, filter to a specific annotator

    Returns:
        (human_scores, llm_scores) as numpy arrays of the same length
    """
    subset = df[df["metric"] == metric].copy()

    if annotator_id is not None:
        subset = subset[subset["annotator_id"] == annotator_id]

    # Drop rows with missing scores
    subset = subset.dropna(subset=["human_score", "llm_score"])

    if exclude_not_applicable:
        subset = subset[(subset["human_score"] != -1.0) & (subset["llm_score"] != -1.0)]

    return (
        subset["human_score"].to_numpy(dtype=float),
        subset["llm_score"].to_numpy(dtype=float),
    )


def get_annotator_scores_wide(
    df: pd.DataFrame,
    metric: str,
    exclude_not_applicable: bool = True,
) -> pd.DataFrame:
    """Pivot annotator scores to wide format for inter-annotator agreement.

    Args:
        df: Long-format DataFrame from load_* functions
        metric: Metric name (e.g., 'correctness')
        exclude_not_applicable: If True, remove rows where any score is -1.0

    Returns:
        DataFrame with annotation_id as index, annotator_ids as columns,
        numeric scores as values
    """
    subset = df[df["metric"] == metric].copy()

    if exclude_not_applicable:
        subset = subset[subset["human_score"] != -1.0]

    if len(subset) == 0:
        return pd.DataFrame()

    wide = subset.pivot_table(
        index="annotation_id",
        columns="annotator_id",
        values="human_score",
        aggfunc="first",
    )

    # Drop cases where any annotator is missing
    wide = wide.dropna()

    return wide


def get_duplicate_pairs(
    df: pd.DataFrame,
    metric: str,
    annotator_id: Optional[str] = None,
    exclude_not_applicable: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract paired scores for duplicate questions (original vs duplicate).

    Finds rows where ``duplicate_of`` is non-null, matches them to the original
    row (by question text within the same annotator), and returns paired score
    arrays for measuring self-consistency.

    Args:
        df: Long-format DataFrame from load_* functions
        metric: Metric name
        annotator_id: If set, filter to a specific annotator
        exclude_not_applicable: If True, exclude -1.0 scores

    Returns:
        (original_scores, duplicate_scores) as numpy arrays of the same length
    """
    subset = df[df["metric"] == metric].copy()

    if annotator_id is not None:
        subset = subset[subset["annotator_id"] == annotator_id]

    subset = subset.dropna(subset=["human_score"])

    if exclude_not_applicable:
        subset = subset[subset["human_score"] != -1.0]

    if "duplicate_of" not in subset.columns or "question" not in subset.columns:
        return np.array([]), np.array([])

    duplicates = subset[subset["duplicate_of"].notna() & (subset["duplicate_of"] != "")]
    if len(duplicates) == 0:
        return np.array([]), np.array([])

    originals = subset[subset["duplicate_of"].isna() | (subset["duplicate_of"] == "")]

    original_scores = []
    duplicate_scores = []
    for _, dup_row in duplicates.iterrows():
        match = originals[originals["question"] == dup_row["question"]]
        if len(match) > 0:
            original_scores.append(match.iloc[0]["human_score"])
            duplicate_scores.append(dup_row["human_score"])

    return np.array(original_scores, dtype=float), np.array(
        duplicate_scores, dtype=float
    )


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows (those with a non-null ``duplicate_of`` value).

    Args:
        df: Long-format DataFrame from load_* functions

    Returns:
        DataFrame with duplicate rows removed
    """
    if "duplicate_of" not in df.columns:
        return df.copy()
    mask = df["duplicate_of"].isna() | (df["duplicate_of"] == "")
    return df[mask].copy()


def get_majority_human_scores(
    df: pd.DataFrame,
    metric: str,
    exclude_not_applicable: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute majority-vote human scores and pair with LLM scores.

    For each case, takes the most common human score across annotators.
    Useful for human-vs-LLM alignment when multiple annotators exist.

    Args:
        df: Long-format DataFrame from load_* functions
        metric: Metric name
        exclude_not_applicable: If True, exclude -1.0 scores

    Returns:
        (annotation_ids, majority_human_scores, llm_scores) as numpy arrays
    """
    subset = df[df["metric"] == metric].dropna(subset=["human_score", "llm_score"])

    if exclude_not_applicable:
        subset = subset[(subset["human_score"] != -1.0) & (subset["llm_score"] != -1.0)]

    if len(subset) == 0:
        return np.array([]), np.array([]), np.array([])

    # Majority vote per annotation_id
    majority = (
        subset.groupby("annotation_id")
        .agg(
            human_score=(
                "human_score",
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            ),
            llm_score=("llm_score", "first"),
        )
        .reset_index()
    )

    return (
        majority["annotation_id"].to_numpy(),
        majority["human_score"].to_numpy(dtype=float),
        majority["llm_score"].to_numpy(dtype=float),
    )


def load_llm_runs(
    annotator_csvs: Dict[str, str],
    model_llm_scores_multi: Dict[str, List[str]],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load all LLM judge runs for inter-judge agreement analysis.

    For each model and each run file, loads LLM scores and annotates with
    ``model_id`` and ``run_id`` (run1, run2, run3, ...).  The resulting
    DataFrame contains one row per (annotation_id, metric, model_id, run_id).

    Args:
        annotator_csvs: Mapping of annotator_id -> annotator CSV path.
            Used only to resolve the set of valid annotation_ids and
            to carry the ``duplicate_of`` column for :func:`drop_duplicates`.
        model_llm_scores_multi: Mapping of model_id -> list of JSON paths
            (one per run).
        metrics: Which metrics to include (defaults to ANNOTATION_METRICS).

    Returns:
        Long-format DataFrame with columns:
        annotation_id, metric, model_id, run_id, llm_score, duplicate_of
    """
    if metrics is None:
        metrics = ANNOTATION_METRICS

    # Build a reference DataFrame of valid annotation_ids from one annotator CSV
    ref_csv = next(iter(annotator_csvs.values()))
    ref = pd.read_csv(ref_csv)
    human_cols = [f"human_{m}" for m in metrics]
    for col in human_cols:
        ref[col] = ref[col].fillna("").astype(str).str.strip()
    complete_mask = ref[human_cols].replace("", pd.NA).notna().all(axis=1)
    ref = ref[complete_mask][["annotation_id", "trace_id", "duplicate_of"]].copy()

    frames = []
    for model_id, run_paths in model_llm_scores_multi.items():
        for run_idx, json_path in enumerate(run_paths):
            run_id = f"run{run_idx + 1}"
            llm_df = _load_llm_scores_from_json(json_path, metrics)
            merged = ref.merge(llm_df, on="trace_id", how="inner")
            for _, row in merged.iterrows():
                for m in metrics:
                    llm_score = row.get(f"llm_{m}", None)
                    if llm_score is None:
                        continue
                    frames.append(
                        {
                            "annotation_id": row["annotation_id"],
                            "metric": m,
                            "model_id": model_id,
                            "run_id": run_id,
                            "llm_score": float(llm_score),
                            "duplicate_of": row.get("duplicate_of", ""),
                        }
                    )

    if not frames:
        return pd.DataFrame(
            columns=[
                "annotation_id",
                "metric",
                "model_id",
                "run_id",
                "llm_score",
                "duplicate_of",
            ]
        )

    return pd.DataFrame(frames)


def get_llm_scores_wide_by_run(
    llm_runs_df: pd.DataFrame,
    metric: str,
    model_id: str,
) -> pd.DataFrame:
    """Pivot LLM run scores to wide format for inter-judge agreement.

    Analogous to :func:`get_annotator_scores_wide` but operates on the
    LLM-runs DataFrame produced by :func:`load_llm_runs`.

    Args:
        llm_runs_df: Output of :func:`load_llm_runs`.
        metric: Metric name (e.g., 'relevancy').
        model_id: Model to select (e.g., 'GPT-5').

    Returns:
        DataFrame with annotation_id as index and run_ids as columns,
        dropping rows where any run is missing.
    """
    subset = llm_runs_df[
        (llm_runs_df["metric"] == metric) & (llm_runs_df["model_id"] == model_id)
    ].copy()

    if len(subset) == 0:
        return pd.DataFrame()

    wide = subset.pivot_table(
        index="annotation_id",
        columns="run_id",
        values="llm_score",
        aggfunc="first",
    )
    wide = wide.dropna()
    return wide


def get_majority_llm_vote_df(
    df: pd.DataFrame,
    llm_runs_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Replace llm_score in df with the majority vote across LLM runs.

    For each (annotation_id, metric, model_id) triple, computes the mode
    across runs and overwrites ``llm_score`` in the main DataFrame.

    Args:
        df: Main long-format DataFrame (from :func:`load_multiple_models`).
        llm_runs_df: Output of :func:`load_llm_runs`.
        metrics: Which metrics to update (defaults to ANNOTATION_METRICS).

    Returns:
        Copy of ``df`` with ``llm_score`` replaced by majority-voted LLM scores.
    """
    if metrics is None:
        metrics = ANNOTATION_METRICS

    majority = (
        llm_runs_df[llm_runs_df["metric"].isin(metrics)]
        .groupby(["annotation_id", "metric", "model_id"])["llm_score"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        .reset_index()
        .rename(columns={"llm_score": "llm_score_majority"})
    )

    result = df.copy()
    result = result.merge(
        majority, on=["annotation_id", "metric", "model_id"], how="left"
    )
    mask = result["llm_score_majority"].notna()
    result.loc[mask, "llm_score"] = result.loc[mask, "llm_score_majority"]
    result = result.drop(columns=["llm_score_majority"])
    return result


def get_majority_vote_df(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    exclude_not_applicable: bool = True,
) -> pd.DataFrame:
    """Aggregate annotator scores via majority vote into a long-format DataFrame.

    For each ``(annotation_id, metric)`` pair, the most common human score
    across annotators is kept.  The returned DataFrame has the same schema as
    the input but with ``annotator_id`` set to ``"majority_vote"`` on every
    row, so it can be passed directly to :func:`build_trainset`.

    Args:
        df: Long-format DataFrame from load_* functions
        metrics: Which metrics to include (defaults to ANNOTATION_METRICS)
        exclude_not_applicable: If True, exclude -1.0 scores before voting

    Returns:
        Long-format DataFrame with majority-voted human scores
    """
    if metrics is None:
        metrics = ANNOTATION_METRICS

    subset = df[df["metric"].isin(metrics)].dropna(subset=["human_score"])

    if exclude_not_applicable:
        subset = subset[subset["human_score"] != -1.0]

    if len(subset) == 0:
        return pd.DataFrame()

    majority = (
        subset.groupby(["annotation_id", "metric"])
        .agg(
            human_score=(
                "human_score",
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            ),
            llm_score=("llm_score", "first"),
            human_label=("human_label", "first"),
        )
        .reset_index()
    )

    context_cols = [
        "annotation_id",
        "question",
        "agent_output",
        "evaluation_context",
        "tool_calls",
        "expected_tool_calls",
        "accepted_tool_calls",
        "human_justification",
        "category",
        "test_name",
        "session",
        "duplicate_of",
    ]
    available = [c for c in context_cols if c in df.columns]
    context = df.drop_duplicates("annotation_id")[available]

    result = majority.merge(context, on="annotation_id", how="left")
    result["annotator_id"] = "majority_vote"

    return result
