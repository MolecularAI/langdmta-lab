"""Agreement metrics for human-LLM alignment and inter-annotator reliability."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from langdmta_eval.alignment.data_loader import (
    ANNOTATION_METRICS,
    get_annotator_scores_wide,
    get_duplicate_pairs,
    get_llm_scores_wide_by_run,
    get_majority_human_scores,
    get_paired_scores,
)


@dataclass
class AlignmentResult:
    """Holds agreement metrics for a single evaluation metric."""

    metric_name: str
    n_samples: int
    exact_match_rate: float
    cohens_kappa_unweighted: float
    cohens_kappa_weighted: float
    spearman_correlation: float
    spearman_pvalue: float
    confusion_matrix: np.ndarray
    labels: List[float] = field(default_factory=list)


@dataclass
class InterAnnotatorResult:
    """Holds inter-annotator agreement for a single evaluation metric."""

    metric_name: str
    n_samples: int
    n_annotators: int
    fleiss_kappa: float
    mean_pairwise_kappa: float
    pairwise_kappas: Dict[str, float] = field(default_factory=dict)


@dataclass
class SelfConsistencyResult:
    """Holds self-consistency metrics for a single annotator and metric."""

    annotator_id: str
    metric_name: str
    n_pairs: int
    exact_match_rate: float
    mean_absolute_difference: float
    cohens_kappa_weighted: float


# --- Standard labels for each metric type ---
STANDARD_LABELS = {
    "correctness": [0.0, 0.5, 1.0],
    "output_relevancy": [0.0, 0.5, 1.0],
    "output_clarity": [0.0, 0.5, 1.0],
    "overall_solvedness": [0.0, 0.5, 1.0],
    "ethical_awareness": [0.0, 0.5, 1.0],
    "capability_awareness": [0.0, 0.5, 1.0],
    "boldness": [0.0, 0.5, 1.0],
}


def compute_exact_match(human: np.ndarray, llm: np.ndarray) -> float:
    """Fraction of exact score matches."""
    if len(human) == 0:
        return np.nan
    return float(np.mean(human == llm))


def _to_str_labels(arr: np.ndarray) -> np.ndarray:
    """Convert float score array to string labels for sklearn compatibility."""
    return np.array([str(v) for v in arr])


def compute_cohens_kappa(
    human: np.ndarray,
    llm: np.ndarray,
    weights: Optional[str] = None,
    labels: Optional[List[float]] = None,
) -> float:
    """Cohen's kappa between two raters.

    Args:
        human: Array of human scores
        llm: Array of LLM scores
        weights: None for unweighted, 'linear' or 'quadratic' for weighted
        labels: All possible label values

    Returns:
        Kappa score, or np.nan if undefined (e.g., constant arrays)
    """
    if len(human) < 2:
        return np.nan

    # Cohen's kappa is undefined if either rater uses only one category
    if len(np.unique(human)) < 2 and len(np.unique(llm)) < 2:
        if np.array_equal(human, llm):
            return 1.0  # Perfect agreement on a single category
        return np.nan

    # sklearn requires discrete labels, not continuous floats
    h_str = _to_str_labels(human)
    l_str = _to_str_labels(llm)
    lab_str = [str(v) for v in labels] if labels is not None else None

    try:
        return float(cohen_kappa_score(h_str, l_str, weights=weights, labels=lab_str))
    except (ValueError, ZeroDivisionError):
        return np.nan


def compute_spearman(
    human: np.ndarray,
    llm: np.ndarray,
) -> Tuple[float, float]:
    """Spearman rank correlation.

    Returns:
        (correlation, p-value), or (np.nan, np.nan) if undefined
    """
    if len(human) < 3:
        return np.nan, np.nan

    # Spearman is undefined for constant arrays
    if np.std(human) == 0 or np.std(llm) == 0:
        return np.nan, np.nan

    try:
        result = spearmanr(human, llm)
        return float(result.correlation), float(result.pvalue)
    except (ValueError, RuntimeWarning):
        return np.nan, np.nan


def compute_confusion_matrix(
    human: np.ndarray,
    llm: np.ndarray,
    labels: Optional[List[float]] = None,
) -> Tuple[np.ndarray, List[float]]:
    """Confusion matrix (human on rows, LLM on columns).

    Returns:
        (matrix, labels) tuple where matrix is integer counts
    """
    if labels is None:
        labels = sorted(set(np.concatenate([human, llm])))

    # sklearn needs string labels for float arrays
    h_str = _to_str_labels(human)
    l_str = _to_str_labels(llm)
    lab_str = [str(v) for v in labels]

    cm = confusion_matrix(h_str, l_str, labels=lab_str)
    return cm.astype(int), list(labels)


def compute_alignment(
    human: np.ndarray,
    llm: np.ndarray,
    metric_name: str,
    labels: Optional[List[float]] = None,
) -> AlignmentResult:
    """Compute all agreement metrics for a single evaluation metric."""
    if labels is None:
        labels = STANDARD_LABELS.get(metric_name, [0.0, 0.5, 1.0])

    n = len(human)
    exact_match = compute_exact_match(human, llm)
    kappa_uw = compute_cohens_kappa(human, llm, weights=None, labels=labels)
    kappa_w = compute_cohens_kappa(human, llm, weights="linear", labels=labels)
    spearman_rho, spearman_p = compute_spearman(human, llm)

    if n > 0:
        cm, cm_labels = compute_confusion_matrix(human, llm, labels=labels)
    else:
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        cm_labels = labels

    return AlignmentResult(
        metric_name=metric_name,
        n_samples=n,
        exact_match_rate=exact_match,
        cohens_kappa_unweighted=kappa_uw,
        cohens_kappa_weighted=kappa_w,
        spearman_correlation=spearman_rho,
        spearman_pvalue=spearman_p,
        confusion_matrix=cm,
        labels=cm_labels,
    )


def compute_all_alignments(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    exclude_not_applicable: bool = True,
    annotator_id: Optional[str] = None,
    use_majority_vote: bool = False,
    include_overall: bool = True,
) -> Dict[str, AlignmentResult]:
    """Compute alignment for all metrics in the DataFrame.

    Args:
        df: Long-format DataFrame from data_loader
        metrics: Which metrics to compute (defaults to all ANNOTATION_METRICS)
        exclude_not_applicable: Whether to exclude -1.0 scores
        annotator_id: If set, compute alignment for a specific annotator only
        use_majority_vote: If True, use majority vote across annotators as human score
        include_overall: If True, add an ``overall`` entry pooling all metrics

    Returns:
        Dict mapping metric_name -> AlignmentResult
    """
    if metrics is None:
        metrics = ANNOTATION_METRICS

    results = {}
    all_human: List[float] = []
    all_llm: List[float] = []

    for metric in metrics:
        if use_majority_vote:
            _, human, llm = get_majority_human_scores(
                df, metric, exclude_not_applicable=exclude_not_applicable
            )
        else:
            human, llm = get_paired_scores(
                df, metric,
                exclude_not_applicable=exclude_not_applicable,
                annotator_id=annotator_id,
            )

        results[metric] = compute_alignment(human, llm, metric)
        all_human.extend(human.tolist())
        all_llm.extend(llm.tolist())

    if include_overall:
        results["overall"] = compute_alignment(
            np.array(all_human), np.array(all_llm), "overall",
        )

    return results


def compute_all_alignments_by_model(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    use_majority_vote: bool = True,
    include_overall: bool = True,
) -> Dict[str, Dict[str, AlignmentResult]]:
    """Compute alignment for each model in the DataFrame.

    Args:
        df: Long-format DataFrame with a ``model_id`` column
        metrics: Which metrics to compute (defaults to all ANNOTATION_METRICS)
        use_majority_vote: If True, use majority vote across annotators
        include_overall: If True, add an ``overall`` entry per model

    Returns:
        Dict mapping model_id -> {metric_name -> AlignmentResult}
    """
    if "model_id" not in df.columns:
        raise ValueError("DataFrame must have a 'model_id' column")

    results = {}
    for model_id, model_df in df.groupby("model_id"):
        results[model_id] = compute_all_alignments(
            model_df,
            metrics=metrics,
            use_majority_vote=use_majority_vote,
            include_overall=include_overall,
        )
    return results


def multi_model_summary_table(
    all_results: Dict[str, Dict[str, AlignmentResult]],
) -> pd.DataFrame:
    """Format multi-model alignment results as a single summary DataFrame.

    Args:
        all_results: Output of :func:`compute_all_alignments_by_model`

    Returns:
        DataFrame with a ``model`` column prepended to the standard
        alignment summary columns.
    """
    frames = []
    for model_id, results in all_results.items():
        table = alignment_summary_table(results)
        table.insert(0, "model", model_id)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def alignment_summary_table(
    results: Dict[str, AlignmentResult],
) -> pd.DataFrame:
    """Format alignment results as a summary DataFrame (Table 5).

    Returns:
        DataFrame with columns: metric, n, exact_match, kappa_unweighted,
        kappa_weighted, spearman_rho, spearman_p
    """
    rows = []
    for metric, r in results.items():
        rows.append({
            "metric": metric,
            "n": r.n_samples,
            "exact_match": r.exact_match_rate,
            "kappa_unweighted": r.cohens_kappa_unweighted,
            "kappa_weighted": r.cohens_kappa_weighted,
            "spearman_rho": r.spearman_correlation,
            "spearman_p": r.spearman_pvalue,
        })
    return pd.DataFrame(rows)


# --- Inter-annotator agreement ---


def _build_fleiss_matrix(wide: pd.DataFrame, categories: List[float]) -> np.ndarray:
    """Build the Fleiss kappa rating matrix.

    Args:
        wide: DataFrame with annotation_id as index, annotators as columns
        categories: The possible score values

    Returns:
        Matrix of shape (n_subjects, n_categories) with counts per category
    """
    n_subjects = len(wide)
    n_categories = len(categories)
    matrix = np.zeros((n_subjects, n_categories), dtype=int)

    for i, (_, row) in enumerate(wide.iterrows()):
        for val in row.values:
            if not np.isnan(val):
                idx = categories.index(val) if val in categories else -1
                if idx >= 0:
                    matrix[i, idx] += 1

    return matrix


def _compute_fleiss_kappa(matrix: np.ndarray) -> float:
    """Compute Fleiss' kappa from a rating matrix.

    Args:
        matrix: Shape (n_subjects, n_categories), counts per category

    Returns:
        Fleiss' kappa, or np.nan if undefined
    """
    n_subjects, n_categories = matrix.shape
    n_raters = matrix.sum(axis=1)

    if n_subjects < 2 or n_categories < 2:
        return np.nan

    # Check all subjects have the same number of raters
    n = n_raters[0]
    if not np.all(n_raters == n) or n < 2:
        return np.nan

    # P_i: agreement for each subject
    p_i = (np.sum(matrix ** 2, axis=1) - n) / (n * (n - 1))
    p_bar = np.mean(p_i)

    # P_j: proportion assigned to each category
    p_j = np.sum(matrix, axis=0) / (n_subjects * n)
    p_e = np.sum(p_j ** 2)

    if p_e == 1.0:
        return np.nan  # All raters always agree on the same category

    kappa = (p_bar - p_e) / (1.0 - p_e)
    return float(kappa)


def compute_inter_annotator_agreement(
    df: pd.DataFrame,
    metric: str,
    exclude_not_applicable: bool = True,
) -> InterAnnotatorResult:
    """Compute inter-annotator agreement for a single metric.

    Uses Fleiss' kappa (for 3+ annotators) and mean pairwise Cohen's kappa.

    Args:
        df: Long-format DataFrame from data_loader
        metric: Metric name
        exclude_not_applicable: Whether to exclude -1.0 scores

    Returns:
        InterAnnotatorResult with Fleiss' kappa and pairwise kappas
    """
    wide = get_annotator_scores_wide(df, metric, exclude_not_applicable)

    if len(wide) == 0 or len(wide.columns) < 2:
        return InterAnnotatorResult(
            metric_name=metric,
            n_samples=0,
            n_annotators=len(wide.columns) if len(wide) > 0 else 0,
            fleiss_kappa=np.nan,
            mean_pairwise_kappa=np.nan,
        )

    annotators = list(wide.columns)
    n_annotators = len(annotators)
    categories = STANDARD_LABELS.get(metric, [0.0, 0.5, 1.0])

    # Fleiss' kappa
    fleiss_matrix = _build_fleiss_matrix(wide, categories)
    fleiss_k = _compute_fleiss_kappa(fleiss_matrix)

    # Pairwise Cohen's kappas
    pairwise = {}
    kappa_values = []
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            a1, a2 = annotators[i], annotators[j]
            scores1 = wide[a1].to_numpy()
            scores2 = wide[a2].to_numpy()
            k = compute_cohens_kappa(scores1, scores2, weights="linear", labels=categories)
            pair_key = f"{a1}_vs_{a2}"
            pairwise[pair_key] = k
            if not np.isnan(k):
                kappa_values.append(k)

    mean_pw = float(np.mean(kappa_values)) if kappa_values else np.nan

    return InterAnnotatorResult(
        metric_name=metric,
        n_samples=len(wide),
        n_annotators=n_annotators,
        fleiss_kappa=fleiss_k,
        mean_pairwise_kappa=mean_pw,
        pairwise_kappas=pairwise,
    )


def compute_all_inter_annotator(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    exclude_not_applicable: bool = True,
    include_overall: bool = True,
) -> Dict[str, InterAnnotatorResult]:
    """Compute inter-annotator agreement for all metrics.

    Args:
        df: Long-format DataFrame from data_loader
        metrics: Which metrics to compute (defaults to ANNOTATION_METRICS)
        exclude_not_applicable: Whether to exclude -1.0 scores
        include_overall: If True, add an ``overall`` entry pooling all metrics

    Returns:
        Dict mapping metric_name -> InterAnnotatorResult
    """
    if metrics is None:
        metrics = ANNOTATION_METRICS

    results = {
        metric: compute_inter_annotator_agreement(df, metric, exclude_not_applicable)
        for metric in metrics
    }

    if include_overall:
        # Pool scores across metrics by treating each (annotation_id, metric)
        # as an independent subject
        pooled = df.copy()
        if exclude_not_applicable:
            pooled = pooled[pooled["human_score"] != -1.0]
        pooled = pooled.dropna(subset=["human_score"])

        if len(pooled) > 0 and pooled["annotator_id"].nunique() >= 2:
            # Create a composite subject ID so each (annotation_id, metric) pair
            # is treated as its own item
            pooled["_subject"] = (
                pooled["annotation_id"].astype(str) + "_" + pooled["metric"]
            )
            wide = pooled.pivot_table(
                index="_subject",
                columns="annotator_id",
                values="human_score",
                aggfunc="first",
            ).dropna()

            annotators = list(wide.columns)
            categories = [0.0, 0.5, 1.0]

            fleiss_matrix = _build_fleiss_matrix(wide, categories)
            fleiss_k = _compute_fleiss_kappa(fleiss_matrix)

            pairwise = {}
            kappa_values = []
            for i in range(len(annotators)):
                for j in range(i + 1, len(annotators)):
                    a1, a2 = annotators[i], annotators[j]
                    s1 = wide[a1].to_numpy()
                    s2 = wide[a2].to_numpy()
                    k = compute_cohens_kappa(s1, s2, weights="linear", labels=categories)
                    pairwise[f"{a1}_vs_{a2}"] = k
                    if not np.isnan(k):
                        kappa_values.append(k)

            results["overall"] = InterAnnotatorResult(
                metric_name="overall",
                n_samples=len(wide),
                n_annotators=len(annotators),
                fleiss_kappa=fleiss_k,
                mean_pairwise_kappa=float(np.mean(kappa_values)) if kappa_values else np.nan,
                pairwise_kappas=pairwise,
            )
        else:
            n_ann = pooled["annotator_id"].nunique() if len(pooled) > 0 else 0
            results["overall"] = InterAnnotatorResult(
                metric_name="overall",
                n_samples=0,
                n_annotators=n_ann,
                fleiss_kappa=np.nan,
                mean_pairwise_kappa=np.nan,
            )

    return results


def inter_annotator_summary_table(
    results: Dict[str, InterAnnotatorResult],
) -> pd.DataFrame:
    """Format inter-annotator results as a summary DataFrame.

    Returns:
        DataFrame with columns: metric, n, n_annotators, fleiss_kappa,
        mean_pairwise_kappa, plus individual pairwise kappas
    """
    rows = []
    for metric, r in results.items():
        row = {
            "metric": metric,
            "n": r.n_samples,
            "n_annotators": r.n_annotators,
            "fleiss_kappa": r.fleiss_kappa,
            "mean_pairwise_kappa": r.mean_pairwise_kappa,
        }
        row.update(r.pairwise_kappas)
        rows.append(row)
    return pd.DataFrame(rows)


# --- Self-consistency ---


def compute_self_consistency(
    df: pd.DataFrame,
    metric: str,
    annotator_id: str,
    exclude_not_applicable: bool = True,
) -> SelfConsistencyResult:
    """Compute self-consistency for one annotator on one metric.

    Compares the annotator's scores on duplicate questions against their
    scores on the corresponding originals.

    Args:
        df: Long-format DataFrame from data_loader (must include
            ``duplicate_of`` and ``question`` columns)
        metric: Metric name
        annotator_id: Which annotator to evaluate
        exclude_not_applicable: Whether to exclude -1.0 scores

    Returns:
        SelfConsistencyResult with agreement statistics
    """
    originals, duplicates = get_duplicate_pairs(
        df, metric, annotator_id=annotator_id,
        exclude_not_applicable=exclude_not_applicable,
    )

    n = len(originals)
    if n == 0:
        return SelfConsistencyResult(
            annotator_id=annotator_id,
            metric_name=metric,
            n_pairs=0,
            exact_match_rate=np.nan,
            mean_absolute_difference=np.nan,
            cohens_kappa_weighted=np.nan,
        )

    exact_match = compute_exact_match(originals, duplicates)
    mad = float(np.mean(np.abs(originals - duplicates)))

    labels = STANDARD_LABELS.get(metric, [0.0, 0.5, 1.0])
    kappa_w = compute_cohens_kappa(originals, duplicates, weights="linear", labels=labels)

    return SelfConsistencyResult(
        annotator_id=annotator_id,
        metric_name=metric,
        n_pairs=n,
        exact_match_rate=exact_match,
        mean_absolute_difference=mad,
        cohens_kappa_weighted=kappa_w,
    )


def compute_all_self_consistency(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    annotator_ids: Optional[List[str]] = None,
    exclude_not_applicable: bool = True,
    include_overall: bool = True,
) -> Dict[str, Dict[str, SelfConsistencyResult]]:
    """Compute self-consistency for all metrics and annotators.

    Args:
        df: Long-format DataFrame from data_loader
        metrics: Which metrics to compute (defaults to ANNOTATION_METRICS)
        annotator_ids: Which annotators (defaults to all in df)
        exclude_not_applicable: Whether to exclude -1.0 scores
        include_overall: If True, add an ``overall`` entry pooling all metrics

    Returns:
        Nested dict: metric_name -> annotator_id -> SelfConsistencyResult
    """
    if metrics is None:
        metrics = ANNOTATION_METRICS
    if annotator_ids is None:
        annotator_ids = sorted(df["annotator_id"].unique().tolist())

    results: Dict[str, Dict[str, SelfConsistencyResult]] = {}
    for metric in metrics:
        results[metric] = {}
        for ann_id in annotator_ids:
            results[metric][ann_id] = compute_self_consistency(
                df, metric, ann_id, exclude_not_applicable,
            )

    if include_overall:
        results["overall"] = {}
        for ann_id in annotator_ids:
            all_orig: List[float] = []
            all_dupl: List[float] = []
            for metric in metrics:
                orig, dupl = get_duplicate_pairs(
                    df, metric, annotator_id=ann_id,
                    exclude_not_applicable=exclude_not_applicable,
                )
                all_orig.extend(orig.tolist())
                all_dupl.extend(dupl.tolist())

            orig_arr = np.array(all_orig)
            dupl_arr = np.array(all_dupl)
            n = len(orig_arr)
            if n > 0:
                exact = compute_exact_match(orig_arr, dupl_arr)
                mad = float(np.mean(np.abs(orig_arr - dupl_arr)))
                kappa = compute_cohens_kappa(
                    orig_arr, dupl_arr, weights="linear", labels=[0.0, 0.5, 1.0],
                )
            else:
                exact = mad = kappa = np.nan

            results["overall"][ann_id] = SelfConsistencyResult(
                annotator_id=ann_id,
                metric_name="overall",
                n_pairs=n,
                exact_match_rate=exact,
                mean_absolute_difference=mad,
                cohens_kappa_weighted=kappa,
            )

    return results


def compute_llm_inter_judge_agreement(
    llm_runs_df: "pd.DataFrame",
    metric: str,
    model_id: str,
) -> "InterAnnotatorResult":
    """Compute inter-judge agreement across LLM runs for one metric and model.

    Mirrors :func:`compute_inter_annotator_agreement` but operates on LLM
    run scores instead of human annotator scores.  The three runs act as
    independent "raters" and the same Fleiss' kappa + pairwise Cohen's kappa
    measures are applied.

    Args:
        llm_runs_df: Output of :func:`~langdmta_eval.alignment.data_loader.load_llm_runs`.
        metric: Metric name (e.g., 'relevancy').
        model_id: Model to evaluate (e.g., 'GPT-5').

    Returns:
        InterAnnotatorResult with Fleiss' kappa and pairwise kappas between runs.
    """
    wide = get_llm_scores_wide_by_run(llm_runs_df, metric, model_id)

    if len(wide) == 0 or len(wide.columns) < 2:
        return InterAnnotatorResult(
            metric_name=metric,
            n_samples=0,
            n_annotators=len(wide.columns) if len(wide) > 0 else 0,
            fleiss_kappa=np.nan,
            mean_pairwise_kappa=np.nan,
        )

    runs = list(wide.columns)
    n_runs = len(runs)
    categories = STANDARD_LABELS.get(metric, [0.0, 0.5, 1.0])

    fleiss_matrix = _build_fleiss_matrix(wide, categories)
    fleiss_k = _compute_fleiss_kappa(fleiss_matrix)

    pairwise = {}
    kappa_values = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            r1, r2 = runs[i], runs[j]
            s1 = wide[r1].to_numpy()
            s2 = wide[r2].to_numpy()
            k = compute_cohens_kappa(s1, s2, weights="linear", labels=categories)
            pair_key = f"{r1}_vs_{r2}"
            pairwise[pair_key] = k
            if not np.isnan(k):
                kappa_values.append(k)

    mean_pw = float(np.mean(kappa_values)) if kappa_values else np.nan

    return InterAnnotatorResult(
        metric_name=metric,
        n_samples=len(wide),
        n_annotators=n_runs,
        fleiss_kappa=fleiss_k,
        mean_pairwise_kappa=mean_pw,
        pairwise_kappas=pairwise,
    )


def compute_all_llm_inter_judge_agreement(
    llm_runs_df: "pd.DataFrame",
    models: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    include_overall: bool = True,
) -> Dict[str, Dict[str, "InterAnnotatorResult"]]:
    """Compute inter-judge agreement for all models and metrics.

    Args:
        llm_runs_df: Output of :func:`~langdmta_eval.alignment.data_loader.load_llm_runs`.
        models: Which models to evaluate (defaults to all in llm_runs_df).
        metrics: Which metrics to evaluate (defaults to ANNOTATION_METRICS).
        include_overall: If True, add an ``overall`` entry pooling all metrics.

    Returns:
        Nested dict: model_id -> metric_name -> InterAnnotatorResult.
    """
    if metrics is None:
        metrics = ANNOTATION_METRICS
    if models is None:
        models = sorted(llm_runs_df["model_id"].unique().tolist())

    results: Dict[str, Dict[str, InterAnnotatorResult]] = {}
    for model_id in models:
        model_results = {
            metric: compute_llm_inter_judge_agreement(llm_runs_df, metric, model_id)
            for metric in metrics
        }

        if include_overall:
            model_subset = llm_runs_df[llm_runs_df["model_id"] == model_id].copy()
            model_subset["_subject"] = (
                model_subset["annotation_id"].astype(str) + "_" + model_subset["metric"]
            )
            wide_all = model_subset.pivot_table(
                index="_subject",
                columns="run_id",
                values="llm_score",
                aggfunc="first",
            ).dropna()

            if len(wide_all) > 0 and len(wide_all.columns) >= 2:
                runs = list(wide_all.columns)
                categories = [0.0, 0.5, 1.0]
                fleiss_matrix = _build_fleiss_matrix(wide_all, categories)
                fleiss_k = _compute_fleiss_kappa(fleiss_matrix)

                pairwise = {}
                kappa_values = []
                for i in range(len(runs)):
                    for j in range(i + 1, len(runs)):
                        r1, r2 = runs[i], runs[j]
                        s1 = wide_all[r1].to_numpy()
                        s2 = wide_all[r2].to_numpy()
                        k = compute_cohens_kappa(s1, s2, weights="linear", labels=categories)
                        pairwise[f"{r1}_vs_{r2}"] = k
                        if not np.isnan(k):
                            kappa_values.append(k)

                model_results["overall"] = InterAnnotatorResult(
                    metric_name="overall",
                    n_samples=len(wide_all),
                    n_annotators=len(runs),
                    fleiss_kappa=fleiss_k,
                    mean_pairwise_kappa=float(np.mean(kappa_values)) if kappa_values else np.nan,
                    pairwise_kappas=pairwise,
                )
            else:
                model_results["overall"] = InterAnnotatorResult(
                    metric_name="overall",
                    n_samples=0,
                    n_annotators=0,
                    fleiss_kappa=np.nan,
                    mean_pairwise_kappa=np.nan,
                )

        results[model_id] = model_results

    return results


def llm_inter_judge_summary_table(
    results_by_model: Dict[str, Dict[str, "InterAnnotatorResult"]],
) -> "pd.DataFrame":
    """Format LLM inter-judge agreement results as a summary DataFrame.

    Stacks one table per model (using :func:`inter_annotator_summary_table`)
    and prepends a ``model`` column.

    Args:
        results_by_model: Output of :func:`compute_all_llm_inter_judge_agreement`.

    Returns:
        DataFrame with columns: model, metric, n, n_annotators (runs),
        fleiss_kappa, mean_pairwise_kappa, plus individual pairwise kappas.
    """
    frames = []
    for model_id, results in results_by_model.items():
        table = inter_annotator_summary_table(results)
        table.insert(0, "model", model_id)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def self_consistency_summary_table(
    results: Dict[str, Dict[str, SelfConsistencyResult]],
) -> pd.DataFrame:
    """Format self-consistency results as a summary DataFrame.

    Args:
        results: Output of :func:`compute_all_self_consistency`

    Returns:
        DataFrame with columns: metric, annotator, n_pairs, exact_match,
        mean_abs_diff, kappa_weighted
    """
    rows = []
    for metric, annotators in results.items():
        for ann_id, r in annotators.items():
            rows.append({
                "metric": metric,
                "annotator": ann_id,
                "n_pairs": r.n_pairs,
                "exact_match": r.exact_match_rate,
                "mean_abs_diff": r.mean_absolute_difference,
                "kappa_weighted": r.cohens_kappa_weighted,
            })
    return pd.DataFrame(rows)
