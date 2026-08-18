"""DSPy-compatible metric functions for judge signature optimization.

These metrics compare a judge's predicted scores against human ground truth,
enabling DSPy optimizers (LabeledFewShot, BootstrapFewShot, MIPROv2) to
maximize human-LLM alignment.
"""

from typing import Callable, Dict, List, Optional

from langdmta_eval.alignment.data_loader import ANNOTATION_METRICS
from langdmta_eval.judges.score_mappings import normalize_score


def make_alignment_metric(
    target_metrics: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Callable:
    """Create a DSPy-compatible metric function for alignment optimization.

    The returned function compares predicted judge scores against human
    ground truth labels stored in the example. It returns a score between
    0.0 and 1.0, with partial credit for adjacent-category matches.

    Args:
        target_metrics: Which evaluation metrics to compare. Defaults to
            all ANNOTATION_METRICS.
        weights: Optional per-metric weights (must sum to 1.0 if provided).
            Defaults to equal weighting.

    Returns:
        A function with signature: metric(example, prediction, trace=None) -> float
    """
    if target_metrics is None:
        target_metrics = list(ANNOTATION_METRICS)

    if weights is None:
        weights = {m: 1.0 / len(target_metrics) for m in target_metrics}

    def metric(example, prediction, trace=None) -> float:
        total_score = 0.0
        total_weight = 0.0

        for m in target_metrics:
            # Get human ground truth from example
            human_label = getattr(example, m, None)
            if human_label is None:
                continue

            # Get predicted label from prediction
            pred_label = getattr(prediction, m, None)
            if pred_label is None:
                continue

            # Normalize both to numeric
            human_val = normalize_score(human_label)
            pred_val = normalize_score(pred_label)

            if human_val is None or pred_val is None:
                continue

            # Skip NOT APPLICABLE comparisons
            if human_val == -1.0 or pred_val == -1.0:
                # Give credit if both are N/A, penalize if only one is
                if human_val == -1.0 and pred_val == -1.0:
                    score = 1.0
                else:
                    score = 0.0
            else:
                # Partial credit: 1.0 for exact match, 0.5 for adjacent, 0.0 for far
                diff = abs(human_val - pred_val)
                if diff == 0.0:
                    score = 1.0
                elif diff <= 0.5:
                    score = 0.5
                else:
                    score = 0.0

            w = weights.get(m, 0.0)
            total_score += score * w
            total_weight += w

        if total_weight == 0.0:
            return 0.0

        return total_score / total_weight

    return metric
