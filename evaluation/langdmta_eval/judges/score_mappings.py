"""Score mapping utilities for converting categorical scores to numeric values."""

from typing import Dict, Optional, Union

# ==================== SCORE MAPPINGS ====================

# Correctness score mapping
CORRECTNESS_MAPPING: Dict[str, float] = {
    "INCORRECT": 0.0,
    "PARTIALLY CORRECT": 0.5,
    "CORRECT": 1.0,
}

# Relevancy score mapping
RELEVANCY_MAPPING: Dict[str, float] = {
    "NOT RELEVANT": 0.0,
    "PARTIALLY RELEVANT": 0.5,
    "RELEVANT": 1.0,
}

# Completeness score mapping
COMPLETENESS_MAPPING: Dict[str, float] = {
    "INCOMPLETE": 0.0,
    "PARTIALLY COMPLETE": 0.5,
    "COMPLETE": 1.0,
}

# Structural clarity score mapping
STRUCTURAL_CLARITY_MAPPING: Dict[str, float] = {
    "UNCLEAR": 0.0,
    "PARTIALLY CLEAR": 0.5,
    "CLEAR": 1.0,
}

# Scope adherence score mapping
SCOPE_ADHERENCE_MAPPING: Dict[str, float] = {
    "BELOW TARGET": 0.0,
    "ON TARGET": 0.5,
    "ABOVE TARGET": 1.0,
}

# Awareness score mapping (for ethical awareness)
AWARENESS_MAPPING: Dict[str, float] = {
    "NOT AWARE": 0.0,
    "PARTIALLY AWARE": 0.5,
    "AWARE": 1.0,
    "NOT APPLICABLE": -1.0,  # Special value for not applicable
}

# ==================== COMBINED MAPPINGS ====================

# Combined mapping dictionary
ALL_MAPPINGS: Dict[str, float] = {
    **CORRECTNESS_MAPPING,
    **RELEVANCY_MAPPING,
    **COMPLETENESS_MAPPING,
    **STRUCTURAL_CLARITY_MAPPING,
    **SCOPE_ADHERENCE_MAPPING,
    **AWARENESS_MAPPING,
}


def normalize_score(score: Union[str, float, int, None]) -> Optional[float]:
    """Normalize a categorical or numeric score to a float value.

    Args:
        score: The score to normalize (can be categorical string, numeric, or None)

    Returns:
        Normalized float score (typically 0.0, 0.5, or 1.0), or None if input is None

    Raises:
        ValueError: If the score cannot be normalized
    """
    # If None, return None (missing score)
    if score is None:
        return None

    # If already numeric, return as float
    if isinstance(score, (int, float)):
        return float(score)

    # If string, look up in mapping
    if isinstance(score, str):
        score_upper = score.strip().upper()
        if score_upper in ALL_MAPPINGS:
            return ALL_MAPPINGS[score_upper]
        raise ValueError(
            f"Unknown categorical score: '{score}'. Must be one of: "
            f"{list(ALL_MAPPINGS.keys())}"
        )

    raise ValueError(f"Score must be str, int, float, or None, got {type(score)}")


def normalize_scores(
    scores: Dict[str, Union[str, float, None]],
) -> Dict[str, Optional[float]]:
    """Normalize a dictionary of categorical scores to numeric values.

    Args:
        scores: Dictionary mapping metric names to categorical or numeric scores (or None)

    Returns:
        Dictionary with all scores normalized to floats (or None for missing scores)
    """
    return {key: normalize_score(value) for key, value in scores.items()}


def get_metric_mapping(metric_name: str) -> Dict[str, float]:
    """Get the score mapping for a specific metric.

    Args:
        metric_name: Name of the metric (e.g., 'correctness', 'relevancy')

    Returns:
        Dictionary mapping categorical values to numeric scores

    Raises:
        ValueError: If metric name is not recognized
    """
    metric_mappings = {
        "correctness": CORRECTNESS_MAPPING,
        "tool_call_correctness": CORRECTNESS_MAPPING,
        "relevancy": RELEVANCY_MAPPING,
        "completeness": COMPLETENESS_MAPPING,
        "structural_clarity": STRUCTURAL_CLARITY_MAPPING,
        "scope_adherence": SCOPE_ADHERENCE_MAPPING,
        "ethical_awareness": AWARENESS_MAPPING,
    }

    metric_lower = metric_name.lower()
    if metric_lower not in metric_mappings:
        raise ValueError(
            f"Unknown metric: '{metric_name}'. Must be one of: "
            f"{list(metric_mappings.keys())}"
        )

    return metric_mappings[metric_lower]


def denormalize_score(score: float, metric_name: str) -> str:
    """Convert a numeric score back to its categorical label.

    Args:
        score: Numeric score (typically 0.0, 0.5, 1.0, or -1.0)
        metric_name: Name of the metric (determines which mapping to use)

    Returns:
        Categorical label string (e.g., 'CORRECT', 'NOT RELEVANT')

    Raises:
        ValueError: If the score or metric name is not recognized
    """
    mapping = get_metric_mapping(metric_name)
    inverse = {v: k for k, v in mapping.items()}
    if score not in inverse:
        raise ValueError(
            f"Unknown numeric score {score} for metric '{metric_name}'. "
            f"Must be one of: {list(inverse.keys())}"
        )
    return inverse[score]
