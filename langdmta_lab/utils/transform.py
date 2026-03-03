import math


def sigmoid(threshold: float, threshold_p: float, k: float) -> float:
    """Compute and return the center value corresponding to a sigmoid with the given threshold and slope."""
    center = threshold + (math.log10(1.0 / threshold_p - 1.0) / k)
    return center


def reverse_sigmoid(threshold: float, threshold_p: float, k: float) -> float:
    """Compute and return the center value corresponding to the reverse sigmoid for the given threshold."""
    center = threshold + (math.log10(1.0 / (1.0 - threshold_p) - 1.0) / k)
    return center


def double_sigmoid(
    threshold_low: float, threshold_high: float, threshold_p: float, k: float
) -> tuple[float, float]:
    """Return the low and high centers for a double-sigmoid defined by the given thresholds."""
    center_low = sigmoid(threshold_low, threshold_p, k)
    center_high = reverse_sigmoid(threshold_high, threshold_p, k)
    return center_low, center_high
