"""Judge signature optimization for maximizing human-LLM alignment."""

from langdmta_eval.optimization.alignment_metric import make_alignment_metric
from langdmta_eval.optimization.signature_optimizer import (
    JUDGE_METRICS,
    OptimizationConfig,
    SignatureOptimizer,
    build_trainset,
    build_trainset_from_paths,
)
from langdmta_eval.optimization.versioning import (
    SignatureVersion,
    load_version_history,
    save_version_history,
    version_comparison_table,
)

__all__ = [
    # Metric
    "make_alignment_metric",
    # Optimizer
    "OptimizationConfig",
    "SignatureOptimizer",
    "JUDGE_METRICS",
    "build_trainset",
    "build_trainset_from_paths",
    # Versioning
    "SignatureVersion",
    "save_version_history",
    "load_version_history",
    "version_comparison_table",
]
