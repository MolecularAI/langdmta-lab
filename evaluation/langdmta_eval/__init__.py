"""LangDMTA Evaluation Package.

A package for evaluating drug discovery agentic system outputs.
"""

from langdmta_eval.schemas import (
    BatchEvaluationResult,
    EvaluationResult,
    TestCase,
)
from langdmta_eval.evaluators.base import BaseEvaluator, EvaluatorConfig, build_model_kwargs, create_lm
from langdmta_eval.evaluators.dspy_evaluator import DSPyEvaluator
from langdmta_eval.judges.score_mappings import normalize_score, normalize_scores
from langdmta_eval.judges.signatures import (
    BasicJudgeSignature,
    CapabilityJudgeSignature,
    ComprehensiveJudgeSignature,
    EthicalJudgeSignature,
    get_judge_signature,
)
from langdmta_eval.metrics.tool_validation import (
    check_remove_repeated,
    get_sequence_correctness,
    validate_tool_sequence,
    validate_tool_sequence_batch,
)
from langdmta_eval.schemas import (
    load_results_from_json,
    save_results_to_json,
)
from langdmta_eval.visualization.visualize import (
    plot_metric_boxplots,
    plot_metric_correlation,
    plot_metric_pairplot,
    plot_radar_chart,
    plot_score_by_category,
    plot_score_heatmap,
    plot_score_histograms,
    results_to_dataframe,
)

# Alignment analysis
from langdmta_eval.alignment import (
    ANNOTATION_METRICS,
    AlignmentResult,
    InterAnnotatorResult,
    alignment_summary_table,
    compute_all_alignments,
    compute_all_inter_annotator,
    get_paired_scores,
    inter_annotator_summary_table,
    load_all_annotations,
    load_annotation_pair,
    load_multiple_annotators,
    plot_alignment_bar_chart,
    plot_alignment_comparison,
    plot_all_confusion_matrices,
    plot_multi_alignment_comparison,
)

# Score denormalization
from langdmta_eval.judges.score_mappings import denormalize_score

# Optimization
from langdmta_eval.optimization import (
    OptimizationConfig,
    SignatureOptimizer,
    SignatureVersion,
    build_trainset,
    make_alignment_metric,
    version_comparison_table,
)

__version__ = "0.1.0"

__all__ = [
    # Data models
    "TestCase",
    "EvaluationResult",
    "BatchEvaluationResult",
    # Evaluators
    "BaseEvaluator",
    "DSPyEvaluator",
    "EvaluatorConfig",
    "build_model_kwargs",
    "create_lm",
    # Judge signatures
    "BasicJudgeSignature",
    "EthicalJudgeSignature",
    "CapabilityJudgeSignature",
    "ComprehensiveJudgeSignature",
    "get_judge_signature",
    # Score mappings
    "normalize_score",
    "normalize_scores",
    "denormalize_score",
    # Tool validation
    "check_remove_repeated",
    "get_sequence_correctness",
    "validate_tool_sequence",
    "validate_tool_sequence_batch",
    # Utils
    "save_results_to_json",
    "load_results_from_json",
    # Visualization
    "plot_score_histograms",
    "plot_score_by_category",
    "plot_score_heatmap",
    "plot_metric_correlation",
    "plot_metric_boxplots",
    "plot_metric_pairplot",
    "plot_radar_chart",
    "results_to_dataframe",
    # Alignment
    "ANNOTATION_METRICS",
    "AlignmentResult",
    "InterAnnotatorResult",
    "load_annotation_pair",
    "load_multiple_annotators",
    "load_all_annotations",
    "get_paired_scores",
    "compute_all_alignments",
    "compute_all_inter_annotator",
    "alignment_summary_table",
    "inter_annotator_summary_table",
    "plot_all_confusion_matrices",
    "plot_alignment_bar_chart",
    "plot_alignment_comparison",
    "plot_multi_alignment_comparison",
    # Optimization
    "make_alignment_metric",
    "OptimizationConfig",
    "SignatureOptimizer",
    "build_trainset",
    "SignatureVersion",
    "version_comparison_table",
]
