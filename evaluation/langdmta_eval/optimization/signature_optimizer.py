"""DSPy judge signature optimization pipeline.

Optimizes judge signatures to maximize human-LLM alignment using
DSPy's built-in optimizers (LabeledFewShot, BootstrapFewShot, MIPROv2).
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import dspy
import numpy as np
import pandas as pd

from langdmta_eval.alignment.data_loader import ANNOTATION_METRICS, load_annotation_pair
from langdmta_eval.judges.score_mappings import denormalize_score
from langdmta_eval.judges.signatures import get_judge_signature
from langdmta_eval.optimization.alignment_metric import make_alignment_metric


@dataclass
class OptimizationConfig:
    """Configuration for signature optimization."""

    optimizer_type: Literal["labeled_fewshot", "bootstrap_fewshot", "miprov2"] = (
        "labeled_fewshot"
    )
    judge_type: Literal[
        "basic",
        "ethical",
        "capability",
        "comprehensive",
    ] = "comprehensive"
    model_name: str = "anthropic/claude-sonnet-4-5"
    max_bootstrapped_demos: int = 3
    max_labeled_demos: int = 4
    temperature: float = 0.0
    num_trials: int = 5
    seed: int = 42
    model_kwargs: dict = field(default_factory=dict)
    auto: Optional[Literal["none", "medium", "aggressive"]] = None


# -- Mapping from judge_type to which metrics are available --
JUDGE_METRICS: Dict[str, List[str]] = {
    "basic": ["correctness", "relevancy", "completeness", "structural_clarity"],
    "ethical": [
        "correctness",
        "relevancy",
        "completeness",
        "structural_clarity",
        "ethical_awareness",
    ],
    "capability": [
        "correctness",
        "relevancy",
        "completeness",
        "structural_clarity",
        "scope_adherence",
    ],
    "comprehensive": [
        "correctness",
        "relevancy",
        "completeness",
        "structural_clarity",
        "ethical_awareness",
        "scope_adherence",
    ],
}


def build_trainset(
    annotation_df: pd.DataFrame,
    judge_type: str = "comprehensive",
) -> List[dspy.Example]:
    """Convert a long-format annotation DataFrame to DSPy Examples.

    Each Example has:
    - Input fields matching the judge signature (task, output, etc.)
    - Label fields with categorical human scores (CORRECT, etc.)
    - .with_inputs() to mark input fields

    Args:
        annotation_df: Long-format DataFrame from data_loader
        judge_type: Which judge type to build examples for

    Returns:
        List of dspy.Example objects ready for optimization
    """
    target_metrics = JUDGE_METRICS.get(judge_type, ANNOTATION_METRICS)

    # Pivot from long to wide: one row per (annotation_id, annotator_id)
    cases = annotation_df.pivot_table(
        index=["annotation_id", "annotator_id"],
        columns="metric",
        values="human_score",
        aggfunc="first",
    ).reset_index()

    # Also grab the context fields from the original data
    context_cols = ["annotation_id", "annotator_id"]
    meta_fields = ["category", "test_name", "session"]
    for col in meta_fields:
        if col in annotation_df.columns:
            mapping = (
                annotation_df.drop_duplicates(["annotation_id", "annotator_id"])
                .set_index(["annotation_id", "annotator_id"])[col]
                .to_dict()
            )
            cases[col] = cases.set_index(["annotation_id", "annotator_id"]).index.map(
                lambda x: mapping.get(x, "")
            )

    # Get the original question/output/context from the raw annotation data
    # These come from the annotator CSV (one per annotation_id)
    raw_context = annotation_df.drop_duplicates("annotation_id").set_index(
        "annotation_id"
    )

    examples = []
    for _, row in cases.iterrows():
        ann_id = row["annotation_id"]

        # Skip if context not available
        if ann_id not in raw_context.index:
            continue

        ctx = raw_context.loc[ann_id]

        # Build input fields
        kwargs = {
            "task": str(ctx.get("question", "") if hasattr(ctx, "get") else ""),
            "output": str(ctx.get("agent_output", "") if hasattr(ctx, "get") else ""),
            "evaluation_context": str(
                ctx.get("evaluation_context", "") if hasattr(ctx, "get") else ""
            ),
            "tool_calls": str(ctx.get("tool_calls", "") if hasattr(ctx, "get") else ""),
            "expected_tool_calls": str(
                ctx.get("expected_tool_calls", "") if hasattr(ctx, "get") else ""
            ),
            "accepted_tool_calls": str(
                ctx.get("accepted_tool_calls", "") if hasattr(ctx, "get") else ""
            ),
        }

        # Use human justification as the ground-truth reasoning for ChainOfThought demos
        justification = str(
            ctx.get("human_justification", "") if hasattr(ctx, "get") else ""
        )
        if justification:
            kwargs["reasoning"] = justification

        # Build label fields (categorical strings)
        for metric in target_metrics:
            if metric in row.index and not pd.isna(row[metric]):
                score = float(row[metric])
                try:
                    kwargs[metric] = denormalize_score(score, metric)
                except ValueError:
                    continue

        example = dspy.Example(**kwargs).with_inputs(
            "task",
            "output",
            "evaluation_context",
            "tool_calls",
            "expected_tool_calls",
            "accepted_tool_calls",
        )
        examples.append(example)

    return examples


def build_trainset_from_paths(
    annotator_csv: str,
    llm_scores_csv: str,
    annotator_id: Optional[str] = None,
    judge_type: str = "comprehensive",
) -> List[dspy.Example]:
    """Convenience: load annotation pair and convert to trainset.

    Args:
        annotator_csv: Path to annotated CSV
        llm_scores_csv: Path to companion LLM scores CSV
        annotator_id: Optional annotator label
        judge_type: Which judge type

    Returns:
        List of dspy.Example objects
    """
    df = load_annotation_pair(annotator_csv, llm_scores_csv, annotator_id=annotator_id)
    return build_trainset(df, judge_type=judge_type)


class SignatureOptimizer:
    """Orchestrates judge signature optimization.

    Usage:
        config = OptimizationConfig(optimizer_type="labeled_fewshot")
        trainset = build_trainset(annotation_df)
        optimizer = SignatureOptimizer(config, trainset)
        optimized = optimizer.optimize()
        score = optimizer.evaluate(optimized)
        optimizer.save_optimized(optimized, "optimized_judge.json")
    """

    def __init__(
        self,
        config: OptimizationConfig,
        trainset: List[dspy.Example],
        target_metrics: Optional[List[str]] = None,
    ):
        self.config = config
        self.trainset = trainset

        if target_metrics is None:
            target_metrics = JUDGE_METRICS.get(config.judge_type, ANNOTATION_METRICS)
        self.target_metrics = target_metrics

        from langdmta_eval.evaluators.base import create_lm

        self.lm = create_lm(config)
        dspy.configure(lm=self.lm)

        # Create the student (judge to optimize)
        judge_signature = get_judge_signature(config.judge_type)
        self.student = dspy.ChainOfThought(judge_signature)

        # Create the alignment metric
        self.metric = make_alignment_metric(target_metrics=target_metrics)

    def optimize(self) -> dspy.Module:
        """Run optimization and return the optimized module.

        Returns:
            Optimized dspy.Module (ChainOfThought with few-shot demos)
        """
        if self.config.optimizer_type == "labeled_fewshot":
            optimizer = dspy.LabeledFewShot(k=self.config.max_labeled_demos)
            optimized = optimizer.compile(
                self.student,
                trainset=self.trainset,
            )

        elif self.config.optimizer_type == "bootstrap_fewshot":
            optimizer = dspy.BootstrapFewShot(
                metric=self.metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
            )
            optimized = optimizer.compile(
                self.student,
                trainset=self.trainset,
            )

        elif self.config.optimizer_type == "miprov2":
            if len(self.trainset) < 20:
                warnings.warn(
                    f"MIPROv2 works best with 50+ examples. You have {len(self.trainset)}. "
                    "Consider using labeled_fewshot or bootstrap_fewshot instead.",
                    UserWarning,
                    stacklevel=2,
                )
            optimizer = dspy.MIPROv2(
                metric=self.metric,
                num_threads=1,
                auto=self.config.auto,
                seed=self.config.seed,
            )
            optimized = optimizer.compile(
                self.student,
                trainset=self.trainset,
                num_trials=self.config.num_trials,
            )

        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

        return optimized

    def evaluate(
        self,
        module: dspy.Module,
        evalset: Optional[List[dspy.Example]] = None,
    ) -> float:
        """Evaluate a module's alignment on a dataset:
            module: The DSPy module to evaluate
            evalset: Optional evaluation set. If None, uses trainset.

        Returns:
            Mean alignment score (0.0 to 1.0)
        """
        if evalset is None:
            evalset = self.trainset

        evaluator = dspy.Evaluate(
            devset=evalset,
            metric=self.metric,
            num_threads=1,
            display_progress=False,
        )

        result = evaluator(module)
        return float(result.score) if hasattr(result, "score") else float(result)

    def save_optimized(self, module: dspy.Module, path: str) -> None:
        """Save optimized module to disk.

        Args:
            module: The optimized DSPy module
            path: File path to save to (JSON format)
        """
        module.save(path)

    def load_optimized(self, path: str) -> dspy.Module:
        """Load a previously saved optimized module.

        Args:
            path: File path to load from

        Returns:
            The loaded DSPy module
        """
        judge_signature = get_judge_signature(self.config.judge_type)
        module = dspy.ChainOfThought(judge_signature)
        module.load(path)
        return module
