"""DSPy-based evaluator implementation."""

import asyncio
from typing import List

import dspy

from langdmta_eval.schemas import EvaluationResult, TestCase
from langdmta_eval.evaluators.base import BaseEvaluator, EvaluatorConfig, create_lm
from langdmta_eval.judges.score_mappings import normalize_scores
from langdmta_eval.judges.signatures import get_judge_signature


class DSPyEvaluator(BaseEvaluator):
    """DSPy-based evaluator using Chain of Thought reasoning."""

    def __init__(self, config: EvaluatorConfig):
        """Initialize the DSPy evaluator.

        Args:
            config: Evaluator configuration
        """
        super().__init__(config)

        lm = create_lm(config)
        dspy.configure(lm=lm)

        # Get the appropriate judge signature
        judge_signature = get_judge_signature(config.judge_type)
        self.judge = dspy.ChainOfThought(judge_signature)

        if config.optimized_path:
            from pathlib import Path

            if not Path(config.optimized_path).exists():
                raise FileNotFoundError(
                    f"Optimized judge not found: {config.optimized_path}"
                )
            self.judge.load(config.optimized_path)

    async def evaluate(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case.

        Args:
            test_case: The test case to evaluate

        Returns:
            Evaluation result with normalized scores
        """
        # Prepare inputs for the judge
        evaluation_context_str = test_case.note or ""
        tool_calls_str = str(test_case.tool_calls)
        expected_tool_calls_str = str(test_case.expected_tool_calls.get("complete", []))
        accepted_tool_calls_str = str(test_case.expected_tool_calls.get("accepted", []))
        # question_variation_str = str(test_case.question_variation or 0)

        # Run the judge (DSPy is synchronous, but we wrap in async for interface)
        loop = asyncio.get_event_loop()
        judgment = await loop.run_in_executor(
            None,
            lambda: self.judge(
                task=test_case.question,
                output=test_case.agent_output,
                evaluation_context=evaluation_context_str,
                tool_calls=tool_calls_str,
                expected_tool_calls=expected_tool_calls_str,
                accepted_tool_calls=accepted_tool_calls_str,
                # question_variation=question_variation_str,
            ),
        )

        # Extract scores from judgment
        raw_scores = {}
        # metrics
        if hasattr(judgment, "correctness"):
            raw_scores["correctness"] = judgment.correctness
        if hasattr(judgment, "relevancy"):
            raw_scores["relevancy"] = judgment.relevancy
        if hasattr(judgment, "structural_clarity"):
            raw_scores["structural_clarity"] = judgment.structural_clarity
        if hasattr(judgment, "completeness"):
            raw_scores["completeness"] = judgment.completeness
        if hasattr(judgment, "ethical_awareness"):
            raw_scores["ethical_awareness"] = judgment.ethical_awareness
        if hasattr(judgment, "scope_adherence"):
            raw_scores["scope_adherence"] = judgment.scope_adherence

        # Normalize scores to numeric values
        normalized_scores = normalize_scores(raw_scores)

        # Extract reasoning
        reasoning = getattr(judgment, "notes", "")

        return EvaluationResult(
            test_case=test_case,
            scores=normalized_scores,
            reasoning=reasoning,
            raw_output=judgment,
        )

    async def evaluate_batch(
        self, test_cases: List[TestCase], verbose: bool = False
    ) -> List[EvaluationResult]:
        """Evaluate a batch of test cases with concurrent execution.

        Args:
            test_cases: List of test cases to evaluate
            verbose: Whether to print progress information

        Returns:
            List of evaluation results
        """
        results = []
        total = len(test_cases)

        # Create a semaphore to limit concurrent evaluations
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def evaluate_with_semaphore(
            test_case: TestCase, index: int
        ) -> EvaluationResult:
            async with semaphore:
                if verbose:
                    print(f"Evaluating test case {index + 1}/{total}...", end="\r")
                for attempt in range(3):
                    try:
                        return await self.evaluate(test_case)
                    except Exception as e:
                        if attempt == 2:
                            raise
                        if verbose:
                            print(
                                f"\nTest case {index + 1} attempt {attempt + 1} failed ({type(e).__name__}), retrying in {2 ** attempt}s..."
                            )
                        await asyncio.sleep(2**attempt)

        # Run evaluations concurrently with rate limiting
        raw_results = await asyncio.gather(
            *[
                evaluate_with_semaphore(test_case, i)
                for i, test_case in enumerate(test_cases)
            ],
            return_exceptions=True,
        )

        results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, BaseException):
                if verbose:
                    print(f"\nTest case {i + 1} failed: {type(r).__name__}: {r}")
            else:
                results.append(r)

        if verbose:
            failed = len(raw_results) - len(results)
            print(f"\nCompleted evaluation of {total} test cases ({failed} failed).")

        return results
