"""Base evaluator interface."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import dspy

from langdmta_eval.schemas import EvaluationResult, TestCase


@dataclass
class EvaluatorConfig:
    """Configuration for evaluators.

    Attributes:
        model_name: Name of the model to use (e.g., 'azure/gpt-5', 'bedrock/anthropic.claude-sonnet-4-5-20250514-v1:0')
        judge_type: Type of judge to use ('basic', 'ethical', 'capability', 'comprehensive')
        temperature: Temperature for the model
        max_concurrent: Maximum number of concurrent evaluations
        cache_dir: Optional directory for caching results
        model_kwargs: Provider-specific kwargs forwarded to dspy.LM() / LiteLLM
    """

    model_name: str = "anthropic/claude-sonnet-4-5"
    judge_type: Literal[
        "basic",
        "ethical",
        "capability",
        "comprehensive",
    ] = "basic"
    # temperature: float = 0.0
    max_concurrent: int = 5
    cache_dir: Optional[str] = None
    model_kwargs: dict = field(default_factory=dict)
    optimized_path: Optional[str] = None


def build_model_kwargs(provider: str) -> dict:
    """Build provider-specific kwargs for dspy.LM() / LiteLLM."""
    gateway_url = os.getenv("AI_GATEWAY_URL", "")
    api_key = os.getenv("AI_GATEWAY_KEY", "")
    print(f"gateway_url: {gateway_url}")
    if provider == "azure":
        return {
            "api_key": api_key,
            "api_base": f"{gateway_url}/azure-openai",
            "api_version": "2025-02-01-preview",
        }
    elif provider == "bedrock":
        return {
            "api_key": api_key,
            "api_base": f"{gateway_url}/bedrock",
        }
    elif provider == "gemini":
        os.environ["OPENAI_API_KEY"] = api_key
        return {
            "api_key": api_key,
            "api_base": f"{gateway_url}/vertex-ai-openai",
        }

    return {}


def create_lm(config):
    """Create a dspy.LM from a config with model_name, temperature, and model_kwargs."""
    return dspy.LM(
        model=config.model_name,
        # temperature=config.temperature,
        cache=False, # Turn of cache to enable re-evaluation
        **config.model_kwargs,
    )


class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""

    def __init__(self, config: EvaluatorConfig):
        """Initialize the evaluator.

        Args:
            config: Evaluator configuration
        """
        self.config = config

    @abstractmethod
    async def evaluate(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case.

        Args:
            test_case: The test case to evaluate

        Returns:
            Evaluation result
        """
        pass

    @abstractmethod
    async def evaluate_batch(
        self, test_cases: List[TestCase], verbose: bool = False
    ) -> List[EvaluationResult]:
        """Evaluate a batch of test cases.

        Args:
            test_cases: List of test cases to evaluate
            verbose: Whether to print progress information

        Returns:
            List of evaluation results
        """
        pass

    @classmethod
    def from_config(cls, config: EvaluatorConfig) -> "BaseEvaluator":
        """Create an evaluator from a configuration.

        Args:
            config: Evaluator configuration

        Returns:
            Configured evaluator instance
        """
        return cls(config)
