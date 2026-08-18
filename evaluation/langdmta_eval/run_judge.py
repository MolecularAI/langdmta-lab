import argparse
import asyncio
from pathlib import Path

import pandas as pd

from langdmta_eval import (
    DSPyEvaluator,
    EvaluatorConfig,
    TestCase,
    load_results_from_json,
    save_results_to_json,
)
from langdmta_eval.evaluators.base import build_model_kwargs

PROVIDERS = ("azure", "bedrock", "gemini", "vertex_ai", "direct")


async def main(
    results_path: str,
    model_name: str,
    provider: str = "azure",
    judge_type: str = "comprehensive",
    # temperature: float = 0,
    max_concurrent: int = 5,
    force_rerun: bool = False,
    optimized_path: str | None = None,
    save_path: str | None = None,
):
    model_kwargs = build_model_kwargs(provider)

    config = EvaluatorConfig(
        model_name=model_name,
        judge_type=judge_type,
        # temperature=temperature,
        max_concurrent=max_concurrent,
        model_kwargs=model_kwargs,
        optimized_path=optimized_path,
    )

    print(f"Judge Configuration:")
    print(f"  Provider: {provider}")
    print(f"  Model: {config.model_name}")
    print(f"  Judge Type: {config.judge_type}")
    print(f"  Max Concurrent: {config.max_concurrent}")
    if optimized_path:
        print(f"  Optimized Path: {optimized_path}")

    evaluator = DSPyEvaluator.from_config(config)

    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} test cases from {results_path}")

    model_tag = model_name.replace("/", "_")
    opt_tag = f"_optimized_{Path(optimized_path).stem}" if optimized_path else ""
    if not save_path:
        save_path = results_path.replace(
            ".csv", f"_LLM_evaluation_results_{judge_type}_{model_tag}{opt_tag}.json"
        )
    print("Results will be saved to:", save_path)
    if Path(save_path).exists() and not force_rerun:
        print(f"Cached results found at: {save_path}, loading from cache...")
        csv_results = load_results_from_json(save_path)
        print(f"Loaded {len(csv_results)} cached evaluation results")
    else:
        test_cases = []
        for _, row in df.iterrows():
            test_case_args = {
                "question": row["input"],
                "agent_output": row["output"],
                "tool_calls": eval(row["tool_calls"]),
                "expected_tool_calls": (
                    lambda v: v if isinstance(v, dict) else {"complete": v, "accepted": []}
                )(eval(row["predefined_tools"])),
                "category": row["category"],
                "test_name": row["test_name"],
                "session": row["session"],
                "note": row["note"],
                "trace_id": row["trace_id"],
                "tool_call_correctness": row["tool_call_correctness"],
            }

            if "question_variation" in df.columns and pd.notna(
                row.get("question_variation")
            ):
                test_case_args["question_variation"] = row["question_variation"]

            test_cases.append(TestCase(**test_case_args))

        print(f"Evaluating {len(test_cases)} test cases...")
        csv_results = await evaluator.evaluate_batch(test_cases, verbose=True)
        print(f"Evaluated {len(csv_results)} test cases")

    save_results_to_json(csv_results, save_path)
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM judge evaluation on test results"
    )
    parser.add_argument("results_path", help="Path to the CSV file with test results")
    parser.add_argument(
        "--model", default="azure/gpt-5", help="Model name for the judge"
    )
    parser.add_argument(
        "--provider", default="azure", choices=PROVIDERS, help="LLM provider routing"
    )
    parser.add_argument("--judge-type", default="comprehensive", help="Judge type")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Model temperature"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5, help="Max concurrent evaluations"
    )
    parser.add_argument(
        "--force-rerun", action="store_true", help="Skip cache and re-evaluate"
    )
    parser.add_argument(
        "--optimized-path",
        default=None,
        help="Path to an optimized judge JSON (saved by DSPy module.save()). "
        "Loads optimized few-shot demos into the judge.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Path to save the evaluation results JSON. If not provided, saves alongside the input CSV file.",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            results_path=args.results_path,
            model_name=args.model,
            provider=args.provider,
            judge_type=args.judge_type,
            # temperature=args.temperature,
            max_concurrent=args.max_concurrent,
            force_rerun=args.force_rerun,
            optimized_path=args.optimized_path,
            save_path=args.save_path,
        )
    )
