"""Tool sequence validation utilities."""

import asyncio
import ast
from typing import Any, Dict, List, Tuple, Union

from ..schemas import TestCase


def check_remove_repeated(observations: List[Any]) -> Tuple[bool, List[str]]:
    """Check for repeated tool calls and return distinct tools.

    Checks for repeated tool calls based on two conditions:
    1. Two consecutive calls of the same tool
    2. Input arguments matching

    Example:
        - syn2smiles->molformer->reinvent_scoring->molformer->filtering: not repeated
        - syn2smiles->molformer->molformer->reinvent_scoring: repeated if same input

    Args:
        observations: List of observation objects with 'name' and 'input' attributes

    Returns:
        Tuple of (repeated, tool_list) where:
            - repeated: True if any repeated tool calls were found
            - tool_list: List of distinct tool names in order
    """
    repeated = False
    tool_list: List[str] = []
    prev_tool_input = None

    for obs in observations:
        # Convert input to string for comparison
        if not isinstance(obs.input, str):
            try:
                input_str = ast.literal_eval(str(obs.input))
            except Exception:
                input_str = str(obs.input)
        else:
            input_str = obs.input

        # Check if this is a repeated call
        if len(tool_list) != 0 and obs.name == tool_list[-1]:
            if input_str == prev_tool_input:
                repeated = True
        else:
            tool_list.append(obs.name)
            prev_tool_input = input_str

    return repeated, tool_list


def get_sequence_correctness(
    tool_calls: List[str], predefined_sequences: Dict[str, List[List[str]]]
) -> str:
    """Validate tool call sequence against predefined sequences.

    Args:
        tool_calls: List of tools called by the agent
        predefined_sequences: Dictionary with 'complete' and 'accepted' keys,
                            each containing lists of valid tool sequences

    Returns:
        String indicating correctness level:
            - 'complete': Exact match with a complete sequence
            - 'partial': Match with an accepted sequence
            - 'incorrect': No match with any valid sequence
    """
    if tool_calls in predefined_sequences["complete"]:
        return "complete"
    if tool_calls in predefined_sequences["accepted"]:
        return "partial"
    return "incorrect"


def validate_tool_sequence(
    tool_calls: List[str],
    expected_complete: List[List[str]],
    expected_accepted: List[List[str]],
) -> Dict[str, Any]:
    """Validate tool sequence and return detailed result.

    Args:
        tool_calls: List of tools called by the agent
        expected_complete: List of complete (ideal) tool sequences
        expected_accepted: List of accepted (partial) tool sequences

    Returns:
        Dictionary with validation results including:
            - correctness: 'complete', 'partial', or 'incorrect'
            - is_correct: Boolean indicating if sequence is valid
            - matched_sequence: The matched sequence (if any)
            - expected_sequences: All expected sequences
    """
    predefined_sequences = {
        "complete": expected_complete,
        "accepted": expected_accepted,
    }

    correctness = get_sequence_correctness(tool_calls, predefined_sequences)

    result = {
        "correctness": correctness,
        "is_complete": correctness == "complete",
        "is_partial": correctness == "partial",
        "matched_sequence": None,
        "expected_sequences": predefined_sequences,
    }

    # Find the matched sequence
    if correctness == "complete":
        result["matched_sequence"] = tool_calls
    elif correctness == "partial":
        result["matched_sequence"] = tool_calls

    return result


async def validate_tool_sequence_batch(
    test_cases: Union[List[TestCase], List[Dict[str, Any]]],
    max_concurrent: int = 10,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Validate tool sequences for a batch of test cases with concurrent execution.

    Args:
        test_cases: List of TestCase objects or dictionaries with tool call information
        max_concurrent: Maximum number of concurrent validations (default: 10)
        verbose: Whether to print progress information

    Returns:
        List of validation results, one for each test case
    """
    results = []
    total = len(test_cases)

    # Create a semaphore to limit concurrent validations
    semaphore = asyncio.Semaphore(max_concurrent)

    async def validate_with_semaphore(
        test_case: Union[TestCase, Dict[str, Any]], index: int
    ) -> Dict[str, Any]:
        async with semaphore:
            if verbose:
                print(f"Validating test case {index + 1}/{total}...", end="\r")

            # Extract data from TestCase or dict
            if isinstance(test_case, TestCase):
                tool_calls = test_case.tool_calls
                expected_complete = test_case.expected_tool_calls["complete"]
                expected_accepted = test_case.expected_tool_calls["accepted"]
            else:
                tool_calls = test_case["tool_calls"]
                expected_complete = test_case.get("expected_complete", [])
                expected_accepted = test_case.get("expected_accepted", [])

            # Run validation (make it async-friendly by using run_in_executor)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                validate_tool_sequence,
                tool_calls,
                expected_complete,
                expected_accepted,
            )

            # Add test case reference to result
            result["test_case"] = test_case

            return result

    # Run validations concurrently with rate limiting
    results = await asyncio.gather(
        *[
            validate_with_semaphore(test_case, i)
            for i, test_case in enumerate(test_cases)
        ]
    )

    if verbose:
        print(f"\nCompleted validation of {total} test cases.")

    return results
