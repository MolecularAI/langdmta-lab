import argparse
import logging
import os
import re
import tomllib
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_core.callbacks import FileCallbackHandler
from langfuse.callback import CallbackHandler

from langdmta_lab.base.base_graph import BaseGraphExecutor
from langdmta_lab.config import LANGDMTA_GRAPH_DIR, TEST_ASSETS_DIR, TEST_QUESTION_DIR
from langdmta_lab.mcps.helpers import start_mcps, stop_mcps
from langdmta_lab.multiagent.constants import NODE_PACKAGES


class SafeFileCallbackHandler(FileCallbackHandler):
    """
    Safely handles the start of a chain by defaulting `None` values
    for `serialized` and `inputs` to empty dictionaries. This is a patch fix
    for passing FileCallbackHandler in multiagent setting

    Args:
        serialized (Optional[Dict[str, Any]]): Serialized chain data.
        inputs (Optional[Dict[str, Any]]): Input values passed to the chain.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        None
    """

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        if inputs is None:
            inputs = {}
        if serialized is None:
            serialized = {}
        super().on_chain_start(serialized, inputs, **kwargs)


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test questions and log to LangFuse")

    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of times to run the same question (default: 1)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="",
        help="Session id (will be concatenated with trial number)",
    )
    parser.add_argument(
        "--traces",
        nargs="+",
        default=None,
        help="One or more traces to test (e.g. --traces Synonym2SMILES Molformer). "
        "If omitted, runs all tests.",
    )
    parser.add_argument(
        "--log-file", type=str, default="test_questions.log", help="Log file name"
    )
    parser.add_argument(
        "--mock-category",
        type=str,
        default="Workflow",
        help="Category for which to use mock tools (default: Workflow)",
    )

    return parser.parse_args()


def read_test_data(test_data_file: str) -> dict:
    with open(test_data_file, "rb") as file:
        test_data = tomllib.load(file)
    return test_data


async def main():

    args = parse_command_line()
    log_file = args.log_file
    # Clear handlers to ensure new logging setup works
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    file_handler = SafeFileCallbackHandler(log_file)

    load_dotenv()
    test_data_path = os.path.join(TEST_QUESTION_DIR, "test_questions.toml")
    test_data = read_test_data(test_data_path)

    for i in range(args.num_trials):
        for category in test_data.keys():
            if (args.mock_category.lower() == "all") or (category == args.mock_category):
                os.environ["USE_MOCK"] = "true"
            else:
                os.environ["USE_MOCK"] = "false"
            mcp_processes = start_mcps()
            if not mcp_processes:
                logging.error("MCP servers failed to start.")
                continue

            if args.traces:
                traces = [
                    trace for trace in args.traces if trace in test_data[category].keys()
                ]
            else:
                traces = test_data[category].keys()

            for name in traces:
                langfuse_handler = CallbackHandler(
                    session_id=f"{args.session_id}_trial_{i+1}",  # Identifier for this session
                    user_id=os.environ[
                        "LANGFUSE_USER_ID"
                    ],  # Identifier for the user or operator
                    trace_name=name,  # Custom name for this trace
                    tags=[category, "test"],  # Custom tag for this trace
                )
                agent = BaseGraphExecutor(
                    node_packages=NODE_PACKAGES,
                    graph_config=os.path.join(LANGDMTA_GRAPH_DIR, "graph.yaml"),
                    callback_handlers=[file_handler, langfuse_handler],
                )

                for test_pairs in test_data[category][name]:
                    question, _ = test_pairs["question"], test_pairs["answer"]
                    if "input_file_in_test_assets" in test_pairs.keys():
                        filenames = re.findall(r"\b[\w-]+\.(?:csv|sq|sdf|zip|in)\b", question)
                        for file in filenames:
                            question = question.replace(
                                file, os.path.join(TEST_ASSETS_DIR, file)
                            )
                    logging.info(
                        f"========== Trial {i+1}: Running question from "
                        f"{category} {name}: {question}"
                    )
                    try:
                        output = await agent.run(question)
                        if isinstance(output, tuple):
                            response, worker_output_files = output
                            logging.info(f"Worker output files: {worker_output_files}")
                        else:
                            response = output
                        logging.info(f"Final answer: {response}")
                        logging.info(
                            f"LangFuse Trace URL: {langfuse_handler.langfuse.get_trace_url()}"
                        )
                    except Exception as e:
                        logging.error(e)
                        logging.info(
                            f"LangFuse Trace URL: {langfuse_handler.langfuse.get_trace_url()}"
                        )
                        continue

            stop_mcps(mcp_processes)


# Run from command line for example: python tests/test_questions.py
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
