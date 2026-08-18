import argparse
import logging
import os
import re
import tomllib
from typing import Any, Dict, Optional

import nest_asyncio
from dotenv import load_dotenv
from langchain_core.callbacks import FileCallbackHandler
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse, get_client, propagate_attributes

from langdmta_lab.base.base_graph import BaseGraphExecutor
from langdmta_lab.config import LANGDMTA_GRAPH_DIR, TEST_ASSETS_DIR, TEST_QUESTION_DIR
from langdmta_lab.mcps.helpers import start_mcps, stop_mcps
from langdmta_lab.multiagent.constants import NODE_PACKAGES

nest_asyncio.apply()


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
    parser.add_argument(
        "--test-questions-path",
        type=str,
        default=os.path.join(TEST_QUESTION_DIR, "test_questions.toml"),
        help="Path to test questions TOML file (default: tests/assets/test_questions.toml)",
    )

    return parser.parse_args()


def read_test_data(test_data_file: str) -> dict:
    with open(test_data_file, "rb") as file:
        test_data = tomllib.load(file)
    return test_data


def setup_logging(log_file: str) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def set_mock_environment(category: str, mock_category: str) -> None:
    if mock_category.lower() == "all" or category == mock_category:
        os.environ["USE_MOCK"] = "true"
    else:
        os.environ["USE_MOCK"] = "false"


def get_traces_for_category(category_data: dict, selected_traces: list[str] | None):
    if selected_traces:
        return [trace for trace in selected_traces if trace in category_data]
    return category_data.keys()


def create_agent(file_handler: SafeFileCallbackHandler) -> BaseGraphExecutor:
    langfuse_handler = CallbackHandler()
    return BaseGraphExecutor(
            node_packages=NODE_PACKAGES,
            graph_config=os.path.join(LANGDMTA_GRAPH_DIR, "graph.yaml"),
            callback_handlers=[file_handler, langfuse_handler],
        )


def resolve_question_paths(question: str, test_pairs: dict) -> str:
    if "input_file_in_test_assets" not in test_pairs:
        return question

    filenames = re.findall(r"\b[\w-]+\.(?:csv|sq|sdf|zip|in)\b", question)
    for file in filenames:
        question = question.replace(file, os.path.join(TEST_ASSETS_DIR, file))
    return question


def log_trace_url(langfuse: Langfuse) -> None:
    try:
        trace_id = langfuse.get_current_trace_id()
        trace_url = langfuse.get_trace_url(trace_id=trace_id)
        logging.info(f"Langfuse Trace URL: {trace_url}")
    except Exception:
        pass


async def run_single_question(
    agent: BaseGraphExecutor,
    langfuse: Langfuse,
    question: str,
    category: str,
    name: str,
    trial_num: int,
    session_id: str,
    user_id: str,
) -> None:
    logging.info(
        f"========== Trial {trial_num}: Running question from "
        f"{category} {name}: {question}"
    )
    with langfuse.start_as_current_observation(
        as_type="span",
        name=name,
        input={
            "question": question,
            "category": category,
            "trace": name,
            "trial": trial_num,
        },
    ) as root_span:
        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            tags=[category, "test"],
            trace_name=name,
            metadata={"test_script": "True", "trial": str(trial_num)},
        ):
            try:
                output = await agent.run(question)
                trace_url = langfuse.get_trace_url()

                if isinstance(output, tuple):
                    response, worker_output_files = output
                    logging.info(f"Worker output files: {worker_output_files}")
                else:
                    response = output

                root_span.update_trace(output={"final_answer": response})

                logging.info(f"Final answer: {response}")
                logging.info(f"Langfuse Trace URL: {trace_url}")

            except Exception as e:
                logging.error(e)
                log_trace_url(langfuse)


async def run_trace_tests(
    langfuse: Langfuse,
    file_handler: SafeFileCallbackHandler,
    category: str,
    category_data: dict,
    traces,
    trial_num: int,
    session_id: str,
    user_id: str,
) -> None:
    for name in traces:
        agent = create_agent(file_handler)

        for test_pairs in category_data[name]:
            question = resolve_question_paths(test_pairs["question"], test_pairs)

            await run_single_question(
                agent=agent,
                langfuse=langfuse,
                question=question,
                category=category,
                name=name,
                trial_num=trial_num,
                session_id=session_id,
                user_id=user_id,
            )


async def run_trial(
    trial_num: int,
    args: argparse.Namespace,
    test_data: dict,
    langfuse: Langfuse,
    file_handler: SafeFileCallbackHandler,
    user_id: str,
) -> None:
    session_id = f"{args.session_id}_trial_{trial_num}"

    for category, category_data in test_data.items():
        set_mock_environment(category, args.mock_category)

        mcp_processes = start_mcps()
        if not mcp_processes:
            logging.error("MCP servers failed to start.")
            continue

        traces = get_traces_for_category(category_data, args.traces)

        await run_trace_tests(
            langfuse=langfuse,
            file_handler=file_handler,
            category=category,
            category_data=category_data,
            traces=traces,
            trial_num=trial_num,
            session_id=session_id,
            user_id=user_id,
        )

        stop_mcps(mcp_processes)

async def main():

    args = parse_command_line()

    load_dotenv()
    langfuse = get_client()
    user_id = os.environ["LANGFUSE_USER_ID"]

    test_data_path = args.test_questions_path
    test_data = read_test_data(test_data_path)

    with SafeFileCallbackHandler(args.log_file) as file_handler:
        for i in range(args.num_trials):
            await run_trial(
                trial_num=i + 1,
                args=args,
                test_data=test_data,
                langfuse=langfuse,
                file_handler=file_handler,
                user_id=user_id,
            )

    langfuse.flush()


# Run from command line for example: python tests/test_questions.py
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
