import argparse
import ast
import os
import re
import tomllib
from operator import attrgetter
from typing import Any, List

import pandas as pd
from dotenv import load_dotenv
from langfuse import Langfuse

from langdmta_lab.config import TEST_QUESTION_DIR
from langdmta_lab.utils.constants import TOOL_NAMES


class LangfuseSummarizer:
    """
    Summarizes Langfuse sessions by extracting key metrics.

    Args:
        session_ids (List[str]): List of Langfuse session IDs to summarize.
        concat (bool): If True, concatenates all session summaries into a single output file.
        output_path (str): Directory to save the output files.
    """

    def __init__(
        self,
        session_ids: List[str],
        concatenate_sessions: bool,
        output_path: str,
    ) -> None:
        load_dotenv()
        self.session_ids = session_ids
        self.concat = concatenate_sessions
        self.output_path = output_path
        self._read_test_question_dict()
        self.langfuse = Langfuse()
        self.langfuse_link = os.environ["LANGFUSE_HOST"]

    def _read_test_question_dict(self):
        test_data = self._read_test_data()
        self._create_test_question_dict(test_data)

    def _read_test_data(self) -> dict:
        test_data_path = os.path.join(TEST_QUESTION_DIR, "test_questions.toml")
        with open(test_data_path, "rb") as file:
            test_data = tomllib.load(file)
        return test_data

    def _create_test_question_dict(self, test_data: dict):
        test_question_dict = {}
        for category in test_data.keys():
            for name in test_data[category].keys():
                for test_pairs in test_data[category][name]:
                    question = test_pairs["question"]
                    test_question_dict[question] = {
                        "category": category,
                        "name": name,
                        "sequences": test_pairs["sequences"],
                    }
        self.question_dict = test_question_dict

    def _fetch_session_traces(self, session_id: str) -> List[Any]:
        """Fetches the data of all traces from a session."""
        session_traces = self.langfuse.fetch_traces(session_id=session_id)
        all_session_trace_ids = [trace.id for trace in session_traces.data]
        if session_traces.meta.total_pages > 1:
            for page in range(2, session_traces.meta.total_pages + 1):
                next_page_traces = self.langfuse.fetch_traces(session_id=session_id, page=page)
                all_session_trace_ids.extend([trace.id for trace in next_page_traces.data])

        all_session_trace_data = [
            self.langfuse.fetch_trace(trace).data for trace in all_session_trace_ids
        ]
        return all_session_trace_data

    def _summarize(self, trace_data: Any) -> dict:
        """Extracts fields and summarizes a trace."""
        trace_input, trace_output = self._prepare_trace(trace_data)
        tool_call_obs, tokens, errors = self._prepare_obs(trace_data.observations)
        repeated, tool_calls = check_remove_repeated(tool_call_obs)
        question = self._prepare_question(trace_input)
        question_data = self.question_dict.get(question, {})
        category = question_data.get("category")
        test_name = question_data.get("name")
        predefined_tools = question_data.get("sequences", [])
        correct_tools = (
            get_sequence_correctness(tool_calls, predefined_tools) if predefined_tools else ""
        )

        summary = {
            "trace_id": trace_data.id,
            "trace_link": self.langfuse_link + trace_data.html_path,
            "timestamp": trace_data.timestamp,
            "input": trace_input,
            "output": trace_output,
            "time": trace_data.latency,
            "tokens": tokens,
            "tool_calls": tool_calls,
            "repeated_tool_calls": repeated,
            "predefined_tools": predefined_tools,
            "tool_call_correctness": correct_tools,
            "errors": errors,
            "category": category,
            "test_name": test_name,
        }

        return summary

    def run(self):
        all_session_summaries = []

        for session_id in self.session_ids:
            session_summary = self._get_session_summary(session_id)
            if self.concat:
                all_session_summaries.append(session_summary)
            else:
                session_summary.to_csv(
                    os.path.join(self.output_path, f"{session_id.replace('/', '_')}.csv"),
                    index=False,
                )

        if self.concat:
            session_ids_renamed = [session.replace("/", "_") for session in self.session_ids]
            all_sessions_df = pd.concat(all_session_summaries)
            all_sessions_df = all_sessions_df.sort_values(by=["input", "session"])

            prefixes = [re.sub(r"_trial_\d+$", "", session) for session in session_ids_renamed]
            if len(set(prefixes)) == 1:
                filename = os.path.join(
                    self.output_path,
                    prefixes[0]
                    + "".join(
                        [
                            re.findall(r"_trial_\d+$", session)[0]
                            for session in session_ids_renamed
                        ]
                    )
                    + ".csv",
                )
            else:
                filename = os.path.join(
                    self.output_path, f"{'_'.join(session_ids_renamed)}.csv"
                )

            all_sessions_df.to_csv(filename, index=False)

    def _get_session_summary(self, session_id: str) -> pd.DataFrame:
        all_session_traces = self._fetch_session_traces(session_id)
        session_summary = [self._summarize(trace) for trace in all_session_traces]
        df = pd.DataFrame(session_summary)
        df["session"] = session_id
        return df

    def _prepare_trace(self, trace_data: Any) -> tuple[str, str]:
        """Extracts input and output from a trace."""

        trace_input = trace_data.input["messages"][0][-1]
        if isinstance(trace_data.output, dict) and trace_data.output != {}:
            trace_output = trace_data.output["messages"][-1]["content"]
        else:
            trace_output = trace_data.output

        return trace_input, trace_output

    def _prepare_obs(self, observations: List[Any]) -> tuple[Any, int, dict]:
        """
        Processes a list of observations. Returns the list of tool calls
        excluding SMILESValidator, total token count and aggregated errors.
        """

        tool_call_obs = []
        total_tokens = []
        errors = []

        for obs in observations:
            total_tokens.append(obs.totalTokens)

            if obs.name in TOOL_NAMES and obs.name != "smilesvalidator":
                tool_call_obs.append(obs)

            if "ERROR" in obs.level:
                if obs.status_message not in errors:
                    errors.append(obs.status_message)

        tool_call_obs.sort(key=attrgetter("start_time"))

        return tool_call_obs, sum(total_tokens), errors

    def _prepare_question(self, question: str) -> str:
        """Processes a question by removing the directory from file paths."""
        filenames = re.findall(r"\S*\/\S+\.(?:csv|sq|sdf|zip|in)\b", question)
        for file in filenames:
            question = question.replace(file, file.split("/")[-1])
        return question


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--session-ids",
        type=lambda s: [str(item) for item in s.split(",")],
        help="List of session ids (comma separated)",
    )
    parser.add_argument(
        "--concatenate-sessions",
        type=bool,
        default=False,
        help="Concatenate all sessions into 1 output file (default: False)",
    )
    parser.add_argument("--output-folder", type=str, default="", help="Output folder path")

    return parser.parse_args()


def check_remove_repeated(observations: List[Any]) -> tuple[bool, List[str]]:
    """
    Checks for repeated tool calls and returns whether there were any repetition along with the
    list of distinct tools.
    Conditions for repeated: two consecutive calls of the same tool + input arguments matching.
    Example:
    - syn2smiles->molformer->reinvent_scoring->molformer->filtering: not repeated
    - syn2smiles->molformer->molformer->reinvent_scoring: repeated if same input in molformer
    """
    repeated = False
    tool_list = []
    prev_tool_input = None

    for obs in observations:
        if not isinstance(obs.input, str):
            try:
                input_str = ast.literal_eval(str(obs.input))
            except Exception:
                input_str = str(obs.input)
        else:
            input_str = obs.input

        if len(tool_list) != 0 and obs.name == tool_list[-1]:
            if input_str == prev_tool_input:
                repeated = True
        else:
            tool_list.append(obs.name)
            prev_tool_input = input_str

    return repeated, tool_list


def get_sequence_correctness(tool_calls: List[str], predefined_sequences: dict) -> str:
    if tool_calls in predefined_sequences["complete"]:
        return "complete"
    if tool_calls in predefined_sequences["accepted"]:
        return "partial"
    return "incorrect"


# python evaluation/langfuse_summary.py --session-ids trial_1,trial_2
if __name__ == "__main__":
    args = parse_command_line()
    load_dotenv()

    summarizer = LangfuseSummarizer(
        args.session_ids, args.concatenate_sessions, args.output_folder
    )
    summarizer.run()
