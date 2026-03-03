import logging
from pathlib import Path
from typing import Any, Type

import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langdmta_lab.subagents.Analyzer.Analyzer import Analyzer
from langdmta_lab.base.base_node import BaseNode
from langdmta_lab.base.base_state import State
from langdmta_lab.config import RESPONSE_PREFIX
from langdmta_lab.utils.io import create_temp_dir

# Show full cell contents (no '...')
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CSVAnalyzerException(Exception):
    """Custom exception for CSV Analyzer errors."""

    pass


def logg_error(message: str, exception_class: Type[Exception]) -> None:
    logging.error(message)
    raise exception_class(message)


class AnalyzerNode(BaseNode):
    def __init__(
        self, name: str = "Analyzer", model_name: str = "gpt-4o-openai", **kwargs: Any
    ) -> None:
        self.llm_kwargs = kwargs
        self.model_name = model_name
        self.agent = None
        super().__init__(name)

    async def _init_agent(self) -> CompiledStateGraph:

        return await Analyzer(
            config_filename=f"{self.name}.yaml", model_name=self.model_name, **self.llm_kwargs
        ).create_agent()

    @staticmethod
    def _validate_input_file(csv_path: str) -> str:
        # Ensure it has .csv extension
        path = Path(csv_path)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")

        # Check existence
        if not path.is_file():
            logg_error(
                f"Invalid CSV file path: {csv_path}, it must ends with .csv"
                " and the file must exist.",
                CSVAnalyzerException,
            )

        return str(path)

    def _load_messages(self, state: State) -> str:
        instruction = state["task_description"]
        csv_path = state["worker_input_file"]
        csv_path = self._validate_input_file(csv_path)
        # df = pd.read_csv(csv_path)
        self.output_csv_path = f"{create_temp_dir()}/filtered_output.csv"
        self.output_plot_path = f"{create_temp_dir()}/plot.png"
        instruction += f" Load the dataframe from {csv_path}. "
        instruction += f" If it is required, save the processed df to {self.output_csv_path}. "
        instruction += f" If plotting is required, save the plot to {self.output_plot_path}. "

        return instruction

    async def _invoke_response(self, messages: str) -> str:
        self.agent = self.agent or await self._init_agent()
        assert self.agent is not None
        try:
            response = self.agent.run(messages)  # type: ignore
        except Exception as e:
            logg_error(
                f"Error during CSV analysis: {e}",
                CSVAnalyzerException,
            )
            return ""

        if not response:
            logg_error(
                "No response from agent executor.",
                CSVAnalyzerException,
            )
            return ""
        else:
            return response

    def _handover_response(self, response: str) -> Command:
        response_formatted = f"{RESPONSE_PREFIX} {self.name}:\n{response}"
        output_files = []
        if Path(self.output_csv_path).exists():
            output_files.append(self.output_csv_path)
        if Path(self.output_plot_path).exists():
            output_files.append(self.output_plot_path)
        worker_output_files = {self.name: output_files}
        handover_dict = {
            "messages": [HumanMessage(content=response_formatted, name=self.name)],
            "worker_output_files": worker_output_files,
        }
        return Command(update=handover_dict, goto="supervisor")
