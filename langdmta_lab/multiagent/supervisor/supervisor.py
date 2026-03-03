import os
from pathlib import Path
from types import new_class
from typing import Any, Dict, List, Literal, Union

from langgraph.graph import END
from typing_extensions import TypedDict

from langdmta_lab.base.base_node import BaseNode
from langdmta_lab.base.base_state import State
from langdmta_lab.base.base_supervisor import BaseSupervisor
from langdmta_lab.config import END_TOKEN, LANGDMTA_GRAPH_DIR, PARSING_TAG, RESPONSE_PREFIX

# from langdmta_lab.tools.utils.constants import SUPPORTED_PROPERTIES


class Supervisor(BaseSupervisor):

    def __init__(
        self,
        end_token: str = END_TOKEN,
        model_name: str = "gpt-4o",
        graph_yaml_path: Union[str, Path] = os.path.join(LANGDMTA_GRAPH_DIR, "graph.yaml"),
        prompt_fp: Union[str, Path] = os.path.join(
            LANGDMTA_GRAPH_DIR, "supervisor", "supervisor.yaml"
        ),
        member_dir: Union[str, Path] = os.path.join(LANGDMTA_GRAPH_DIR, "agents", "configs"),
        **kwargs,
    ):
        super().__init__(
            end_token=end_token,
            model_name=model_name,
            graph_yaml_path=graph_yaml_path,
            prompt_fp=prompt_fp,
            member_dir=member_dir,
            **kwargs,
        )

    # @property
    # def description_injections(self) -> Dict[str, str]:
    #    return {"{{supported_properties}}": ", ".join(SUPPORTED_PROPERTIES)}

    @property
    def start_node_name(self) -> str:
        """
        ChatInvent start node name
        """
        return "__start__"

    def create_router(self) -> object:
        annotations = {
            "next": Literal[tuple(self.options)],
            "task_description": str,
            "retrieved_information": str,
            "worker_input_file": str,
            "worker_output_files": Dict[str, List[str]],
        }

        Router = new_class(
            "Router",
            (TypedDict,),
            exec_body=lambda ns: ns.update(
                {
                    "__annotations__": annotations,
                    "__doc__": (
                        "Worker to route to next, its task description, "
                        "the `retrieved_information`, input file path for the next worker,"
                        "and a dict with format with key being the last worker name and. "
                        "the value being a list of output file paths from that worker."
                        f"If the question is answered by a worker, route to {self.end_token}."
                    ),
                }
            ),
        )

        return Router


class SupervisorNode(BaseNode):
    def __init__(
        self,
        name: str = "Supervisor",
        end_token: str = END_TOKEN,
        model_name: str = "gpt-4o-openai",
        **kwargs: Any,
    ) -> None:
        super().__init__(name)
        supervisor = Supervisor(end_token=end_token, model_name=model_name, **kwargs)
        self.router = supervisor.create_router()
        self.llm = supervisor.llm
        self.prompt = supervisor.prompt
        self.end_token = end_token
        self._last_message = None
        self._worker_output_files: Dict[str, List[str]] = {}

    def _load_messages(self, state: State) -> List[str]:
        self._last_message = state["messages"][-1].content
        self._worker_output_files = state.get("worker_output_files", {})
        memory = state["memory"]
        if not isinstance(memory, list):
            memory = [memory]
        messages = (
            [
                {"role": "system", "content": self.prompt},
            ]
            + memory
            + state["messages"]
        )
        return messages

    async def _invoke_response(self, messages: List[str]) -> Dict:
        response = await self.llm.with_structured_output(self.router).ainvoke(messages)
        return response

    def _handover_response(
        self, response: Dict
    ) -> Dict[str, Union[str, List[Any], Dict[str, List[str]]]]:
        goto = response["next"]
        self.supervisor_info = response.get("retrieved_information", "")
        worker_input_file = response.get("worker_input_file", "")
        handover_dict: Dict[str, Union[str, List[Any], Dict[str, List[str]]]] = {}
        handover_dict["retrieved_information"] = self.supervisor_info
        handover_dict["worker_output_files"] = {}
        if goto == self.end_token:
            goto = END
            final_answer = self._parse_final_answer()
            handover_dict["messages"] = [final_answer]
            handover_dict["worker_output_files"] = self._worker_output_files
        handover_dict["next"] = goto
        handover_dict["task_description"] = response["task_description"]
        handover_dict["worker_input_file"] = worker_input_file

        return handover_dict

    def _parse_final_answer(self) -> str:

        if self._last_message is None or RESPONSE_PREFIX not in self._last_message:
            """account for question that does not require calling any agent"""
            if self.supervisor_info:
                return self.supervisor_info
            else:
                return ""
        elif PARSING_TAG in self._last_message:
            """account for nodes that directly returns the unparsed output"""
            return self.supervisor_info
        else:
            final_answer = self._last_message
            return final_answer.split(RESPONSE_PREFIX)[-1].strip()

    def _strip_file_path(self, message_to_process: str) -> str:
        prompt = """
        You have only one task: remove any sentence that contains
        a file path. Example file path: `/tmp/abcd/output.csv`.
        Do not modify anything else.
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message_to_process},
        ]

        response = self.llm.invoke(messages)
        return response.content
