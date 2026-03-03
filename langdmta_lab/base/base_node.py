import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langgraph.types import Command

from langdmta_lab.base.base_state import State
from langdmta_lab.config import RESPONSE_PREFIX

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BaseNode(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    async def __call__(self, state: State) -> Union[Command, Dict]:
        messages = self._load_messages(state)
        logging.info(f"{self.name} - input - {messages}")
        response = await self._invoke_response(messages)
        logging.info(f"{self.name} - response - {response}")
        state_update = self._handover_response(response)
        logging.info(f"{self.name} - handover - {state_update}")
        return state_update

    @abstractmethod
    def _load_messages(self, state: State) -> Any:
        pass

    @abstractmethod
    async def _invoke_response(self, messages: Any) -> Any:
        pass

    @abstractmethod
    def _handover_response(self, response: Any) -> Union[Command, Dict]:
        pass


class BaseAgentNode(BaseNode):
    def __init__(self, name: str, model_name: str, task_description: bool = True) -> None:
        super().__init__(name)
        self.model_name = model_name
        self.agent = None
        self.use_task_description = task_description

    @abstractmethod
    async def _init_agent(self) -> Runnable:
        """
        Subclasses must implement this to return the agent used for invocation.
        """
        pass

    def _load_messages(self, state: State) -> State:
        """
        Load all previous messages / create task description.
        """
        if self.use_task_description:
            return self._pack_task_description(state)
        return self._pack_memory(state)

    def _pack_task_description(self, state: State) -> State:
        """
        Expose the task description instead of the full question
        """
        node_state = State()
        messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        messages += [HumanMessage(state["task_description"])]
        worker_input_file = state.get("worker_input_file", "")
        if worker_input_file:
            messages += [HumanMessage(f"The input file is located at {worker_input_file}.")]
        node_state["messages"] = messages
        return node_state

    def _pack_memory(self, state: State) -> State:
        """
        Expose the question and output of previous workers, no task description.
        """
        node_state = State()
        memory = state["memory"]
        if not isinstance(memory, list):
            memory = [memory]
        node_state["messages"] = memory + state["messages"]
        worker_input_file = state.get("worker_input_file", "")
        if worker_input_file:
            node_state["messages"] += [
                HumanMessage(f"The input file is located at {worker_input_file}.")
            ]
        return node_state

    async def _invoke_response(self, messages: Dict) -> str:
        self.agent = self.agent or await self._init_agent()
        assert self.agent is not None
        response = await self.agent.ainvoke(messages)
        final_response = response["messages"][-1].content
        return final_response

    def _handover_response(self, response: str) -> Command:
        response_formatted = f"{RESPONSE_PREFIX} {self.name}:\n{response}"
        handover_dict = {
            "messages": [HumanMessage(content=response_formatted, name=self.name)]
        }
        return Command(update=handover_dict, goto="supervisor")
