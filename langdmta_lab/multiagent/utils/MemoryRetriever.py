import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from langdmta_lab.base.base_node import BaseNode
from langdmta_lab.base.base_state import State
from langdmta_lab.base.models import MODEL_REGISTRY
from langdmta_lab.config import LANGDMTA_GRAPH_DIR
from langdmta_lab.utils.io import load_yaml_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MemoryRetriever:
    def __init__(
        self, model_name: str, config_name: str = "MemoryRetriever.yaml", **kwargs: Any
    ):
        self.llm = MODEL_REGISTRY[model_name](**kwargs)
        self.prompt = self._load_prompt(config_name)

    def _load_prompt(self, config_filename: str) -> str:
        """Loads the prompt and members from YAML configuration."""
        config = load_yaml_data(
            str(Path(LANGDMTA_GRAPH_DIR) / "utils" / "configs" / config_filename)
        )
        prompt = config["prompt"]
        return prompt


class MemoryRetrieverNode(BaseNode):
    def __init__(
        self, name: str = "MemoryRetriever", model_name: str = "gpt-4o-openai", **kwargs: Any
    ) -> None:
        super().__init__(name)
        memory_retriever = MemoryRetriever(model_name=model_name, **kwargs)
        self.llm = memory_retriever.llm
        self.prompt = memory_retriever.prompt

    def _load_messages(self, state: State) -> List[Union[HumanMessage, Dict]]:
        query = state["messages"][-1]
        memory = state["memory"]
        prompt = {"role": "system", "content": self.prompt}
        return [prompt, memory, query]

    async def _invoke_response(self, messages: List[Union[HumanMessage, Dict]]) -> str:
        query = messages[-1]
        if isinstance(query, HumanMessage) and not query.content:
            return ""
        response = self.llm.invoke(messages).content
        return response

    def _handover_response(self, response: str) -> Dict[str, Union[str, List[Any]]]:
        response_formatted = f"Chat History Start:\n{response}\nChat History End"
        handover_dict = {"memory": [HumanMessage(content=response_formatted, name=self.name)]}
        return Command(update=handover_dict, goto="supervisor")
