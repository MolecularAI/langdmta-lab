from typing import Any

from langgraph.graph.state import CompiledStateGraph

from langdmta_lab.base.base_node import BaseAgentNode
from langdmta_lab.subagents.Utility.Utility import Utility


class UtilityNode(BaseAgentNode):
    def __init__(
        self, name: str = "Utility", model_name: str = "gpt-4o-openai", **kwargs: Any
    ) -> None:
        self.llm_kwargs = kwargs
        super().__init__(name, model_name)

    async def _init_agent(self) -> CompiledStateGraph:
        return await Utility(
            config_filename=f"{self.name}.yaml", model_name=self.model_name, **self.llm_kwargs
        ).create_agent()
