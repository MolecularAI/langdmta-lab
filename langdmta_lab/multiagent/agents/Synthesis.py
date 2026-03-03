from typing import Any

from langgraph.graph.state import CompiledStateGraph

from langdmta_lab.base.base_node import BaseAgentNode
from langdmta_lab.subagents.Synthesis.Synthesis import Synthesis


class SynthesisNode(BaseAgentNode):
    def __init__(
        self,
        name: str = "Synthesis",
        model_name: str = "gpt-4o-openai",
        task_description: bool = False,
        **kwargs: Any,
    ):
        self.llm_kwargs = kwargs
        super().__init__(name, model_name, task_description)

    async def _init_agent(self) -> CompiledStateGraph:
        return await Synthesis(
            config_filename=f"{self.name}.yaml", model_name=self.model_name, **self.llm_kwargs
        ).create_agent()
