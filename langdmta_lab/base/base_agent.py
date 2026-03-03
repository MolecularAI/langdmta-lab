from typing import Any, Callable, Dict, List, Tuple

from langchain_core.runnables import Runnable
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from langdmta_lab.base.helpers import configure_url
from langdmta_lab.base.models import MODEL_REGISTRY
from langdmta_lab.utils.io import load_yaml_data


class BaseAgent:
    def __init__(self, config_filename: str, model_name: str = "gpt-4o-openai", **kwargs: Any):
        self.prompt, self.mcp_config = self._load_prompt(config_filename)
        self.tools = None
        llm = MODEL_REGISTRY[model_name]
        self.llm = llm(**kwargs)

    def _load_prompt(self, config_filename: str) -> Tuple[str, Dict[str, str]]:
        """Loads the prompt and members from YAML configuration."""
        config = load_yaml_data(config_filename)
        prompt = config["prompt"]
        mcp_config = config["mcp_config"]
        mcp_config["url"] = configure_url(mcp_config["url"])
        return prompt, mcp_config

    async def _load_tools(self) -> List[Callable]:
        client = MultiServerMCPClient({"mcp_server": self.mcp_config})

        tools = await client.get_tools()
        return tools

    async def _modify_prompt(self) -> str:
        return self.prompt

    async def create_agent(self) -> Runnable:
        self.tools = await self._load_tools()  # type: ignore
        prompt = await self._modify_prompt()
        agent = create_react_agent(
            self.llm.bind_tools(tools=self.tools, parallel_tool_calls=False),
            tools=self.tools,
            prompt=prompt,
        )
        return agent
