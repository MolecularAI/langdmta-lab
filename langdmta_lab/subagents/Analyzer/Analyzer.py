import json
from pathlib import Path
from typing import Any, Tuple

from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.state import CompiledStateGraph

from langdmta_lab.base.helpers import configure_url
from langdmta_lab.base.models import MODEL_REGISTRY
from langdmta_lab.utils.io import load_yaml_data


class Analyzer:
    def __init__(self, config_filename: str, model_name: str, **kwargs: Any):
        self.agent_dir = Path(__file__).resolve().parent
        self.prompt, self.mcp_config = self._load_prompt(config_filename)
        llm = MODEL_REGISTRY[model_name]
        self.llm = llm(**kwargs)

    def _load_prompt(self, config_filename: str) -> Tuple[str, dict[str, Any]]:
        """
        Customise _load_prompt for Analyzer:
        Dynamically loading the supported reinvent scoring propertes
        """
        config = load_yaml_data(
            str(self.agent_dir / config_filename)
        )
        prompt = config["prompt"]
        mcp_config = config["mcp_config"]
        mcp_config["url"] = configure_url(mcp_config["url"])
        return prompt, mcp_config

    async def _modify_prompt(self) -> str:
        client = MultiServerMCPClient({"mcp_server": self.mcp_config})
        supported_components = json.loads(
            (
                await client.get_resources(
                    "mcp_server", uris=["scoring://supported_components"]
                )
            )[0].as_string()
        )

        descriptions_design = json.loads(
            (
                await client.get_resources(
                    "mcp_server", uris=["scoring://descriptions_analyzer"]
                )
            )[0].as_string()
        )

        prompt = self.prompt
        property_info = ""
        for property, description in descriptions_design.items():
            if property in supported_components:
                property_info += f"\n**{property}**: {description}\n"
        prompt = prompt.replace("{{supported_properties_info}}", property_info)
        prompt = prompt.replace("{{supported_properties}}", ", ".join(supported_components))
        return prompt

    async def create_agent(
        self,
    ) -> CompiledStateGraph:
        prompt = await self._modify_prompt()
        return create_python_agent(
            llm=self.llm,
            tool=PythonREPLTool(),
            verbose=True,
            show_intermediate_steps=True,
            prefix=prompt,
            agent_executor_kwargs=dict(handle_parsing_errors=True),
        )
