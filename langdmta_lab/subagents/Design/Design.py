import json
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from langdmta_lab.base.base_agent import BaseAgent


class Design(BaseAgent):

    def __init__(self, config_filename: str, model_name: str, **kwargs: Any):
        self.agent_dir = Path(__file__).resolve().parent
        config_fp = str(self.agent_dir / config_filename)
        super().__init__(config_filename=config_fp, model_name=model_name, **kwargs)
        self.prompt, self.members = self._load_prompt(config_fp)

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
            (await client.get_resources("mcp_server", uris=["scoring://descriptions_design"]))[
                0
            ].as_string()
        )

        prompt = self.prompt
        property_info = ""
        for property, description in descriptions_design.items():
            if property in supported_components:
                property_info += f"\n**{property}**: {description}\n"
        prompt = prompt.replace("{{special_properties_info}}", property_info)
        prompt = prompt.replace("{{supported_properties}}", ", ".join(supported_components))
        return prompt
