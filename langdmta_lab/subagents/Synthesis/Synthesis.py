from pathlib import Path
from typing import Any

from langdmta_lab.base.base_agent import BaseAgent


class Synthesis(BaseAgent):
    def __init__(self, config_filename: str, model_name: str, **kwargs: Any):
        self.agent_dir = Path(__file__).resolve().parent
        config_fp = str(self.agent_dir / config_filename)
        super().__init__(config_filename=config_fp, model_name=model_name, **kwargs)
        self.prompt, self.members = self._load_prompt(config_fp)
