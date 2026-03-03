from abc import ABC
from pathlib import Path
from types import new_class
from typing import Any, Dict, List, Literal, Tuple, Union

from typing_extensions import TypedDict

from langdmta_lab.base.models import MODEL_REGISTRY
from langdmta_lab.utils.io import load_yaml_data


class BaseSupervisor(ABC):
    SUPERVISOR_NODE_NAME = "supervisor"

    def __init__(
        self,
        end_token: str,
        model_name: str,
        graph_yaml_path: Union[str, Path],
        prompt_fp: Union[str, Path],
        member_dir: Union[str, Path],
        **kwargs: Any,
    ):
        self.end_token = end_token
        self.graph_yaml_path = graph_yaml_path
        self.member_dir = member_dir

        self.prompt = self._load_prompt(prompt_fp)

        llm = MODEL_REGISTRY[model_name]
        self.llm = llm(**kwargs)

    # Extension hooks
    # ------------------------------------------------------------------
    @property
    def internal_prompt_key(self) -> str:
        """
        YAML key used to load the member → supervisor description.
        """
        return "prompt"

    @property
    def member_prompt_key(self) -> str:
        """
        YAML key used to load the member → supervisor description.
        """
        return "prompt_to_supervisor"

    @property
    def description_injections(self) -> Dict[str, str]:
        """
        Mapping of placeholders to injected text used when rendering
        member descriptions.

        Example:
            {
                "{{supported_properties}}": "title, author, date",
                "{{capabilities}}": "search, summarize, reason"
            }

        Notes:
        - Placeholders are simple string replacements.
        - Return an empty dict to disable injection (default).
        - Subclasses should explicitly opt in to injection behavior.
        """
        return {}

    @property
    def start_node_name(self) -> str:
        """
        Placeholder for start node name
        """
        return "START"

    # Prompt loading
    # ------------------------------------------------------------------

    def _load_prompt(self, prompt_fp: Union[str, Path]) -> str:
        self.members, self.descriptions = self.get_member_and_descriptions()

        worker_description = " ".join(self.descriptions)

        # inject description values such as supported properties etc.
        for placeholder, value in self.description_injections.items():
            worker_description = worker_description.replace(placeholder, value)

        prompt = load_yaml_data(prompt_fp)[self.internal_prompt_key]
        return prompt.format(WORKER_DESCRIPTION=worker_description)

    def get_member_and_descriptions(self) -> Tuple[List[str], List[str]]:
        """
        Identify supervisor members by selecting nodes that have
        an outgoing edge to the supervisor in graph.yaml, then load
        their descriptions from <Member>.yaml.
        """
        graph = load_yaml_data(str(self.graph_yaml_path))

        members: List[str] = []
        descriptions: List[str] = []

        supervisor_name = self.SUPERVISOR_NODE_NAME

        # 1. Find nodes that route TO the supervisor
        member_names = {
            edge["from"]
            for edge in graph.get("edges", [])
            if edge.get("to") == supervisor_name and edge.get("from") != self.start_node_name
        }

        # 2. Load descriptions only for those members
        for name in sorted(member_names):
            members.append(name)
            descriptions.append(self._load_member_description(name))

        return members, descriptions

    def _load_member_description(self, member_name: str) -> str:
        """
        Load member description using a configurable YAML key.
        """
        config_path = Path(self.member_dir) / f"{member_name}.yaml"
        data = load_yaml_data(str(config_path))

        description = data[self.member_prompt_key]
        return f"\n### {member_name}:\n{description}"

    # Routing
    # ------------------------------------------------------------------

    @property
    def options(self) -> List[str]:
        return self.members + [self.end_token]

    @property
    def return_options(self) -> List[str]:
        return self.members + ["__end__"]

    def create_router(self) -> object:
        """
        Create a minimal router TypedDict.

        This default implementation documents the minimal contract:
        the supervisor must decide which worker to route to next.
        """
        annotations = {
            "next": Literal[tuple(self.options)],
        }

        Router = new_class(
            "BaseRouter",
            (TypedDict,),
            exec_body=lambda ns: ns.update(
                {
                    "__annotations__": annotations,
                    "__doc__": (
                        "Minimal router output.\n\n"
                        "Fields:\n"
                        "- next: worker name to route to next"
                    ),
                }
            ),
        )

        return Router
