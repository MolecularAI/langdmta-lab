import importlib
import inspect
import logging
import os
import pkgutil
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, create_model

from langdmta_lab.base.base_node import BaseAgentNode, BaseNode
from langdmta_lab.utils.constants import parse_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _module_list_from_package_path(package_path: str, members: List[str] = []) -> Dict:
    module_dict = {}
    package = importlib.import_module(package_path)

    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, prefix=package.__name__ + "."
    ):
        obj_name = module_name.split(".")[-1]
        if not members or obj_name in members:
            module_dict[obj_name] = importlib.import_module(module_name)
    return module_dict


def node_class_condition(cls: object) -> bool:
    return (
        inspect.isclass(cls)
        and issubclass(cls, BaseNode)
        and cls not in (BaseNode, BaseAgentNode)
    )


def _basetool_class_list_from_modules(
    module_dict: Dict, instantiate: bool, class_condition: Callable[[type], bool]
) -> Dict:
    class_dict = {}

    for module_name, module in module_dict.items():
        for name, obj in inspect.getmembers(module):
            # Use the provided condition to filter the class
            if "Base" in name:
                continue
            if class_condition(obj):
                try:
                    if instantiate:
                        class_dict[module_name] = obj()
                    else:
                        class_dict[module_name] = obj
                except TypeError as e:
                    raise ValueError(f"Could not import tool. {e}")
    return class_dict


def load_class_from_path(
    package_path: str,
    class_condition: Callable[[type], bool],
    instantiate: bool = True,
    members: List[str] = [],
) -> Dict:
    module_dict = _module_list_from_package_path(package_path, members)
    class_dict = _basetool_class_list_from_modules(
        module_dict, instantiate, class_condition=class_condition
    )
    return class_dict


def yaml2schema(config: Dict[str, Any]) -> Tuple[BaseModel, str]:
    tool_description = config["description"]
    tool_name = config["name"]
    field_defs = config["fields"]
    fields = {}
    for name, props in field_defs.items():
        field_type = parse_type(props["type"])
        field_desc = props["description"]
        required = props.get("required", False)
        default = props.get("default", ...)
        if not required:
            field_type = Optional[field_type]
        fields[name] = (field_type, Field(default=default, description=field_desc))

    tool_schema = create_model(tool_name, **fields)
    return tool_schema, tool_description


def configure_url(txt: str) -> str:
        """Configures the MCP URL in the mcp_config."""
        environ_var = re.findall(r"\$\{.+?\}", txt)
        for item in environ_var:
            if item[2:-1] not in os.environ:
                raise ValueError(f"'{item[2:-1]}' not in environment variables")
            txt = txt.replace(item, os.environ[item[2:-1]])
        return txt