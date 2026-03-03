import importlib
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx
import requests
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from openai import BadRequestError, RateLimitError

from langdmta_lab.base.base_state import State
from langdmta_lab.base.helpers import (
    load_class_from_path,
    node_class_condition,
)
from langdmta_lab.utils.io import load_yaml_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GraphBuilder:
    def __init__(
        self, config_path: str, node_packages: list[str], filter_memory: bool = False
    ):
        """Initialize the builder with a YAML filename"""
        self.config_path = config_path
        self.node_packages = node_packages
        self._filter_memory = filter_memory
        self.node_map: Dict[str, Any] = {}

    def load_node_maps(self) -> None:
        """Load and merge node maps from base and supervisor packages."""
        node_maps = [
            load_class_from_path(pkg, node_class_condition, instantiate=False)
            for pkg in self.node_packages
        ]

        self.node_map = {k: v for m in node_maps for k, v in m.items()}

    def load_config(self) -> Dict[str, Any]:
        config = load_yaml_data(str(self.config_path))
        if self._filter_memory:
            self._add_memory_edges(config)
        return config

    def resolve_routing_function(self, routing_cfg: Dict[str, Any]) -> Callable:
        """
        Resolve a routing function from routing configuration.

        Expected formats:
        1. {"function": "route_from_supervisor"}   (local import)
        2. {"import": "pkg.module", "function": "fn"}
        """
        fn_name = routing_cfg["function"]

        if "import" not in routing_cfg:
            raise ValueError(
                "Routing configuration must include 'import' key specifying the module path."
            )

        module_path = routing_cfg["import"]

        module = importlib.import_module(module_path)

        try:
            return getattr(module, fn_name)
        except AttributeError as e:
            raise ImportError(
                f"Routing function '{fn_name}' not found in '{module_path}'"
            ) from e

    def build(self, max_retries_per_node: int, **kwargs: Any) -> CompiledStateGraph:
        """Build and compile the StateGraph from configuration."""
        self.load_node_maps()
        graph_config = self.load_config()

        builder = StateGraph(State)
        node_retry_policy = RetryPolicy(
            initial_interval=1.0,
            max_attempts=max_retries_per_node,
            jitter=False,
            retry_on=self.retry_on,
        )

        for node in graph_config["nodes"]:
            node_name = node["name"]
            node_cls = self.node_map[node_name]

            node_kwargs = node.get("kwargs", {})
            node_obj = node_cls(**node_kwargs, **kwargs)

            builder.add_node(node_name, node_obj, retry=node_retry_policy)

        for edge in graph_config["edges"]:
            src = edge["from"]

            if edge.get("conditional"):
                if "routing" not in edge:
                    raise ValueError("Conditional edge must include 'routing' configuration.")
                routing_fn = self.resolve_routing_function(edge["routing"])
                builder.add_conditional_edges(src, routing_fn)

            elif "to" in edge:
                dst = edge["to"]
                builder.add_edge(src, dst)

        return builder.compile()

    def retry_on(self, exc: Exception) -> bool:
        if isinstance(exc, ConnectionError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return 500 <= exc.response.status_code < 600
        if isinstance(exc, requests.HTTPError):
            return 500 <= exc.response.status_code < 600 if exc.response else True
        if isinstance(
            exc,
            (
                ValueError,
                TypeError,
                ArithmeticError,
                ImportError,
                LookupError,
                NameError,
                SyntaxError,
                RuntimeError,
                ReferenceError,
                StopIteration,
                StopAsyncIteration,
                OSError,
                BadRequestError,
            ),
        ):
            return False
        return True

    def _add_memory_edges(self, data: dict) -> None:
        nodes = data.setdefault("nodes", [])
        if not any(
            isinstance(node, dict) and node.get("name") == "MemoryRetriever" for node in nodes
        ):
            nodes.insert(0, {"name": "MemoryRetriever", "model_name": "gpt-4o-openai"})

        edges = data.setdefault("edges", [])
        for edge in edges:
            if isinstance(edge, dict) and edge.get("from") == "__start__":
                edge["to"] = "MemoryRetriever"
        edges.append({"from": "MemoryRetriever", "to": "supervisor"})
        data["edges"] = edges


class BaseGraphExecutor:
    """A base langgraph executor that wraps the graph invocation"""

    def __init__(
        self,
        node_packages: list[str],
        graph_config: str,
        memory_history: Optional[List[Union[HumanMessage, AIMessage]]] = None,
        callback_handlers: Optional[List[BaseCallbackHandler]] = None,
        filter_memory: bool = False,
        max_retries_per_node: int = 3,
        **kwargs: Any,
    ):
        self.graph = GraphBuilder(
            node_packages=node_packages, config_path=graph_config, filter_memory=filter_memory
        ).build(max_retries_per_node, **kwargs)

        if callback_handlers:
            for callback in callback_handlers:
                if not isinstance(callback, BaseCallbackHandler):
                    raise TypeError(f"Invalid callback handler: {callback}")
        self.callback_handlers = callback_handlers

        self.memory = ""
        if memory_history:
            self._configure_memory(memory_history)

    async def run(
        self, input_prompt: str, max_retries: int = 5
    ) -> Tuple[str, Dict[str, List[str]]]:
        config = {"callbacks": self.callback_handlers}
        input = {}
        input["messages"] = [("user", input_prompt)]
        input["memory"] = HumanMessage(content=self.memory)

        base_delay = 5  # Initial delay in seconds

        for retry in range(max_retries):
            try:
                output = await self.graph.ainvoke(
                    input=input,
                    config=config,
                )
                response = output["messages"][-1].content
                worker_output_files = output["worker_output_files"]
                return response, worker_output_files
            except RateLimitError as e:
                wait_time = base_delay * (2**retry) + random.uniform(0, 1)
                logging.error(e)
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            except BadRequestError as e:
                logging.error(e)
                logging.warning("Retrying query due to BadRequestError...")

        msg = f"Failed to process '{input_prompt}' after {max_retries} retries."
        logging.error(msg)
        return msg, {}

    def _configure_memory(self, memory_history: List[Union[HumanMessage, AIMessage]]) -> None:
        if memory_history:
            self.memory = get_buffer_string(memory_history)
            self.memory = f"Chat History Start:\n{self.memory}\nChat History End"
        else:
            self.memory = ""
