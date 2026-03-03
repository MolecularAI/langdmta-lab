import inspect
import json
import logging
import os
import subprocess
from functools import wraps
from typing import Any, Callable, Dict, List

import deepdiff
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from langdmta_lab.config import LANGDMTA_MCPS_DIR, TEST_ASSETS_DIR
from langdmta_lab.utils.constants import MCP_SERVERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def mock_if_available(key: str, mocked_data: Dict[str, Any]) -> Callable:
    """
    Decorator that checks for mocked data and returns it if available.

    Args:
        key: The key to look up in the mocked_data dictionary
        mocked_data: Dictionary containing mock data loaded from mock_data.json

    Returns:
        Decorator function that wraps the original function
    """

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if mocked_data:
                # Get function arguments as a dictionary for comparison
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                func_kwargs = dict(bound_args.arguments)

                mocked_output = find_mock_output(mocked_data.get(key, []), func_kwargs)
                if mocked_output:
                    logging.info(f"Using mocked output for {key} with inputs={func_kwargs}")
                    return mocked_output
            return func(*args, **kwargs)

        return wrapper

    return decorator


def find_mock_output(
    mocked_data: List[Dict[str, Any]], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    for item in mocked_data:
        if "smiles_file" in item["Input"] and item["Input"]["smiles_file"] != "":
            item["Input"]["smiles_file"] = os.path.join(
                TEST_ASSETS_DIR, item["Input"]["smiles_file"]
            )
        if not deepdiff.DeepDiff(
            item["Input"],
            kwargs,
            ignore_string_case=True,
            ignore_order=True,
            ignore_numeric_type_changes=True,
        ):
            if "output_path" in item["Output"]:
                item["Output"]["output_path"] = os.path.join(
                    TEST_ASSETS_DIR, item["Output"]["output_path"]
                )
            return item["Output"]
    logging.info(f"No mock data found for input {kwargs}")
    return {}


def load_mock_data(file_path: str) -> Dict[str, Any]:
    load_dotenv()
    mocked_data = {}
    if os.environ.get("USE_MOCK", "false").lower() == "true":
        mock_data_file = os.path.join(file_path, "mock_data.json")
        with open(mock_data_file, "r") as fileobj:
            mocked_data = json.load(fileobj)
    return mocked_data


def start_mcps(mcp_servers: List[str] = MCP_SERVERS) -> List[Any]:
    mcp_processes = []
    logs_dir = os.path.join(LANGDMTA_MCPS_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for server_name in mcp_servers:
        server_dir = os.path.join(LANGDMTA_MCPS_DIR, server_name)
        server_script = os.path.join(server_dir, "server.py")
        log_file = os.path.join(logs_dir, f"{server_name}.log")
        if os.path.exists(server_script):
            log_handle = open(log_file, "w")
            try:
                process = subprocess.Popen(
                    ["python", server_script],
                    cwd=server_dir,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                mcp_processes.append((process, log_handle, server_name))
                logging.info(f"Started {server_name} MCP server")
            except Exception as e:
                logging.error(f"Failed to start {server_name} MCP server: {e}")
                log_handle.close()
        else:
            logging.warning(f"Server script not found: {server_script}")

    if not mcp_processes:
        logging.error("No MCP servers were started.")

    return mcp_processes


def stop_mcps(mcp_processes: List[Any]) -> None:
    for process, log_handle, server_name in mcp_processes:
        try:
            process.terminate()
            process.wait(timeout=5)
            logging.info(f"Stopped {server_name} MCP server")
        except subprocess.TimeoutExpired:
            process.kill()
            logging.warning(f"Forcefully killed {server_name} MCP server")
        except Exception as e:
            logging.error(f"Error stopping {server_name} MCP server: {e}")
        finally:
            log_handle.close()


def create_mcp_server_instance(server_name: str, port_env_var: str) -> Any:
    load_dotenv()
    env_var = os.environ.get(port_env_var)
    if not env_var:
        raise ValueError(f"Environment variable '{port_env_var}' is not set.")
    mcp = FastMCP(server_name, port=int(env_var))
    return mcp
