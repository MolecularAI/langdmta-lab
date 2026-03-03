import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def create_temp_dir() -> Path:
    path = tempfile.mkdtemp()
    path_obj = Path(path)
    path_obj.mkdir(exist_ok=True, parents=True)
    return path_obj


def load_yaml_data(filepath: Union[Path, str]) -> Dict[str, Any]:
    """Loads data from a YAML file.

    Args:
        filepath: Path to the YAML file.

    Returns:
        A dictionary of the YAML contents
    """
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    return data
