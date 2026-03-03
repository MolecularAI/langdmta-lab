import logging
import tomllib
from pathlib import Path
import subprocess
from typing import Any, Dict, List

from langdmta_lab.utils.io import create_temp_dir
from langdmta_lab.utils.validators import validate_filename
from langdmta_lab.utils.transform import double_sigmoid, reverse_sigmoid, sigmoid

DEFAULT_LOW = -99999
DEFAULT_HIGH = 99999
K_SMOOTH = 0.25  # smoothness
P_AT_THRESHOLD = 0.8  # reward at threshold
SPAN_RATIO = 0.2  # +20% or -20% around the center

def load_rl_template() -> Dict[str, Any]:
    """Load reinforcement learning template JSON"""
    with open(Path(__file__).parent / "reinforcement_learning_template.toml", "rb") as fileobj:
        template = tomllib.load(fileobj)
    return template


def prepare_smiles_input(smiles: List[str]) -> str:
    input_path = str(create_temp_dir() / "input.smi")
    with open(input_path, "a") as input_file:
        for smi in smiles:
            input_file.write(f"{smi}\n")
    return validate_filename(input_path)


def update_template_with_model(
    template: Dict[str, Any], model: str, input_smiles: List[str], ENV_PATH: str
) -> Dict[str, Any]:
    smiles_file = prepare_smiles_input(input_smiles)

    if model == "molformer":
        prior = f"{ENV_PATH}/REINVENT4/priors/mol2mol_scaffold_generic.prior"
        template["parameters"]["smiles_file"] = smiles_file
    elif model == "reinvent":
        template["inception"] = {"smiles_file": smiles_file}
        prior = f"{ENV_PATH}/REINVENT4/priors/reinvent.prior"
    else:
        raise ValueError(f"Model '{model}' not supported")

    template["parameters"]["prior_file"] = prior
    template["parameters"]["agent_file"] = prior
    return template

def set_device(template: Dict[str, Any], env_path: str) -> Dict[str, Any]:
    """Set the device depending on GPU availability."""

    arglist = [
        "conda",
        "run",
        "-p",
        env_path,
        "python",
        "-c", 
        "import torch; print(torch.cuda.is_available())",
    ]
    logging.info(" ".join(arglist))
    gpu_available = subprocess.check_output(arglist).decode("utf-8")
    logging.info(f"GPU available: {gpu_available}")
    
    if gpu_available.strip() == "True":
        template["device"] = "cuda:0"
    elif gpu_available.strip() == "False":
        template["device"] = "cpu"
    else:
        logging.error("Unexpected output from GPU availability check.")
    return template

def filter_rl_scoring_components(
    template: Dict[str, Any],
    properties: Dict[str, Any],
) -> Dict[str, Any]:  # type: ignore
    """Filter scoring components based on the provided properties."""

    filtered_components = []
    for component in template["stage"][0]["scoring"]["component"]:
        property_key = list(component.keys())[0]
        if "".join(property_key.lower().split()) not in [
            "".join(scorer.lower().split()) for scorer in properties
        ]:
            continue
        if property_key in properties:
            updated_component = update_component_with_criteria(component, property_key, properties)
            filtered_components.append(updated_component)
    template["stage"][0]["scoring"]["component"] = filtered_components
    return template

def update_component_with_criteria(
    component: Dict[str, Any],
    scoring_property: str,
    criteria: Dict[str, Dict],
) -> Dict[str, Any]:

    low = criteria[scoring_property].get("low", None)
    high = criteria[scoring_property].get("high", None)
    transform = component[scoring_property]["endpoint"][0]["transform"]

    # if not low or high value is provided, get from existing transform parameters
    if not (low and high):
        if ("low" and "high") in transform:
            low = float(transform["low"])
            high = float(transform["high"])

    if low != DEFAULT_LOW and high != DEFAULT_HIGH:
        transform = _double_sigmoid_transform(low, high, transform)
    elif low != DEFAULT_LOW and high == DEFAULT_HIGH:
        transform = _sigmoid_transform(low, transform)
    elif low == DEFAULT_LOW and high != DEFAULT_HIGH:
        transform = _reverse_sigmoid_transform(high, transform)
    else:
        logging.error(
            f"Both low and high values for '{scoring_property}' are default. "
            "Please specify at least one of them.",
        )
    component[scoring_property]["endpoint"][0]["transform"] = transform

    return component


def _double_sigmoid_transform(
    low: float, high: float, transform: Dict[str, Any]
) -> Dict[str, Any]:
    k = transform.get("coef_div", 500)
    center_low, center_high = double_sigmoid(low, high, P_AT_THRESHOLD, K_SMOOTH)
    updated_transform = {
            "type": "double_sigmoid",
            "low": center_low * (1.0 - SPAN_RATIO),
            "high": center_high * (1.0 + SPAN_RATIO),
            "coef_div": k,
            "coef_si": transform.get("coef_si", 20) / k,
            "coef_se": transform.get("coef_se", 20) / k,
    }
    return updated_transform

def _sigmoid_transform(low: float, transform: Dict[str, Any]) -> Dict[str, Any]:
    center = sigmoid(low, P_AT_THRESHOLD, K_SMOOTH)
    updated_transform = {
            "type": "sigmoid",
            "low": center * (1.0 - SPAN_RATIO),
            "high": center * (1.0 + SPAN_RATIO),
            "k": K_SMOOTH,
    }
    return updated_transform

def _reverse_sigmoid_transform(high: float, transform: Dict[str, Any]) -> Dict[str, Any]:
    center = reverse_sigmoid(high, P_AT_THRESHOLD, K_SMOOTH)
    updated_transform = {
            "type": "reverse_sigmoid",
            "low": center * (1.0 - SPAN_RATIO),
            "high": center * (1.0 + SPAN_RATIO),
            "k": K_SMOOTH,
    }
    return updated_transform
