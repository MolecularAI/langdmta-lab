from typing import Any, Dict, List, Union

INPUT_SMILES = "Input_SMILES"
GENERATED_SMILES = "SMILES"
DEFAULT_SMILES = "SMILES"
SIMILARITY_METRIC = "Tanimoto"
DEFAULT_BOUND = 99999
PROPERTY_ATTRIBUTE_KEY = "attribute"
CANONICAL_SMILES = "RDKit_SMILES (REINVENT)"
COMPOUND_COLUMN = "Compound Name"


def parse_type(type_str: str) -> Any:
    base_types = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "List[str]": List[str],
        "List[int]": List[int],
        "Dict[str, str]": Dict[str, str],
        "Dict[str, int]": Dict[str, int],
        "Dict[str, Dict[str, str]]": Dict[str, Dict[str, str]],
        "Dict": Dict,
        "Dict[str, Dict[str, float]]": Dict[str, Dict[str, float]],
        "List[List[str]]": List[List[str]],
        "Union[str,List[Dict[str, Any]]]": Union[str, List[Dict[str, Any]]],
        "List[Dict[str, Any]]": List[Dict[str, Any]],
    }
    if type_str not in base_types:
        raise ValueError(f"Unsupported type: {type_str}")
    return base_types[type_str]


TOOL_NAMES = [
    "smilesvalidator",
    "synonyms2smiles",
    "molformer",
    "reinvent_scoring",
    "Analyzer",
    "aizynthfinder",
    "precedent_finder",
    "reinforcement_learning",
]

MCP_SERVERS = ["utility", "design", "synthesis"]
