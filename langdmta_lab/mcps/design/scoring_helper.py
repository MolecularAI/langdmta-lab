import logging
import tomllib
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from langdmta_lab.utils.io import create_temp_dir
from langdmta_lab.utils.validators import validate_filename


def prepare_smiles_input(smiles: List[str], smiles_file: str, smiles_col: str) -> str:
    """
    Prepares the input SMILES file for scoring.

    If both the ref_smiles and smiles_file are provided, they will be concatenated.
    If a CSV file is provided as input, the smiles_col should be present
    in the CSV file.
    """
    if smiles_file.endswith(".csv"):
        data = pd.read_csv(smiles_file)
        if smiles_col not in data.columns:
            raise ValueError(
                f"Column '{smiles_col}' not found in the CSV file.",
            )

        input_data = pd.DataFrame({smiles_col: smiles})
        updated_data = pd.concat([data, input_data], ignore_index=True).drop_duplicates(
            subset=[smiles_col]
        )
        input_path = f"{create_temp_dir()}/{Path(smiles_file).name}"
        updated_data.to_csv(input_path, index=False, header=True)
    else:
        input_path = smiles_file or str(create_temp_dir() / "input.smi")
        with open(input_path, "a") as input_file:
            for smi in smiles:
                input_file.write(f"{smi}\n")

    return validate_filename(input_path)


def filter_scoring_components(
    template: Dict[str, Any],
    properties: Dict[str, Any],
) -> Dict[str, Any]:  # type: ignore
    """Filter scoring components based on the provided properties."""

    filtered_components = []
    for component in template["scoring"]["component"]:
        property_key = list(component.keys())[0]
        if "".join(property_key.lower().split()) not in [
            "".join(scorer.lower().split()) for scorer in properties
        ]:
            continue
        if property_key in SPECIAL_SCORERS and (properties[property_key]):
            components = SPECIAL_SCORERS[property_key](
                template=template,
                **properties[property_key],
            )
            filtered_components.extend(components)
        elif property_key in SPECIAL_SCORERS and (not properties[property_key]):
            raise ValueError(f"additional input required for {property_key} but not provided")
        elif property_key in properties:
            filtered_components.append(component)
    template["scoring"]["component"] = filtered_components
    return template


def load_scoring_template(smiles_file: str, smiles_col: str) -> Dict[str, Any]:
    """Load scoring template JSON and update with input file details"""
    with open(Path(__file__).parent / "scoring_components_template.toml", "rb") as fileobj:
        template = tomllib.load(fileobj)

    template["parameters"]["smiles_file"] = smiles_file
    template["parameters"]["smiles_column"] = smiles_col
    return template


def scoring_custom_alerts(template: Dict[str, Any], **kwargs: Any) -> List[Dict[str, Any]]:
    """Configure Custom alert scoring within the given template."""
    KEY = "custom_alerts"
    smarts = kwargs.get("smiles/smarts")
    if not smarts:
        logging.warning(
            f"{KEY} component requires input in the form of substructures, "
            "provided either by name or as SMARTS patterns. "
            "Since no input was provided, the default substructures will be used."
        )
        return []

    for component in template["scoring"]["component"]:
        property_key = list(component.keys())[0]
        if property_key == KEY:
            component[KEY]["endpoint"][0]["params"]["smarts"] = smarts
            return [component]
    return []


def scoring_matching_substructure(
    template: Dict[str, Any], **kwargs: Any
) -> List[Dict[str, Any]]:
    """Configure the Matching Substructure scoring within the given template."""
    smarts_list = kwargs.get("smiles/smarts")
    if not smarts_list:
        raise ValueError(
            "Matching Substructure component requires input in the form of substructures, "
            "provided either by name or as SMARTS patterns.",
        )

    components = []
    for smarts in smarts_list:
        components.append(
            {
                "MatchingSubstructure": {
                    "endpoint": [
                        {
                            "name": f"Matching substructure {smarts}",
                            "weight": 1,
                            "params": {"smarts": smarts, "use_chirality": False},
                        }
                    ]
                }
            }
        )
    return components


def scoring_group_count(template: Dict[str, Any], **kwargs: Any) -> List[Dict[str, Any]]:
    """Configure the Group Count scoring within the given template."""
    smarts_list = kwargs.get("smiles/smarts")
    if not smarts_list:
        raise ValueError(
            "Group Count component requires input in the form of substructures, "
            "provided either by name or as SMARTS patterns.",
        )

    components = []
    for smarts in smarts_list:
        components.append(
            {
                "GroupCount": {
                    "endpoint": [
                        {
                            "name": f"Group count {smarts}",
                            "weight": 1,
                            "params": {"smarts": f'[{",".join(smarts)}]'},
                        }
                    ]
                }
            }
        )
    return components


SPECIAL_SCORERS = {
    "custom_alerts": scoring_custom_alerts,
    "MatchingSubstructure": scoring_matching_substructure,
    "GroupCount": scoring_group_count,
}
