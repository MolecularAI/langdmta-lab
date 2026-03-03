"""
Design MCP Server
"""

import json
import logging
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from rl_helper import (
    filter_rl_scoring_components,
    load_rl_template,
    set_device,
    update_template_with_model,
)
from scoring_helper import (
    filter_scoring_components,
    load_scoring_template,
    prepare_smiles_input,
)

from langdmta_lab.config import ENV_DIR
from langdmta_lab.mcps.helpers import (
    create_mcp_server_instance,
    load_mock_data,
    mock_if_available,
)
from langdmta_lab.utils.io import create_temp_dir
from langdmta_lab.utils.validators import validate_smiles

# Create the FastMCP server instance
mcp = create_mcp_server_instance("design-server", "DESIGN_MCP_PORT")

mocked_data = load_mock_data(os.path.dirname(__file__))

ENV_PATH = os.path.join(ENV_DIR, "design")
BASE_PATH = Path(__file__).parent.resolve()

SCORING_DESCRIPTIONS = {}
with open(os.path.join(BASE_PATH, "scoring_descriptions.toml"), "rb") as fileobj:
    SCORING_DESCRIPTIONS = tomllib.load(fileobj)


SUPPORT_SCORING_COMPONENTS = []
with open(os.path.join(BASE_PATH, "scoring_components_template.toml"), "rb") as fileobj:
    scoring_components = tomllib.load(fileobj)
    SUPPORT_SCORING_COMPONENTS = [
        list(item.keys())[0] for item in scoring_components["scoring"]["component"]
    ]


@mcp.tool()
@mock_if_available("molformer", mocked_data)
def molformer(smiles_list: List[str], number_of_smiles: int = 50) -> Dict[str, str]:
    """
    Generate new molecules similar to given input molecules. This tool does not enforce any substructure matching constraints, the substructure matching should be performed by a different tool.
    The full set of generated molecules are stored at a CSV file path specified by 'output_path' in the tool output. This CSV file also contains a column that indicates the tanimoto similarity of the generated molecules to the reference compound.
    The Final Output provided to the user MUST contain the CSV file path. In a separate snetence, it must also be mentioned to the user that an output CSV file of the complete results is generated and can be downloaded below. The complete result will contain a total number of generated smiles lesser than the 'number_of_smiles' because duplicate and invalid smiles are filtered out by the tool. You must ALWAYS communicate this information to the user.


    Args:
        smiles_list: list of SMILES strings similar to this example: ['CCC(C)=O', 'CC(=O)OC(C)C']
        number_of_smiles: "an integer that specifies how many similar molecules to generate for each input molecule."
    """  # noqa: E501

    logging.info(f"Molformer called with inputs={locals()}")

    path_obj = create_temp_dir()

    smiles_file = path_obj / "smiles.txt"
    with open(smiles_file, "w") as fileobj:
        fileobj.write("\n".join(smiles_list))

    output_path = path_obj / "molformer-sampling.csv"
    toml_config = f"""
    run_type = "sampling"
    device = "cpu"

    [parameters]
    model_file = "{ENV_PATH}/REINVENT4/priors/mol2mol_medium_similarity.prior"
    smiles_file = "{smiles_file}"
    sample_strategy = "beamsearch"
    output_file = "{output_path}"
    num_smiles = {number_of_smiles}
    unique_molecules = true
    """
    config_file = path_obj / "config.toml"
    with open(config_file, "w") as fileobj:
        fileobj.write(toml_config)

    arglist = [
        "conda",
        "run",
        "--live-stream",
        "-p",
        f"{os.path.join(ENV_PATH, 'env')}",
        "reinvent",
        str(config_file),
    ]
    logging.info(" ".join(arglist))
    _ = subprocess.check_output(arglist).decode("utf-8")

    return {"output_path": str(output_path)}


@mcp.tool()
@mock_if_available("reinvent_scoring", mocked_data)
def reinvent_scoring(
    scoring_properties: Dict[str, Any],
    smiles_file: str = "",
    ref_smiles: List[str] = [],
    smiles_col: str = "SMILES",
) -> Dict[str, Any]:
    """
    Use this tool when the task involves computing the chemical properties of some molecules. Do NOT use this tool if the query just requests similar molecules. This tool **DOES NOT** perform any filtering.

    Instructions for providing input parameters:
    - If the complete set of generated molecules is available as a csv file, you MUST ALWAYS provide it in the 'smiles_file' parameter.
    - When the user requests some directional property adjustments (higher, lower etc) based on some reference compounds, those reference compounds must also be provided in the 'ref_smiles' parameter. (both smiles_file and ref_smiles are required)
    - When the property adjustments are not directional (absence/presence of substructures), then leave "ref_smiles" empty and only provide the "smiles_file".
    - If the user only wants to compute properties for some reference compounds without any generated molecules, then only 'ref_smiles' is required.

    Instructions for providing the **required positional argument:`scoring_properties`**:
    - `scoring_properties`: is a dictionary where each key is a scoring property name mentioned by the user and each value is a dictionary which is empty by default. But special dictionary values are required for the following scoring properties:
        - 'MatchingSubstructure': key: 'smiles/smarts', value: List(str) must be a list of validated SMILES/SMARTS. When multiple substrucures are provided, include them **ALL** at once as a list.
        - 'GroupCount': key: 'smiles/smarts', value: must be a nested list of lists of validated SMILES or SMARTS patterns.

    - The keys must be selected from the list of supported properties. It is essential that `scoring_properties` is always provided, and you must include **only** the properties explicitly mentioned in the user query.

    If the 'GroupCount' scoring property is to be used, you must correctly format the input list. If the user asks for counting multiple features **combined**, use a single list (Example, “How many F **or** Cl are there?” → `[['F', 'Cl']]`). If the user wants to count each substructure **individually**, use separate lists (Example, “How many F and Cl individually?” → `[['F'], ['Cl']]`). Keywords like “**or**” imply grouping together → one inner list. Keywords like “**each**”, “**individually**”, “**separately**”, “**and**” (when distinguishing features) → separate lists.

    The Final Output provided to the user MUST contain the full CSV file path, and the scored molecules' results (if obtained). When explaining the results, it must also be mentioned to the user that an output CSV file of the complete results is generated and can be downloaded below. You must provide the FULL path that starts with "/" and alwayes ends with ".csv".

    Args:
        scoring_properties:  Dictionary of default scoring properties that the user wants to compute.
        smiles_file: Full file path (ends with .csv or .smi) to the file containing the complete generated molecules. A URL should not be passed as input here. The file contains the complete SMILES data, so do not ask the user about information regarding the column name unless the user explicitly states it.
        ref_smiles: (optional) List of reference compounds (validated SMILES) mentioned in the user input. These are either molecules to be scored or the baseline molecules against which property adjustments are computed, **NOT** the substructure user choose to include or exclude. Do not pass native ligand SMILES.
        smiles_col: (optional) Required ONLY if the user explicitly specifies the column name containing the molecules to be scored. The tool uses a default value if the column name is not provided.
    """  # noqa: E501

    logging.info(f"Reinvent scoring called with inputs={locals()}")

    smiles_list = validate_smiles(ref_smiles) if ref_smiles else []
    input_path = prepare_smiles_input(smiles_list, smiles_file, smiles_col)
    template = load_scoring_template(input_path, smiles_col)
    template = filter_scoring_components(template, scoring_properties)

    path_obj = create_temp_dir()
    output_path = path_obj / "reinvent_scoring_output.csv"
    template["parameters"]["output_csv"] = str(output_path)

    config_file = path_obj / "scoring_config.json"
    with open(config_file, "w") as fileobj:
        json.dump(template, fileobj)

    arglist = [
        "conda",
        "run",
        "--live-stream",
        "-p",
        f"{os.path.join(ENV_PATH, 'env')}",
        "reinvent",
        str(config_file),
    ]
    logging.info(" ".join(arglist))
    _ = subprocess.check_output(arglist).decode("utf-8")

    result = {"output_path": str(output_path)}
    if smiles_list:
        output = pd.read_csv(output_path)
        result["results"] = {}  # type: ignore
        matched_rows = []
        for smiles in smiles_list:
            matched_rows = output[output[smiles_col].astype(str) == str(smiles)]

            raw_columns = [col for col in output.columns if "raw" in col.lower()]
            smiles_results = {
                col.split("(")[0].strip(): matched_rows[col].values[0]
                for col in raw_columns
                if col in matched_rows
            }
            result["results"][smiles] = smiles_results  # type: ignore
    return result


@mcp.tool()
@mock_if_available("reinforcement_learning", mocked_data)
def reinforcement_learning(
    input_molecules: List[str] = [],
    criteria: Dict[str, Dict] = {},
    model: str = "molformer",
    epochs: int = 50,
) -> Any:
    """
    Performs Reinforcement Learning (RL)-based molecule generation.

    **Usage Rules:**
    - Use the **Reinforcement Learning** tool only if the user explicitly says: "reinforcement learning", "RL", "staged learning", or "RL-based generation". This must appear **verbatim** in the user's message.
    - If the user only asks to "generate molecules with higher/lower/different properties" but does not mention reinforcement learning, **DO NOT call** this tool. Instead, use another appropriate tool.
    - **Never assume** that property optimization implies reinforcement learning. The user must explicitly state it for reinforcement learning to be used.

    Instructions for providing input parameters:
    - Before running Reinforcement Learning, you must first obtain the numeric property values (from the appropriate tool if not present in the user query) and use them to build the required 'criteria' input.
    - You MUST ALWAYS include the REQUIRED 'criteria' argument.
    - If no specific `criteria` is provided by the user, set the property keys to empty dictionaries (For example: criteria = {"property_key": {}}).

    **Examples of correctly populating the required positional argument: `criteria` (if a property value is provided for criteria, BOTH 'low' and 'high' must be set). Please note the following instructions only apply to the `criteria` dictionary and nothing else.:**
    - Above reference MW → criteria = {"MolecularWeight": {"low": <ref MW>, "high": 99999.0}}
    - Below 450 MW → criteria = {"MolecularWeight": {"low": -99999.0, "high": 450.0}}
    - Between 220-450 MW → criteria = {"MolecularWeight": {"low": 220.0, "high": 450.0}}
    Use raw reference values (not rounded) for “higher/lower than reference” cases.

    **Model selection:**
    - If user requests "use reinvent reinforcement learning", set `model` = 'reinvent'.
    - Otherwise, default to `model` = 'molformer'.

    Args:
        input_molecules: List of input molecules (SMILES) to be used as reference for the RL-based generation.
        criteria:  Dictionary where keys are the scoring components. Their respective values are dictionaries containing 'low' and 'high' keys. Each entry may specify numeric 'low' and/or 'high'. If a bound is omitted, defaults MUST added: low = -99999.0, high =  99999.0. If no scoring property is provided by the user, set criteria to an empty dictionary (i.e., {}).
        model: The RL model to use. Supported values are "molformer" and "reinvent". Default is "molformer".
        epochs: Number of RL epochs to run. Default is 50.

    """  # noqa: E501

    logging.info(f"Reinforcement learning called with inputs={locals()}")

    template = load_rl_template()
    template = set_device(template, f"{os.path.join(ENV_PATH, 'env')}")
    template = filter_rl_scoring_components(template, criteria)
    template = update_template_with_model(template, model, input_molecules, ENV_PATH)
    template["stage"][0]["max_steps"] = epochs
    template["stage"][0]["min_steps"] = min(template["stage"][0]["min_steps"] , epochs)

    path_obj = create_temp_dir()
    logging.info(f"Created temp directory: {path_obj}")
    output_path = path_obj / "reinforcement_learning_output"
    template["parameters"]["summary_csv_prefix"] = str(output_path)
    template["stage"][0]["chkpt_file"] = str(path_obj / "checkpoint.chkpt")

    config_file = path_obj / "reinforcement_learning_config.json"
    with open(config_file, "w") as fileobj:
        json.dump(template, fileobj)

    arglist = [
        "conda",
        "run",
        "--live-stream",
        "-p",
        f"{os.path.join(ENV_PATH, 'env')}",
        "reinvent",
        str(config_file),
    ]
    logging.info(" ".join(arglist))
    _ = subprocess.check_output(arglist).decode("utf-8")

    result = {"output_path": f"{output_path}_1.csv"}
    return result


@mcp.resource("scoring://descriptions_{name}", mime_type="application/json")
def get_scoring_descriptions(name: str) -> Dict[str, str]:
    return SCORING_DESCRIPTIONS.get(name, {})


@mcp.resource("scoring://supported_components", mime_type="application/json")
def get_supported_scoring_components() -> List[str]:
    return SUPPORT_SCORING_COMPONENTS


if __name__ == "__main__":
    mcp.run("streamable-http")
