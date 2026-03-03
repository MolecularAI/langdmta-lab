"""
Synthesis MCP Server
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from rxnutils.routes.readers import read_aizynthfinder_dict

from langdmta_lab.config import ENV_DIR
from langdmta_lab.mcps.helpers import (
    create_mcp_server_instance,
    load_mock_data,
    mock_if_available,
)

# Create the FastMCP server instance
mcp = create_mcp_server_instance("synthesis-server", "SYNTHESIS_MCP_PORT")

mocked_data = load_mock_data(os.path.dirname(__file__))

ENV_PATH = os.path.join(ENV_DIR, "synthesis")


@mcp.tool()
@mock_if_available("aizynthfinder", mocked_data)
def aizynthfinder(product_smiles: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    This tool PREDICTS synthesis routes for input molecules in SMILES format, using a machine learning model to predict routes on the fly. If any molecule is given by name or ID, convert it to SMILES first.
    The output is a list of reaction routes per molecule, where each route uses the format "reactants >> products", with reactants separated by ".".
    Each output includes a "stock" dictionary indicating available reactants.
    Always provide the full reaction steps and stock details for all input molecules in the final answer, clearly explaining the synthesis process and stock status for each input molecule.

    When to use
    • User asks how to predict, make, or sythesize one or more molecules.

    What it is NOT
    • It does not fetch historical routes.
    • It does not expand a full reaction network.

    Args:
        product_smiles: Molecules to be synthesised provided as a list of SMILES strings

    Returns:
        A dictionary mapping each input product SMILES to its synthesis information
    """  # noqa: E501
    logging.info(f"aizynthfinder called with inputs={locals()}")

    path = tempfile.mkdtemp()
    path_obj = Path(path)
    path_obj.mkdir(exist_ok=True, parents=True)

    smiles_file = path_obj / "smiles.txt"
    with open(smiles_file, "w") as fileobj:
        fileobj.write("\n".join(product_smiles))

    output_path = path_obj / "output.json.gz"
    arglist = [
        "conda",
        "run",
        "--live-stream",
        "-p",
        f"{os.path.join(ENV_PATH, 'env')}",
        "aizynthcli",
        "--config",
        f"{os.path.join(ENV_PATH, 'data', 'config.yml')}",
        "--smiles",
        str(smiles_file),
        "--output",
        str(output_path),
        "--policy",
        "uspto",
    ]
    _ = subprocess.check_output(arglist).decode("utf-8")

    data = pd.read_json(output_path, orient="table")
    output = {}
    for smiles, (_, row) in zip(product_smiles, data.iterrows()):
        output[smiles] = {
            "reactions": read_aizynthfinder_dict(row.trees[0]).reaction_smiles(),
            "stock": row.precursors_in_stock.split(", "),
        }
    return output


@mcp.tool()
@mock_if_available("precedent_finder", mocked_data)
def precedent_finder(
    reaction_smiles: List[str], top_n: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    This tool predicts similar reactions for input reaction SMILES. Do not use this tool unless the user query explicitly requests similar reactions.
    It returns a list where each element corresponds to an input reaction and contains the top similar reactions, sorted by relevance.
    Each similar reaction is represented as a dictionary with keys: 'ReactionSmiles' (the similar reaction’s SMILES) and 'goodness' (score).

    Args:
        reaction_smiles: List of multiple reaction SMILES strings similar to this example: ['CC(=O)[S-].O=Nc1ccc(O)cc1>>CC(=O)Nc1ccc(O)cc1', 'Nc1ncc(Br)nc1Br.C1COCCN1>>Nc1ncc(Br)nc1N1CCOCC1'].
        top_n: maximum number of similar reactions that they want to see
    """  # noqa: E501
    logging.info(f"precedent_finder called with inputs={locals()}")

    def find_precedents(smiles: str, top_n: int) -> List[Dict[str, Any]]:
        path = tempfile.mkdtemp()
        path_obj = Path(path)
        path_obj.mkdir(exist_ok=True, parents=True)

        arglist = [
            "conda",
            "run",
            "--live-stream",
            "-p",
            f"{os.path.join(ENV_PATH, 'rxnmapper')}",
            "python",
            f"{os.path.join(ENV_PATH, 'mapper.py')}",
            smiles,
        ]
        output = subprocess.check_output(arglist).decode("utf-8")
        mapped_smiles = output.strip()

        arglist = [
            "conda",
            "run",
            "--live-stream",
            "-p",
            f"{os.path.join(ENV_PATH, 'env')}",
            "python",
            f"{os.path.join(ENV_PATH, 'precedent_finder', 'precedent_finder.py')}",
            "--smiles",
            mapped_smiles,
            "--output",
            str(path_obj / "precedents.csv"),
        ]
        _ = subprocess.check_output(arglist).decode("utf-8")

        data = pd.read_csv(path_obj / "precedents.csv", sep="\t")
        columns = ["ReactionSmiles", "goodness"]
        return data[columns].iloc[:top_n].to_dict(orient="records")

    return {smiles: find_precedents(smiles, top_n) for smiles in reaction_smiles}


if __name__ == "__main__":
    mcp.run("streamable-http")
