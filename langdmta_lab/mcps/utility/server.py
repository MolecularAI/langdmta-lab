"""
Utility MCP Server

A general-purpose MCP server providing various utility tools including:
- Chemical/molecular data queries (PubChem)
- SMILES validation
"""

import logging
import os
from typing import Dict, List, Optional

from rdkit import Chem

from langdmta_lab.mcps.helpers import (
    create_mcp_server_instance,
    load_mock_data,
    mock_if_available,
)
from langdmta_lab.mcps.utility.utility_helpers import synonym2smiles

# Create the FastMCP server instance
mcp = create_mcp_server_instance("utility-server", "UTILITY_MCP_PORT")

mocked_data = load_mock_data(os.path.dirname(__file__))


@mcp.tool()
@mock_if_available("synonyms2smiles", mocked_data)
def synonyms2smiles(synonyms: List[str]) -> Dict[str, Optional[str]]:
    """
    Converts molecule names to their SMILES representations. Always use this tool when molecules are described using names (e.g., "Aspirin") or IDs.
    SMARTS are a superset of SMILES, when SMARTS pattern are required, and not explicitly provided by the user, this tool should also be used to obtain the corresponding SMILES string
    SMILES must not be passed to this tool as they would cause internal errors. If provided compounds contains both synonyms and SMILES, make sure only pass synonyms to this tool.

    Args:
        synonyms: List of molecule names, substructure names or compound identifier. Example:['Paracetamol', 'PI183844', 'another-id', 'amide']. ONLY include the exact name here, don't include any descriptors like "group, moiety, or substructure", for example it should be ['amide'] not ['amide ring']

    Returns:
        A dictionary mapping each input synonym to its corresponding SMILES string.
    """  # noqa: E501

    logging.info(f"Synonyms2smiles called with inputs={locals()}")

    return {synonym: synonym2smiles(synonym) for synonym in synonyms}


@mcp.tool()
def smilesvalidator(compounds: List[str]) -> Dict[str, bool]:
    """
    This tool is used to validate SMILES. It should always be used to determine which compounds in a list are actual SMILES or just compound names.
    Whenever a list of compounds is provided by the user (e.g., ['CPD123456', 'CC(=O)OC(C)C']), this should be the first tool to be used to understand which entries are valid SMILES or not. Even if a user asks to get the SMILES of a compound name, the name should be first validated with this tool to determine if that is already a SMILES.
    Entries that are valid SMILES can be passed down to tools that accept SMILES. Invalid entries should be treated accordingly, another tool to search SMILES based on synonyms must be used first.

    Args:
        compounds: List of compound names to be validated

    Returns:
        A dictionary mapping each input compounds to weather it is SMILES or not.
    """  # noqa: E501

    logging.info(f"smilesvalidator called with inputs={locals()}")

    return {c: True if Chem.MolFromSmiles(c) else False for c in compounds}


if __name__ == "__main__":
    mcp.run("streamable-http")
