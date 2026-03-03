from typing import Optional

import pubchempy as pcp
from rxnutils.chem.utils import desalt_molecules


def synonym2smiles(synonym: str) -> Optional[str]:
    try:
        compounds = pcp.get_compounds(synonym, "name")
    except Exception:
        return None

    if not compounds:
        return None

    smiles = ""
    for compound in compounds:
        smiles = compound.smiles or compound.canonical_smiles
        if "." not in smiles:
            break
    smiles = desalt_molecules([smiles])[0]  # Remove salts
    return smiles  # Return the first non-mixture SMILES found
