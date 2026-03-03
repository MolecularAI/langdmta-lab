from typing import List

from rdkit import Chem


def validate_smiles(smiles_list: List[str]) -> List[str]:
    validated_list = []
    for item in smiles_list:
        if not isinstance(item, str):
            raise TypeError(
                f"expected each item in the list to be string, but got {type(item)}"
            )
        mol = Chem.MolFromSmiles(item, sanitize=False)
        if not mol:
            raise ValueError(f"Invalid SMILES: {item}")
        validated_list.append(item)
    return validated_list


def validate_filename(filename: str) -> str:
    if not isinstance(filename, str):
        raise TypeError(f"Expected filename to be string, but got {type(filename)}")
    else:
        filename = filename.rstrip("\n").strip("'").strip('"')
        if filename.endswith(".csv") or filename.endswith(".smi"):
            return filename
        else:
            raise ValueError(
                f"Expected filename should end with .csv or .smi, but got {filename}"
            )
