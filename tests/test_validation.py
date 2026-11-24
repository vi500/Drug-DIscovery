
from rdkit import Chem
def test_valid_smiles(): assert Chem.MolFromSmiles("CCO") is not None
def test_invalid_smiles(): assert Chem.MolFromSmiles("NOT_A_SMILES") is None
