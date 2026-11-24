
from pathlib import Path
import sys, json, hashlib, platform, subprocess
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Crippen, Lipinski, QED
from code import config

def root_import():
    sys.path.insert(0,str(config.PROJECT_ROOT))
def morgan(mol):
    g=rdFingerprintGenerator.GetMorganGenerator(radius=config.MORGAN_RADIUS,fpSize=config.MORGAN_N_BITS)
    return np.asarray(g.GetFingerprintAsNumPy(mol),dtype=np.float32)
def descriptors(mol):
    return np.array([Descriptors.MolWt(mol),Crippen.MolLogP(mol),Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),Lipinski.NumRotatableBonds(mol),Descriptors.TPSA(mol),
        Descriptors.MolMR(mol),Descriptors.FractionCSP3(mol)],dtype=np.float32)
def mol_from_smiles(s):
    return Chem.MolFromSmiles(str(s))
def lipinski(m):
    return int(Descriptors.MolWt(m)>500)+int(Crippen.MolLogP(m)>5)+int(Lipinski.NumHDonors(m)>5)+int(Lipinski.NumHAcceptors(m)>10)
def versions():
    out={"python":platform.python_version(),"platform":platform.platform()}
    for name in ["rdkit","numpy","pandas","sklearn","torch"]:
        try:
            mod=__import__(name); out[name]=getattr(mod,"__version__","unknown")
        except Exception: out[name]="not_installed"
    try:
        import importlib.metadata as md; out["admet-ai"]=md.version("admet-ai")
    except Exception: out["admet-ai"]="not_installed"
    try:
        import importlib.metadata as md; out["PyTDC"]=md.version("PyTDC")
    except Exception: out["PyTDC"]="not_installed"
    return out
def atomic_csv(df,path): path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False)


def features(smiles):
    rows=[]
    for s in smiles:
        m=mol_from_smiles(s)
        if m is None:
            raise ValueError(f"Invalid SMILES in feature generation: {s}")
        rows.append(np.concatenate([morgan(m), descriptors(m)]))
    return np.asarray(rows,dtype=np.float32)
