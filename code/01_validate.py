
import argparse, sys
from pathlib import Path
import pandas as pd
from rdkit import Chem
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from code import config
def main():
    p=argparse.ArgumentParser(); p.add_argument("--ligands",required=True); p.add_argument("--seed",type=int,default=config.SEED); a=p.parse_args()
    config.dirs(); config.seed_everything(a.seed)
    f=Path(a.ligands)
    if not f.exists(): raise FileNotFoundError(f)
    df=pd.read_csv(f)
    if not {"candidate_id","smiles"}<=set(df.columns): raise ValueError("Input must contain candidate_id,smiles")
    if df.candidate_id.isna().any() or df.smiles.isna().any(): raise ValueError("Missing candidate_id/smiles")
    if df.candidate_id.duplicated().any(): raise ValueError("candidate_id must be unique")
    valid=[]; invalid=[]; dup=[]; seen={}
    for _,r in df.iterrows():
        cid=str(r.candidate_id); sm=str(r.smiles).strip(); m=Chem.MolFromSmiles(sm)
        if m is None: invalid.append({"candidate_id":cid,"smiles":sm,"reason":"invalid_smiles"}); continue
        can=Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)
        if can in seen:
            dup.append({"candidate_id":cid,"smiles":sm,"canonical_smiles":can,"kept_candidate_id":seen[can]}); continue
        seen[can]=cid; valid.append({"candidate_id":cid,"smiles":sm,"canonical_smiles":can})
    pd.DataFrame(valid).to_csv(config.VALIDATED_FILE,index=False)
    pd.DataFrame(invalid,columns=["candidate_id","smiles","reason"]).to_csv(config.INVALID_FILE,index=False)
    pd.DataFrame(dup,columns=["candidate_id","smiles","canonical_smiles","kept_candidate_id"]).to_csv(config.DUPLICATES_FILE,index=False)
    if not valid: raise RuntimeError("No valid unique molecules")
    print(f"valid={len(valid)} invalid={len(invalid)} duplicates={len(dup)}")
if __name__=="__main__": main()
