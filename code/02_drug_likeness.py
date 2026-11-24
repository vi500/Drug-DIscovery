
import argparse,sys
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors,Crippen,Lipinski,QED
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from code import config
def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default=str(config.VALIDATED_FILE)); p.add_argument("--seed",type=int,default=config.SEED); a=p.parse_args()
    config.dirs(); config.seed_everything(a.seed); df=pd.read_csv(a.input)
    out=[]
    for _,r in df.iterrows():
        m=Chem.MolFromSmiles(r.canonical_smiles)
        if m is None: raise RuntimeError(f"RDKit failed on {r.candidate_id}")
        v=int(Descriptors.MolWt(m)>500)+int(Crippen.MolLogP(m)>5)+int(Lipinski.NumHDonors(m)>5)+int(Lipinski.NumHAcceptors(m)>10)
        out.append({"candidate_id":r.candidate_id,"smiles":r.canonical_smiles,"MW":Descriptors.MolWt(m),"LogP":Crippen.MolLogP(m),"HBD":Lipinski.NumHDonors(m),"HBA":Lipinski.NumHAcceptors(m),"RotatableBonds":Lipinski.NumRotatableBonds(m),"TPSA":Descriptors.TPSA(m),"QED":QED.qed(m),"LipinskiViolations":v})
    pd.DataFrame(out).to_csv(config.DRUG_LIKENESS_FILE,index=False); print(f"wrote {config.DRUG_LIKENESS_FILE}")
if __name__=="__main__": main()
