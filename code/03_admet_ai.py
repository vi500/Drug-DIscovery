
import argparse,sys,importlib.metadata as md
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from code import config
def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default=str(config.VALIDATED_FILE)); p.add_argument("--seed",type=int,default=config.SEED); a=p.parse_args()
    config.dirs(); config.seed_everything(a.seed)
    try:
        from admet_ai import ADMETModel
    except Exception as e:
        raise RuntimeError("ADMET-AI is required. Install the pinned dependency in requirements.txt. No fallback is used.") from e
    try: version=md.version("admet-ai")
    except Exception: version="unknown"
    df=pd.read_csv(a.input)
    if df.empty: raise RuntimeError("No molecules for ADMET-AI")
    model=ADMETModel()
    preds=model.predict(smiles=df["smiles"].tolist())
    if not isinstance(preds,pd.DataFrame): preds=pd.DataFrame(preds)
    preds=preds.reset_index().rename(columns={"index":"smiles"})
    # Match by canonical SMILES, never by row order.
    if "smiles" not in preds: raise RuntimeError("ADMET-AI returned no smiles key")
    merged=df[["candidate_id","smiles"]].merge(preds,on="smiles",how="left",validate="one_to_one")
    if len(merged)!=len(df) or merged.drop(columns=["candidate_id","smiles"]).isna().all(axis=1).any():
        raise RuntimeError("ADMET-AI output could not be matched to all candidates by SMILES")
    merged.attrs["admet_ai_version"]=version
    merged.to_csv(config.ADMET_AI_FILE,index=False)
    (config.ADMET_AI_CACHE_DIR/"metadata.txt").write_text(f"admet-ai={version}\n",encoding="utf-8")
    print(f"ADMET-AI {version}: {len(merged)} molecules")
if __name__=="__main__": main()
