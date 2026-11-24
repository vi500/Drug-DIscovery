
import argparse,sys
from pathlib import Path
import pandas as pd,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from code import config
def main():
    p=argparse.ArgumentParser(); p.add_argument("--affinity",required=True); p.add_argument("--seed",type=int,default=config.SEED); a=p.parse_args()
    config.seed_everything(a.seed); s=pd.read_csv(config.SAFETY_SCORES_FILE); aff=pd.read_csv(a.affinity)
    if not {"candidate_id","affinity_norm"}<=set(aff.columns): raise ValueError("Affinity CSV needs candidate_id,affinity_norm")
    if aff.candidate_id.duplicated().any(): raise ValueError("Affinity candidate_id must be unique")
    m=s.merge(aff[["candidate_id","affinity_norm"]],on="candidate_id",how="left",validate="one_to_one")
    m["affinity_missing"]=m.affinity_norm.isna()
    m["reward"]=np.where(m.hard_fail,0.0,config.W_AFFINITY*m.affinity_norm.fillna(0)+config.W_SAFETY*m.safety_score)
    m=m.sort_values(["hard_fail","reward"],ascending=[True,False]).reset_index(drop=True); m["rank"]=np.arange(1,len(m)+1)
    m.to_csv(config.REWARD_FILE,index=False)
    rec=m[(~m.hard_fail)&(~m.affinity_missing)&(m.safety_score>=config.MIN_RECOMMEND_SAFETY)].head(config.SHORTLIST_SIZE)
    rec.to_csv(config.RECOMMENDED_FILE,index=False)
    print(f"ranked={len(m)} recommended={len(rec)} missing_affinity={int(m.affinity_missing.sum())}")
if __name__=="__main__": main()
