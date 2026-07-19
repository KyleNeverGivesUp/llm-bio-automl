import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
def canon(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None
CONC = {8.251e-06:"fc_8p25", 3.30e-05:"fc_33", 9.901e-05:"fc_99"}   # 丢掉 0.98uM(只1个)
TARGETS = ["pEC50","fc_8p25","fc_33","fc_99","logD"]

sc = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_single_concentration_TRAIN.csv")
sc["cs"]=sc["SMILES"].map(canon); sc=sc[sc["cs"].notna() & sc["concentration_M"].isin(CONC)]
sc["col"]=sc["concentration_M"].map(CONC)
sc_piv=sc.pivot_table(index="cs",columns="col",values="log2_fc_estimate",aggfunc="mean").reset_index()
for c in CONC.values():
    if c not in sc_piv: sc_piv[c]=np.nan

tr=pd.read_csv("data/pxr_activity/train.csv"); tr["cs"]=tr["SMILES"].map(canon)
logD=pd.read_csv("scratchpad/logD_lipophilicity.csv"); logD["cs"]=logD["smiles"].map(canon)
test=pd.read_csv("data/pxr_activity/test.csv"); test_c={c for c in (canon(s) for s in test["SMILES"]) if c}

broad=tr[["cs","pEC50"]].merge(sc_piv,on="cs",how="left"); broad["logD"]=np.nan   # 保持 tr 顺序,在最前
sc_only=sc_piv[~sc_piv["cs"].isin(set(tr["cs"]))].copy()
for c in ["pEC50","logD"]: sc_only[c]=np.nan
logd_only=pd.DataFrame({"cs":logD["cs"],"logD":logD["exp"]})
for c in ["pEC50"]+list(CONC.values()): logd_only[c]=np.nan

pool=pd.concat([broad,sc_only,logd_only],ignore_index=True)
pool=pool[pool["cs"].notna()].rename(columns={"cs":"SMILES"})[["SMILES"]+TARGETS]
assert len(set(pool["SMILES"]) & test_c)==0, "LEAK"
# broad 行(前 len(tr) 个)顺序 == train.csv,fold 可按位置对齐
assert (pool["SMILES"].iloc[:len(tr)].to_numpy()==tr["cs"].to_numpy()).all(), "broad 顺序错位"
pool.to_csv("data/pxr_activity/train_approach1.csv", index=False)
print(f"Approach-1 池 {len(pool)} 行 (broad前{len(tr)}个); 防泄漏+折对齐 OK")
for t in TARGETS: print(f"  {t:8s} 非空 {pool[t].notna().sum():6d}")
