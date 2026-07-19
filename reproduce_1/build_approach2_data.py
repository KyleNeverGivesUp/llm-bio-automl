"""Approach 2 data (#1): the proxy-labelled molecules + LogD as a multitask training pool.
#1 trained a 2nd multitask GNN on the SVR-imputed proxy pEC50 labels, WITH LogD as an auxiliary
task, to make a weak-end specialist. Targets: [pEC50(proxy), logD]. No calibration anywhere.
"""
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
def canon(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None

proxy = pd.read_csv("data/pxr_activity/proxy_train.csv")            # SMILES, pEC50 (proxy)
proxy["logD"] = np.nan
logD = pd.read_csv("scratchpad/logD_lipophilicity.csv")
logd_only = pd.DataFrame({"SMILES": logD["smiles"].map(canon), "pEC50": np.nan, "logD": logD["exp"]})
pool = pd.concat([proxy[["SMILES","pEC50","logD"]], logd_only], ignore_index=True)
pool = pool[pool["SMILES"].notna()]

test = pd.read_csv("data/pxr_activity/test.csv")
test_c = {c for c in (canon(s) for s in test["SMILES"]) if c}
assert len(set(pool["SMILES"]) & test_c) == 0, "LEAK: test molecule in Approach-2 pool"
pool.to_csv("data/pxr_activity/train_approach2.csv", index=False)
print(f"Approach-2 池 {len(pool)} 行 (防泄漏 OK): pEC50(proxy)={pool['pEC50'].notna().sum()}, logD={pool['logD'].notna().sum()}")
