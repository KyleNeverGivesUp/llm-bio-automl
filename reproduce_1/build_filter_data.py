"""#1 technique 7 (faithful, ~30): filter training pEC50 compounds with elements or macrocyclic
structures not found in either test set. #1's report: "removing about ~30 compounds with elements
or high-level structures (macrocycles, long acyclic molecules) not found in either test set."

Match #1's ~30 exactly: elements-not-in-test + macrocycle (standard definition, ring >= 12; the test
set's largest ring is 8, so any >=12 ring is test-absent). Filter applies to the pEC50 broad rows
only; aux rows kept."""
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

def feats(sm):
    m = Chem.MolFromSmiles(str(sm))
    if m is None: return None
    rings = [len(r) for r in m.GetRingInfo().AtomRings()]
    return set(a.GetSymbol() for a in m.GetAtoms()), (max(rings) if rings else 0)

pool = pd.read_csv("data/pxr_activity/train_approach1.csv")
broad = pool[pool["pEC50"].notna()].reset_index(drop=True); aux = pool[pool["pEC50"].isna()]
test = pd.read_csv("data/pxr_activity/test.csv")
te = [x for x in (feats(s) for s in test["SMILES"]) if x]
test_elems = set().union(*[x[0] for x in te]); test_maxring = max(x[1] for x in te)
print(f"test envelope: elements={sorted(test_elems)}  max_ring={test_maxring}")

from collections import Counter
drop = []
for i, sm in enumerate(broad["SMILES"]):
    f = feats(sm)
    if f is None: continue
    elems, maxring = f
    if not elems <= test_elems: drop.append((i, "elem"))
    elif maxring >= 12: drop.append((i, "macrocycle"))     # standard macrocycle; test has none (max 8)
print(f"\nfiltered {len(drop)} / {len(broad)}  reasons: {dict(Counter(r for _, r in drop))}")

drop_set = {i for i, _ in drop}
kept_idx = [i for i in range(len(broad)) if i not in drop_set]   # original positions of kept broad rows
keep = broad.iloc[kept_idx].reset_index(drop=True)
out = pd.concat([keep, aux], ignore_index=True)
out.to_csv("data/pxr_activity/train_approach1_filtered.csv", index=False)

# aligned fold file: each kept row keeps its ORIGINAL calibrated scaffold fold, re-indexed 0..N-1
import json
orig = json.loads(open("data/pxr_activity/folds_calibrated.json").read())["assignments"]
aligned = {"assignments": {str(p): int(orig[str(oi)]) for p, oi in enumerate(kept_idx)}}
json.dump(aligned, open("data/pxr_activity/folds_approach1_filtered.json", "w"))
print(f"wrote train_approach1_filtered.csv: {len(keep)} broad + {len(aux)} aux = {len(out)} rows")
print(f"wrote folds_approach1_filtered.json: {len(aligned['assignments'])} kept-row fold labels")
