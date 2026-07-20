"""#1 technique 7: filter ~30 training pEC50 compounds with elements or high-level structures
(macrocycles, long acyclic chains) not found in EITHER test set. Filter applies to the pEC50 broad
rows only; aux rows kept."""
import pandas as pd, numpy as np, networkx as nx
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

def longest_acyclic_chain(m):
    """longest path (in atoms) through NON-ring atoms — captures long chains, not big ring systems."""
    g = nx.Graph()
    ring = {a.GetIdx() for a in m.GetAtoms() if a.IsInRing()}
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i not in ring and j not in ring:
            g.add_edge(i, j)
    best = 0
    for comp in nx.connected_components(g):
        sub = g.subgraph(comp)
        ecc = nx.eccentricity(sub)
        best = max(best, max(ecc.values()) + 1)   # diameter in nodes
    return best

def feats(sm):
    m = Chem.MolFromSmiles(str(sm))
    if m is None: return None
    rings = [len(r) for r in m.GetRingInfo().AtomRings()]
    return (set(a.GetSymbol() for a in m.GetAtoms()), max(rings) if rings else 0, longest_acyclic_chain(m))

pool = pd.read_csv("data/pxr_activity/train_approach1.csv")
broad = pool[pool["pEC50"].notna()].reset_index(drop=True); aux = pool[pool["pEC50"].isna()]
test = pd.read_csv("data/pxr_activity/test.csv")
te = [x for x in (feats(s) for s in test["SMILES"]) if x]
test_elems = set().union(*[x[0] for x in te]); test_maxring = max(x[1] for x in te); test_maxchain = max(x[2] for x in te)
print(f"test envelope: elements={sorted(test_elems)}  max_ring={test_maxring}  max_acyclic_chain={test_maxchain}")

from collections import Counter
drop = []
for i, sm in enumerate(broad["SMILES"]):
    f = feats(sm)
    if f is None: continue
    elems, maxring, chain = f
    if not elems <= test_elems: drop.append((i, "elem"))
    elif maxring > test_maxring: drop.append((i, "macro/bigring"))
    elif chain > test_maxchain: drop.append((i, "long_chain"))
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
