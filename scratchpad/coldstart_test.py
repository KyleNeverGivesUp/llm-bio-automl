"""Cold-start discovery test: empty the registry + seed, run retrieve, see if chemeleon
is autonomously re-discovered via the arXiv literature channel -> HF/GitHub search.
Restores the registry afterward."""
import json
import shutil
from pathlib import Path

from src.agent import model_registry
from src.agent.retrieval_agent import RetrievalAgent

REG = model_registry.REGISTRY_PATH
backup = REG.with_suffix(".json.coldbak")

# --- back up + COLD START (empty seed + registry) ---
if REG.exists():
    shutil.copy(REG, backup)
    REG.unlink()
model_registry._SEED = []            # in-memory only; the .py file is untouched
print(f"[cold] registry emptied (seed patched to []), backup at {backup.name}")

setup = {
    "task": {"type": "regression", "domain": "molecular property",
             "summary": "predict PXR (pregnane-X receptor) pEC50 activity from a molecule's SMILES"},
    "metric": "RAE",
    "schema": {"target_col": "pEC50", "smiles_col": "SMILES"},
}

try:
    r = RetrievalAgent().run(setup, top_k=12, out_path="scratchpad/coldstart_retrieval.json")

    def hit(s):
        return "chemeleon" in str(s).lower()

    lit = r.get("literature_models") or []
    raw = r.get("online_raw") or []
    cands = r.get("candidates") or []
    sel = r.get("selected") or []

    print("\n================ COLD-START RESULT ================")
    print("went_online          :", r.get("went_online"))
    print("retrieve source      :", r.get("source"))
    print("\n--- ① arXiv literature extraction (literature_models) ---")
    print(" ", lit if lit else "(empty — arXiv failed or named nothing)")
    print("  chemeleon in it?   :", any(hit(m) for m in lit))
    print("\n--- ② online search HF/GitHub/Zenodo (online_raw) — chemeleon hits ---")
    chem_raw = [{"ref": c.get("ref"), "src": c.get("library") or c.get("source")} for c in raw if hit(c.get("ref"))]
    for c in chem_raw[:8]:
        print("  ", c)
    if not chem_raw:
        print("   (no chemeleon in online results)")
    print("\n--- ③ candidates pool (what LLM ranked from) — chemeleon hits ---")
    print(" ", [c.get("ref") for c in cands if hit(c.get("ref"))] or "(none)")
    print("\n--- ④ selected (final picks) ---")
    for s in sel:
        print(f"   {str(s.get('ref'))[:50]:<50} family={s.get('family')} mode={s.get('mode')}")
    print("\n>>> VERDICT: chemeleon auto-discovered?  ",
          any(hit(m) for m in lit) or bool(chem_raw) or any(hit(c.get("ref")) for c in cands))
    print("==================================================")
finally:
    # --- restore ---
    if backup.exists():
        shutil.copy(backup, REG)
        backup.unlink()
        print(f"\n[restore] registry restored from backup ({model_registry.load().__len__()} models back)")
