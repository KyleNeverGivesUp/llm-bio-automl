"""Extract IBM SmallMoleculeMultiView (biomed.sm.mv-te-84m) embeddings — frozen-feature member.

A genuinely different family from our graph(CheMeleon)/3D(Uni-Mol) members: a multi-view
foundation model fusing GRAPH + 2D-IMAGE + TEXT views (the image view is novel to our stack).
We use it like the 0.538 solution used MolE: extract a frozen embedding per molecule, then
fit LightGBM downstream (done separately, in the main env).

Loads the model ONCE and reuses it (get_embeddings accepts pretrained_model=).
Runs CPU-only inference — no GPU needed. Writes ibm_emb_train.csv + ibm_emb_test.csv
(Molecule Name, SMILES, ibm_0..ibm_{d-1}); unparseable molecules get a zero row.

Run in the isolated ibm_venv (see setup_ibm_pod.sh):
  ~/ibm_venv/bin/python ibm_embed.py --data-dir ~ --out-dir ~
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "ibm/biomed.sm.mv-te-84m"
SMILES, NAME = "SMILES", "Molecule Name"


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract IBM multi-view embeddings for train+test.")
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--smoke", action="store_true", help="only first 8 rows, to verify the pipeline")
    args = ap.parse_args()

    from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
    from bmfm_sm.core.data_modules.namespace import LateFusionStrategy

    print("loading IBM multi-view model (once)...", flush=True)
    model = SmallMoleculeMultiViewModel.from_pretrained(
        LateFusionStrategy.ATTENTIONAL, model_path=MODEL, inference_mode=True, huggingface=True
    )

    def embed(smi: str) -> np.ndarray:
        e = SmallMoleculeMultiViewModel.get_embeddings(smiles=str(smi), pretrained_model=model)
        return np.asarray(e.detach().cpu() if hasattr(e, "detach") else e, dtype=np.float32).ravel()

    dim = embed("CCO").shape[0]
    print(f"embedding dim = {dim}", flush=True)

    for name in ["train", "test"]:
        df = pd.read_csv(args.data_dir / f"{name}.csv").reset_index(drop=True)
        if args.smoke:
            df = df.iloc[:8].reset_index(drop=True)
        rows = np.zeros((len(df), dim), dtype=np.float32)
        t0 = time.time()
        for i, s in enumerate(df[SMILES]):
            try:
                rows[i] = embed(s)
            except Exception as exc:  # unparseable / featurization failure -> zero row
                print(f"  [{name}] row {i} failed: {exc}", flush=True)
            if i % 200 == 0:
                print(f"  [{name}] {i}/{len(df)}  ({time.time()-t0:.0f}s)", flush=True)
        out = pd.DataFrame(rows, columns=[f"ibm_{j}" for j in range(dim)])
        out.insert(0, SMILES, df[SMILES].values)
        if NAME in df.columns:
            out.insert(0, NAME, df[NAME].values)
        out_path = args.out_dir / f"ibm_emb_{name}.csv"
        out.to_csv(out_path, index=False)
        print(f"wrote {out_path}  shape={rows.shape}", flush=True)


if __name__ == "__main__":
    main()
