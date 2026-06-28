"""Fine-tuning as a first-class pipeline plan — the capability AIBuildAI has and we left out.

The professor approved bringing LLM-orchestrated fine-tuning into the pipeline. The design:
the LLM (designer) proposes a FINE-TUNE PLAN (which backbone, how many epochs, multitask or
not, loss); this module is the "coder" — it instantiates a VERIFIED TEMPLATE script from that
plan (template-based codegen = safe + leak-free, vs raw codegen) and produces the exact GPU
command. The "tuner" runs it (on DSMLP, next to the A5000); then we collect its OOF/test into
the aggregator's plan-dir format so it stacks + gets judged like any other member.

Two verified templates so far (each reached its number in this project):
  - chemeleon : scripts/finetune_cheme_mt5.py  (CheMeleon multitask + MAE, single 0.5904)
  - unimol    : scripts/finetune_unimol.py     (Uni-Mol 3D, single 0.6248)

A fine-tune plan is a dict the designer emits, e.g.
    {"backbone": "chemeleon", "epochs": 50, "label": "ft_cheme_mt"}
    {"backbone": "unimol", "epochs": 15, "tta": 0, "label": "ft_unimol"}
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SMILES, NAME = "SMILES", "Molecule Name"


@dataclass
class FineTunePlan:
    """What the LLM designer proposes for one fine-tuning run."""
    backbone: str                      # "chemeleon" | "unimol"
    epochs: int = 50
    tta: int = 0                       # unimol only: randomized-SMILES test-time aug
    label: str | None = None
    fast: bool = False                 # smoke test: 1 fold, tiny epochs, ~60 rows — verify auto-FT plumbing only
    extra: dict = field(default_factory=dict)

    @property
    def plan_id(self) -> str:
        return self.label or f"ft_{self.backbone}_e{self.epochs}"


# Verified templates. Each entry knows its script, how to turn a plan into CLI args, and where
# the script writes its OOF/test CSVs. Adding a new fine-tunable backbone = one entry here.
TEMPLATES: dict[str, dict] = {
    "chemeleon": {
        "script": "scripts/finetune_cheme_mt5.py",
        "oof": "oof_cheme_mt5.csv", "test": "test_cheme_mt5.csv",
        "args": lambda p, data, out: ["--epochs", str(p.epochs), "--accelerator", "gpu",
                                       "--data-dir", str(data), "--out-dir", str(out)]
                                      + (["--folds", "1", "--max-rows", "60"] if p.fast else []),  # smoke
        "needs": ["train_multitask5.csv", "sc_extra.csv", "test.csv", "folds_calibrated.json"],
    },
    "unimol": {
        "script": "scripts/finetune_unimol.py",
        "oof": "oof_unimol.csv", "test": "test_unimol.csv",
        "args": lambda p, data, out: ["--epochs", str(p.epochs), "--tta", str(p.tta),
                                      "--accelerator", "gpu", "--data-dir", str(data), "--out-dir", str(out)]
                                     + (["--smoke"] if p.fast else []),  # 1 fold, 2 epochs, 60 rows
        "needs": ["train.csv", "test.csv", "folds_calibrated.json"],
    },
}


def build_command(plan: FineTunePlan, repo_dir: Path, data_dir: Path, out_dir: Path) -> list[str]:
    """The 'coder' step: turn a fine-tune plan into the exact GPU training command (template-based)."""
    tpl = TEMPLATES.get(plan.backbone)
    if tpl is None:
        raise ValueError(f"no fine-tune template for backbone {plan.backbone!r}; have {list(TEMPLATES)}")
    script = Path(repo_dir) / tpl["script"]
    return ["python", str(script), *tpl["args"](plan, data_dir, out_dir)]


def collect_results(plan: FineTunePlan, out_dir: Path, plans_root: Path, folds_json: Path,
                    train_csv: Path) -> Path:
    """Integration step: wrap the template's OOF/test CSVs into an aggregator plan dir.

    The fine-tune script writes ``oof_*.csv`` (row_id,SMILES,y_true,y_pred) and ``test_*.csv``
    (Molecule Name,SMILES,pEC50). We re-emit them as ``oof_predictions.csv`` / ``test_predictions.csv``
    with the fold column the aggregator/ridge stack expects.
    """
    tpl = TEMPLATES[plan.backbone]
    oof = pd.read_csv(Path(out_dir) / tpl["oof"]).sort_values("row_id").reset_index(drop=True)
    test = pd.read_csv(Path(out_dir) / tpl["test"])
    folds = json.loads(Path(folds_json).read_text())["assignments"]
    train = pd.read_csv(train_csv)

    plan_dir = Path(plans_root) / plan.plan_id
    plan_dir.mkdir(parents=True, exist_ok=True)
    oof = oof.assign(
        fold=[int(folds[str(i)]) for i in oof["row_id"]],
        **{NAME: [train[NAME].iloc[i] for i in oof["row_id"]]},
    )
    oof[["row_id", "fold", NAME, SMILES, "y_true", "y_pred"]].to_csv(plan_dir / "oof_predictions.csv", index=False)
    test[[SMILES, NAME, "pEC50"]].to_csv(plan_dir / "test_predictions.csv", index=False)
    return plan_dir
