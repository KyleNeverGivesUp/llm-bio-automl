"""LLM-orchestrated fine-tuning — the designer decides WHAT to fine-tune (prof-approved).

Until now fine-tune plans were typed by hand (`--backbone unimol --epochs 15`). This is the
autonomy step: an LLM, given the task + the fine-tunable backbones (with their representation
families) + what's already been tried, PROPOSES a set of fine-tune plans — picking decorrelated
families to stack, with sensible epochs. The pipeline then executes them (src/finetune_runner.py)
and judges, exactly as for hand-written plans.

Mirrors MenuDesigner: an LLMJsonAgent with a curated deterministic fallback so the loop always
makes progress even if the LLM is down or returns junk.
"""

from __future__ import annotations

import json

from src.agent.LLM_base import LLMJsonAgent
from src.finetune_runner import TEMPLATES, FineTunePlan

# What the designer may choose to fine-tune, with the facts it needs to reason about decorrelation.
BACKBONE_FACTS = {
    "chemeleon": {"family": "graph (D-MPNN)", "good_epochs": 50,
                  "note": "multitask (pEC50+counter+Emax+single_conc) + MAE loss; strongest single (~0.59)"},
    "unimol":    {"family": "3D geometry",    "good_epochs": 15,
                  "note": "kfold=1 has no early stop, so cap epochs ~15 to avoid overfitting; ~0.62 single but decorrelated"},
}

# Validated default if the LLM is unavailable: the two decorrelated families that stacked to 0.5706.
_FALLBACK = [FineTunePlan(backbone="chemeleon", epochs=50, label="ft_chemeleon"),
             FineTunePlan(backbone="unimol", epochs=15, label="ft_unimol")]


class FineTuneDesigner(LLMJsonAgent):
    name = "finetune_designer"

    def run(self, context):  # BaseAgent abstract; the manager calls propose() directly
        raise NotImplementedError("call propose()")

    def propose(self, prior_results: list[dict] | None = None) -> list[FineTunePlan]:
        """LLM picks which backbones to fine-tune + epochs; falls back to the validated combo."""
        try:
            raw = self._llm_propose(prior_results or [])
            plans = [self._to_plan(r) for r in raw]
            plans = [p for p in plans if p is not None]
            if plans:
                return plans
        except Exception as e:  # LLM down / bad JSON / unknown backbone -> validated fallback
            print(f"[finetune_designer] LLM propose failed ({e}); using validated fallback")
        return list(_FALLBACK)

    def _llm_propose(self, prior_results: list[dict]) -> list[dict]:
        backbones = {b: f"{v['family']} — {v['note']}" for b, v in BACKBONE_FACTS.items() if b in TEMPLATES}
        board = "\n".join(
            f"  RAE={r['judge_rae']:.4f}  {r.get('plan_id', r.get('backbone',''))}"
            for r in sorted(prior_results, key=lambda r: r.get("judge_rae", 9))[:10]
            if r.get("judge_rae") is not None
        )
        system = (
            "You are an AutoML designer for molecular pEC50 regression (metric RAE, lower is better). "
            "You decide which pretrained foundation models to FINE-TUNE and stack. The key lever is "
            "DECORRELATION: stacking models from DIFFERENT representation families (graph vs 3D) beats "
            "more of the same. You do NOT write training code — you pick backbones + epochs. "
            'Reply with ONLY JSON: {"finetune_plans":[{"backbone":..,"epochs":..,"label":..}, ...]}.'
        )
        user = (
            f"Fine-tunable backbones (family — notes):\n"
            + "\n".join(f"  {b}: {d}" for b, d in backbones.items())
            + (f"\n\nAlready evaluated (lower RAE better):\n{board}" if board else "")
            + "\n\nPropose a small set of fine-tune plans whose stack will minimize RAE — prefer "
            "DECORRELATED families, use each backbone's suggested epochs. Return ONLY the JSON object."
        )
        out = self.call_json(system, user)
        plans = out.get("finetune_plans") if isinstance(out, dict) else None
        if not isinstance(plans, list):
            raise ValueError("LLM did not return a 'finetune_plans' list")
        return plans

    def _to_plan(self, raw: dict) -> FineTunePlan | None:
        bb = str(raw.get("backbone", "")).lower()
        if bb not in TEMPLATES:
            return None  # only known/verified templates
        epochs = int(raw.get("epochs") or BACKBONE_FACTS.get(bb, {}).get("good_epochs", 50))
        return FineTunePlan(backbone=bb, epochs=epochs, tta=int(raw.get("tta", 0)),
                            label=raw.get("label") or f"ft_{bb}")
