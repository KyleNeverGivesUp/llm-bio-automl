"""M3 Designer — the LLM proposes new menu candidates, judged in the loop.

This is the real Designer the PRD called for (the legacy ``designer_agent.py`` was a
stub). It does NOT generate code (that is Approach 2/M4) — it *composes from the
menu*: it picks (featurizer, model, hyperparameters) combinations from the existing
registries. The dynamism is in the **choices**, informed each round by how prior
candidates scored on the **Set-1 judge**.

Two guarantees keep it safe:
  - **Validated by construction.** Every proposal is checked against the featurizer
    and model registries; anything unknown is dropped. So the LLM can't break the
    deterministic execution layer — at worst it proposes nothing useful.
  - **Deterministic fallback.** If the LLM is unavailable or returns junk, a curated
    pool of untried-but-sensible combinations fills the round. The loop never stalls.
"""

from __future__ import annotations

import hashlib
import json

from src.agent.LLM_base import LLMJsonAgent
from src.menu_config import (
    CHEMBERTA,
    CHEMBERTA_100M,
    MODEL_DEFAULTS,
    SCALED_MODELS,
    STOCHASTIC_MODELS,
)
from src.featurizers import available_featurizers
from src.models import available_models
from src.schemas import MenuPlan

BINARY_FEATURIZERS = {"morgan", "maccs", "avalon"}
EMBED_FEATURIZERS = {"chemberta_embedding", "molformer_embedding"}
MOLFORMER = "ibm-research/MoLFormer-XL-both-10pct"
# molformer is dropped by default (weak on analogs); the Designer may still try it.
DEFAULT_SKILL_REF = {"chemberta_embedding": CHEMBERTA, "molformer_embedding": MOLFORMER}
# The LLM tends to echo the short label ("chemberta") instead of the HF model id;
# normalise so a proposal can't point the featurizer at a non-existent model.
VALID_SKILLS = {CHEMBERTA, CHEMBERTA_100M, MOLFORMER}
SKILL_ALIASES = {
    "chemberta": CHEMBERTA, "chemberta-77m": CHEMBERTA, "chemberta_77m": CHEMBERTA, "chemberta77m": CHEMBERTA,
    "chemberta100m": CHEMBERTA_100M, "chemberta-100m": CHEMBERTA_100M, "chemberta_100m": CHEMBERTA_100M,
    "molformer": MOLFORMER, "molformer-xl": MOLFORMER,
}


def _norm_skill(skill_ref, featurizer: str) -> str:
    """Map a proposed skill_ref to a real HF model id (default if unrecognized)."""
    if skill_ref in VALID_SKILLS:
        return skill_ref
    if isinstance(skill_ref, str) and skill_ref.lower() in SKILL_ALIASES:
        return SKILL_ALIASES[skill_ref.lower()]
    return DEFAULT_SKILL_REF.get(featurizer, CHEMBERTA)

# Untried-but-sensible combinations for the deterministic fallback — things the
# frozen menu did NOT include: cross fusions, non-default fingerprint pairings,
# and a couple of hyperparameter variants. (rep components, model, param overrides.)
_FALLBACK = [
    ({"featurizer": "fusion", "components": ["rdkit_descriptors", "morgan"]}, "lightgbm", {}),
    ({"featurizer": "fusion", "components": ["rdkit_descriptors", "maccs"]}, "xgboost", {}),
    ({"featurizer": "fusion", "components": ["chemberta_embedding", "morgan"], "skill_ref": CHEMBERTA}, "lightgbm", {}),
    ({"featurizer": "fusion", "components": ["rdkit_descriptors", "chemberta_embedding", "maccs"], "skill_ref": CHEMBERTA}, "lightgbm", {}),
    ({"featurizer": "rdkit_descriptors"}, "lightgbm", {"n_estimators": 1200, "learning_rate": 0.02, "num_leaves": 63}),
    ({"featurizer": "rdkit_descriptors"}, "xgboost", {"n_estimators": 1000, "max_depth": 4, "learning_rate": 0.03}),
    ({"featurizer": "fusion", "components": ["rdkit_descriptors", "chemberta_embedding"], "skill_ref": CHEMBERTA}, "catboost", {"n_estimators": 1500, "learning_rate": 0.03}),
    ({"featurizer": "chemberta_embedding", "skill_ref": CHEMBERTA_100M}, "lightgbm", {}),
    ({"featurizer": "morgan"}, "catboost", {}),
    ({"featurizer": "rdkit_descriptors"}, "random_forest", {"n_estimators": 800, "max_features": 0.3}),
    ({"featurizer": "fusion", "components": ["rdkit_descriptors", "chemberta_embedding"], "skill_ref": CHEMBERTA}, "mlp_head", {"hidden_layer_sizes": [512, 128]}),
    ({"featurizer": "avalon", "n_bits": 2048}, "lightgbm", {}),
]


class MenuDesigner(LLMJsonAgent):
    name = "menu_designer"

    def run(self, context):  # BaseAgent abstract method; the Manager calls propose() directly
        raise NotImplementedError("MenuDesigner is driven by propose(), not run().")

    # ------------------------------------------------------------------ #
    def propose(self, n: int, prior_results: list[dict], exclude: set[str],
                use_llm: bool = True, log_path=None) -> list[MenuPlan]:
        """Return up to ``n`` NEW, validated MenuPlans not already in ``exclude``.

        ``prior_results`` = [{plan_id, featurizer, model, params, judge_rae}, ...]
        from earlier rounds, given to the LLM so it can iterate toward lower RAE.
        """
        plans: list[MenuPlan] = []
        seen = set(exclude)

        if use_llm:
            try:
                for raw in self._llm_propose(n, prior_results, log_path):
                    plan = self._to_plan(raw)
                    if plan and plan.plan_id not in seen:
                        plans.append(plan)
                        seen.add(plan.plan_id)
            except Exception as e:  # any LLM/parse failure -> fall through to fallback
                print(f"  [designer] LLM proposal failed ({type(e).__name__}: {str(e)[:120]}); using fallback")

        # Top up from the deterministic pool (and guarantee progress if the LLM gave nothing).
        for spec, model, params in _FALLBACK:
            if len(plans) >= n:
                break
            plan = self._to_plan({**spec, "model": model, "params": params})
            if plan and plan.plan_id not in seen:
                plans.append(plan)
                seen.add(plan.plan_id)
        return plans[:n]

    # ------------------------------------------------------------------ #
    def _llm_propose(self, n: int, prior_results: list[dict], log_path) -> list[dict]:
        # molformer is excluded: verified weak on analogs (RAE~1.0, dropped) and its
        # custom-code model is brittle inside a long loop. Keep the search on the
        # representations that actually help.
        feats = [f for f in available_featurizers() if f != "molformer_embedding"] + ["fusion"]
        models = available_models()
        # Compact leaderboard so the prompt stays small but informative.
        ranked = sorted(prior_results, key=lambda r: r.get("judge_rae", 9))[:18]
        board = "\n".join(
            f"  RAE={r['judge_rae']:.4f}  {r['featurizer']}+{r['model']} {json.dumps(r.get('params', {}))[:80]}"
            for r in ranked if r.get("judge_rae") is not None
        )
        system = (
            "You are an AutoML designer for molecular pEC50 regression (metric RAE, lower is better). "
            "You compose candidates from a fixed menu — you do NOT write code. "
            "Reply with ONLY a JSON object: {\"plans\":[{\"featurizer\":..,\"model\":..,\"params\":{..},"
            "\"components\":[..]?,\"skill_ref\":..?}, ...]}."
        )
        user = (
            f"Featurizers: {feats}\n"
            f"Models: {models}\n"
            "Notes: 'fusion' concatenates featurizers listed in params.components (e.g. "
            "['rdkit_descriptors','chemberta_embedding']). Embedding featurizers need skill_ref. "
            "Binary fingerprints are morgan/maccs/avalon. params may set model hyperparameters "
            "(n_estimators, max_depth, learning_rate, num_leaves, hidden_layer_sizes, ...).\n\n"
            f"Candidates already evaluated on the held-out analog judge (lower RAE is better):\n{board or '  (none yet)'}\n\n"
            f"Propose {n} NEW, DIVERSE candidates likely to lower the ensemble RAE or add useful "
            "diversity (different representations/models, sensible hyperparameters). Avoid duplicates "
            "of the list above. Return ONLY the JSON object."
        )
        if log_path is not None:
            out = self._call_json_to_path(system, user, log_path)
        else:
            out = self.call_json(system, user)
        plans = out.get("plans") if isinstance(out, dict) else None
        if not isinstance(plans, list):
            raise ValueError("LLM did not return a 'plans' list")
        return plans

    def _call_json_to_path(self, system, user, log_path):
        """call_json but persist the raw exchange (lightweight LLM log)."""
        try:
            out = self.call_json(system, user)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps({"system": system, "user": user, "parsed": out}, indent=2), encoding="utf-8")
            return out
        except Exception as e:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps({"system": system, "user": user, "error": str(e)}, indent=2), encoding="utf-8")
            raise

    # ------------------------------------------------------------------ #
    def _to_plan(self, raw: dict) -> MenuPlan | None:
        """Validate a raw proposal into a MenuPlan, or None if it's not buildable."""
        if not isinstance(raw, dict):
            return None
        feat = raw.get("featurizer")
        model = raw.get("model")
        if feat not in (set(available_featurizers()) | {"fusion"}) or model not in available_models():
            return None
        # molformer dropped by verdict (weak + brittle in-loop); reject standalone use.
        if feat == "molformer_embedding":
            return None

        params = dict(raw.get("params") or {})
        # Carry through fusion components / embedding skill_ref from either location.
        components = raw.get("components") or params.get("components")
        skill_ref = raw.get("skill_ref") or params.get("skill_ref")

        if feat == "fusion":
            if not isinstance(components, list) or not components:
                return None
            # Drop molformer + dedupe; the fusion featurizer shares ONE skill_ref across
            # components, so it can hold at most one embedding model.
            components = [c for c in dict.fromkeys(components)
                          if c in available_featurizers() and c != "molformer_embedding"]
            if len(components) < 2:
                return None
            if sum(c in EMBED_FEATURIZERS for c in components) > 1:
                return None  # can't give two embeddings different skill_refs
            params["components"] = components
            dense = True  # fusion always contains a dense block here
            embed = next((c for c in components if c in EMBED_FEATURIZERS), None)
            if embed:
                params["skill_ref"] = _norm_skill(skill_ref, embed)
        else:
            dense = feat not in BINARY_FEATURIZERS
            if feat in EMBED_FEATURIZERS:
                params["skill_ref"] = _norm_skill(skill_ref, feat)

        # Layer model defaults under any explicit overrides; set scaling for scale-sensitive models.
        merged = {**MODEL_DEFAULTS.get(model, {}), **params}
        if model in SCALED_MODELS:
            merged["scale"] = dense
        seeds = [42, 1] if model in STOCHASTIC_MODELS else [42]

        sig = hashlib.sha1(json.dumps({"f": feat, "m": model, "p": merged}, sort_keys=True, default=str).encode()).hexdigest()[:6]
        comp_tag = ("+" + "+".join(c[:4] for c in components)) if feat == "fusion" else ""
        plan_id = f"auto__{feat}{comp_tag}__{model}__{sig}"
        return MenuPlan(plan_id=plan_id, name=plan_id, featurizer=feat, model=model,
                        params=merged, seeds=seeds, skill_ref=merged.get("skill_ref"))
