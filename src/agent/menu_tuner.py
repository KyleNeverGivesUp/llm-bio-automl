"""M3 Tuner — the LLM proposes hyperparameters, judged in the loop.

The Designer chooses *which* (featurizer, model) to try; the Tuner refines the
*hyperparameters* of a chosen one. Both optimize the **Set-1 judge** (not
scaffold-CV — our earlier tuning used scaffold-CV, the miscalibrated metric, and
wrongly concluded "defaults are optimal").

Given a base config and the param sets already tried (with their judge RAE), the
LLM proposes N new hyperparameter sets to lower the RAE. Same safety net as the
Designer: a **deterministic perturbation fallback** so the loop never stalls, and
the model wrappers simply ignore any hyperparameter they don't use, so a malformed
proposal can't break execution.
"""

from __future__ import annotations

import hashlib
import json

from src.agent.LLM_base import LLMJsonAgent
from src.menu_config import MODEL_DEFAULTS, SCALED_MODELS, STOCHASTIC_MODELS
from src.models import available_models
from src.schemas import MenuPlan

# Numeric knobs we know how to perturb deterministically per model (for the fallback).
_NUMERIC_KNOBS = {
    "lightgbm": ["n_estimators", "num_leaves", "learning_rate", "subsample", "colsample_bytree", "reg_lambda"],
    "xgboost": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_lambda"],
    "catboost": ["n_estimators", "max_depth", "learning_rate", "reg_lambda"],
    "random_forest": ["n_estimators", "max_depth", "min_samples_leaf"],
    "ridge": ["alpha"],
    "elastic_net": ["alpha", "l1_ratio"],
    "mlp_head": ["mlp_alpha", "learning_rate_init"],
}
# Multiplier schedule applied per proposal index (varies the search without RNG).
_MULT = [0.5, 2.0, 0.7, 1.5, 0.33, 3.0, 0.85, 1.25]


class MenuTuner(LLMJsonAgent):
    name = "menu_tuner"

    def run(self, context):
        raise NotImplementedError("MenuTuner is driven by propose(), not run().")

    def propose(self, featurizer: str, model: str, base_params: dict,
                prior_trials: list[dict], n: int, exclude: set[str],
                use_llm: bool = True, log_path=None) -> list[MenuPlan]:
        """Up to ``n`` NEW hyperparameter variants of (featurizer, model) not in ``exclude``.

        ``prior_trials`` = [{params, judge_rae}, ...] for this same base config.
        """
        if model not in available_models():
            return []
        plans: list[MenuPlan] = []
        seen = set(exclude)

        if use_llm:
            try:
                for params in self._llm_propose(featurizer, model, base_params, prior_trials, n, log_path):
                    plan = self._to_plan(featurizer, model, base_params, params)
                    if plan and plan.plan_id not in seen:
                        plans.append(plan)
                        seen.add(plan.plan_id)
            except Exception as e:
                print(f"  [tuner] LLM proposal failed ({type(e).__name__}: {str(e)[:100]}); using fallback")

        for params in self._fallback(model, base_params, n):
            if len(plans) >= n:
                break
            plan = self._to_plan(featurizer, model, base_params, params)
            if plan and plan.plan_id not in seen:
                plans.append(plan)
                seen.add(plan.plan_id)
        return plans[:n]

    # ------------------------------------------------------------------ #
    def _llm_propose(self, featurizer, model, base_params, prior_trials, n, log_path) -> list[dict]:
        ranked = sorted([t for t in prior_trials if t.get("judge_rae") is not None],
                        key=lambda t: t["judge_rae"])[:12]
        history = "\n".join(f"  RAE={t['judge_rae']:.4f}  {json.dumps(t['params'])[:120]}" for t in ranked)
        defaults = MODEL_DEFAULTS.get(model, {})
        system = (
            "You tune hyperparameters for a molecular pEC50 regressor (metric RAE, lower is better). "
            "Reply ONLY with JSON: {\"params\":[{<hyperparameters>}, ...]} — each entry a complete "
            "hyperparameter set to try."
        )
        user = (
            f"Model: {model}  (featurizer fixed: {featurizer})\n"
            f"Default hyperparameters: {json.dumps(defaults)}\n"
            f"Tunable knobs: {_NUMERIC_KNOBS.get(model, [])}\n"
            f"Trials so far on the held-out analog judge (lower RAE better):\n{history or '  (only defaults)'}\n\n"
            f"Propose {n} NEW hyperparameter sets likely to lower RAE (sensible ranges; vary depth, "
            "estimators, learning rate, regularization). Return ONLY the JSON object."
        )
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                out = self.call_json(system, user)
                log_path.write_text(json.dumps({"system": system, "user": user, "parsed": out}, indent=2), encoding="utf-8")
            except Exception as e:
                log_path.write_text(json.dumps({"system": system, "user": user, "error": str(e)}, indent=2), encoding="utf-8")
                raise
        else:
            out = self.call_json(system, user)
        params = out.get("params") if isinstance(out, dict) else None
        if not isinstance(params, list):
            raise ValueError("LLM did not return a 'params' list")
        return [p for p in params if isinstance(p, dict)]

    def _fallback(self, model, base_params, n) -> list[dict]:
        """Perturb the base numeric knobs deterministically (no RNG — index-driven)."""
        knobs = _NUMERIC_KNOBS.get(model, [])
        base = {**MODEL_DEFAULTS.get(model, {}), **base_params}
        out = []
        for i in range(n * 2):  # generate extra; caller dedupes/trims
            knob = knobs[i % len(knobs)] if knobs else None
            if knob is None:
                break
            mult = _MULT[i % len(_MULT)]
            cur = base.get(knob)
            if not isinstance(cur, (int, float)) or isinstance(cur, bool):
                continue
            new_val = max(1, int(round(cur * mult))) if knob in ("n_estimators", "num_leaves", "max_depth", "min_samples_leaf") else round(cur * mult, 5)
            out.append({**base_params, knob: new_val})
        return out

    # ------------------------------------------------------------------ #
    def _to_plan(self, featurizer, model, base_params, params) -> MenuPlan | None:
        if not isinstance(params, dict):
            return None
        # Keep the featurizer-side keys (components/skill_ref/scale) from the base;
        # layer the proposed model hyperparameters on top of the model defaults.
        carry = {k: base_params[k] for k in ("components", "skill_ref") if k in base_params}
        merged = {**MODEL_DEFAULTS.get(model, {}), **carry, **params}
        if model in SCALED_MODELS and "scale" in base_params:
            merged["scale"] = base_params["scale"]
        seeds = [42, 1] if model in STOCHASTIC_MODELS else [42]
        sig = hashlib.sha1(json.dumps({"f": featurizer, "m": model, "p": merged}, sort_keys=True, default=str).encode()).hexdigest()[:6]
        comp = merged.get("components")
        comp_tag = ("+" + "+".join(c[:4] for c in comp)) if comp else ""
        plan_id = f"tune__{featurizer}{comp_tag}__{model}__{sig}"
        return MenuPlan(plan_id=plan_id, name=plan_id, featurizer=featurizer, model=model,
                        params=merged, seeds=seeds, skill_ref=merged.get("skill_ref"))
