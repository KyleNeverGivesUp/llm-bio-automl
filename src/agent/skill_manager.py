"""Skill-driven LLM manager (architecture B) — the manager REASONS over skills + state.

Unlike the hardcoded run_auto.py sequence, here an LLM manager is given the task, the current
state (pool + best judge RAE), and a registry of skills (each with a description that ENCODES
what we learned manually). Each step the manager decides which skill to invoke next and with
what args; the skill's executor runs the corresponding (already-built) Python. It loops until
the manager calls `finish`. This is the "whole pipeline is LLM-driven" form — the control flow
is the LLM's, not a fixed script.

The skill descriptions carry the hard-won lessons so the manager doesn't re-derive them:
- the performance lever is FINE-TUNING decorrelated foundation models (graph + 3D), not the
  frozen featurizer × sklearn menu (which caps ~0.62 / rank 84);
- stacking DECORRELATED members beats more correlated ones; the Set-1 judge is the reward.

Executors wrap existing code: finetune_designer/finetune_runner, menu_designer/cv_runner,
aggregator, analog_judge, hf_retrieval. No new ML — just LLM-chosen orchestration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.agent.LLM_base import LLMJsonAgent
from src.aggregator import aggregate
from src.analog_judge import judge_csv


@dataclass
class Ctx:
    """Shared state the manager + skills read/write."""
    data_dir: Path
    run_dir: Path
    folds_json: Path
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    folds: object
    collect_only: bool = False                 # fine-tune: reuse predictions/ instead of GPU (Mac testing)
    state: dict = field(default_factory=lambda: {"plans": {}, "best": None, "log": []})


# --- skill executors (wrap existing code) ----------------------------------- #
def _skill_retrieve(ctx: Ctx, args: dict) -> str:
    from src.agent.hf_retrieval import discover_models
    cands = discover_models(top_k=int(args.get("top_k", 12)))
    (ctx.run_dir / "candidates_live.json").write_text(
        json.dumps([c.to_dict() for c in cands], indent=2), encoding="utf-8")
    fams = sorted({c.family for c in cands})
    return f"discovered {len(cands)} models, families={fams}"


def _skill_design_menu(ctx: Ctx, args: dict) -> str:
    from src.agent.menu_designer import MenuDesigner
    from src.cv_runner import run_plan_cv
    n = int(args.get("n", 4))
    prior = [{"plan_id": k, **{x: v.get(x) for x in ("featurizer", "model", "params", "judge_rae")}}
             for k, v in ctx.state["plans"].items()]
    plans = MenuDesigner().propose(n, prior, exclude=set(ctx.state["plans"]), use_llm=True, log_path=None)
    done = 0
    for p in plans:
        try:
            pdir = ctx.run_dir / "plans" / p.plan_id
            run_plan_cv(p, ctx.train_df, ctx.test_df, ctx.folds, out_dir=pdir,
                        cache_dir=ctx.data_dir.parent / "featurizer_cache", refit_full=True)
            jr = judge_csv(pdir / "test_predictions.csv")["rae"]
            ctx.state["plans"][p.plan_id] = {"dir": str(pdir), "judge_rae": jr,
                                             "featurizer": p.featurizer, "model": p.model, "params": p.params}
            done += 1
        except Exception as e:
            ctx.state["plans"].setdefault("_errors", []) if False else None
            print(f"    menu {p.plan_id} failed: {type(e).__name__}")
    return f"ran {done}/{len(plans)} menu plans (frozen featurizers — expect ~0.62, the weak baseline)"


def _skill_finetune(ctx: Ctx, args: dict) -> str:
    import subprocess
    from src.agent.finetune_designer import FineTuneDesigner
    from src.finetune_runner import build_command, collect_results
    repo = ctx.data_dir.parent.parent
    prior = [{"plan_id": k, "judge_rae": v.get("judge_rae")} for k, v in ctx.state["plans"].items()]
    plans = FineTuneDesigner().propose(prior_results=prior)
    done = 0
    for p in plans:
        try:
            out_dir = (repo / "predictions") if ctx.collect_only else (Path("/tmp") / p.plan_id)
            if not ctx.collect_only:
                out_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(build_command(p, repo, ctx.data_dir, out_dir), check=True)
            pdir = collect_results(p, out_dir=out_dir, plans_root=ctx.run_dir / "plans",
                                   folds_json=ctx.folds_json, train_csv=ctx.data_dir / "train.csv")
            jr = judge_csv(pdir / "test_predictions.csv")["rae"]
            ctx.state["plans"][p.plan_id] = {"dir": str(pdir), "judge_rae": jr,
                                             "featurizer": f"finetune:{p.backbone}", "model": "finetune",
                                             "params": {"epochs": p.epochs}}
            done += 1
        except Exception as e:
            print(f"    finetune {p.plan_id} failed: {type(e).__name__}: {str(e)[:80]}")
    return f"fine-tuned {done} decorrelated foundation models (the performance lever)"


def _skill_stack(ctx: Ctx, args: dict) -> str:
    dirs = [Path(p["dir"]) for p in ctx.state["plans"].values()
            if isinstance(p, dict) and p.get("judge_rae") is not None and p["judge_rae"] < 0.95]
    if len(dirs) < 1:
        return "nothing to stack yet"
    aggregate(dirs, ctx.run_dir / "ensemble")
    rae = judge_csv(ctx.run_dir / "ensemble" / "ensemble" / "test_predictions.csv")["rae"]
    prev = ctx.state["best"]["judge_rae"] if ctx.state["best"] else None
    if prev is None or rae < prev:
        ctx.state["best"] = {"judge_rae": rae, "n_members": len(dirs)}
    return f"stacked {len(dirs)} members -> judge RAE {rae:.4f}" + (f" (was {prev:.4f})" if prev else "")


SKILLS = {
    "retrieve": {"executor": _skill_retrieve,
                 "desc": "Live-search HuggingFace + frontier for foundation models, classified by family "
                         "(graph/3D/SMILES). LESSON: the strong, decorrelated families are graph (CheMeleon) "
                         "and 3D (Uni-Mol); SMILES transformers (ChemBERTa) are weak here."},
    "design_menu": {"executor": _skill_design_menu,
                    "desc": "Propose & run frozen featurizer×sklearn plans. LESSON: this is the WEAK baseline "
                            "(caps ~RAE 0.62 / rank 84) — useful only as cheap diverse stack members."},
    "finetune": {"executor": _skill_finetune,
                 "desc": "Fine-tune decorrelated foundation models (graph + 3D) and add to the pool. LESSON: "
                         "THIS is the performance lever (0.62 -> 0.57 / rank 84 -> 20). Always do this."},
    "stack": {"executor": _skill_stack,
              "desc": "Ridge-stack the pool on OOF, judge on Set-1. LESSON: decorrelated members lower RAE; "
                      "correlated ones don't. Run after adding members to measure progress."},
}


class SkillManager(LLMJsonAgent):
    name = "skill_manager"

    def run(self, ctx: Ctx, max_steps: int = 6) -> dict:
        for step in range(1, max_steps + 1):
            d = self._decide(ctx, step, max_steps)
            skill, args, reason = d.get("skill"), d.get("args", {}), d.get("reason", "")
            print(f"[manager step {step}] -> {skill}  ({reason[:80]})")
            if skill == "finish" or skill not in SKILLS:
                ctx.state["log"].append({"step": step, "skill": "finish", "reason": reason})
                break
            result = SKILLS[skill]["executor"](ctx, args or {})
            print(f"           {result}")
            ctx.state["log"].append({"step": step, "skill": skill, "result": result,
                                     "best": ctx.state.get("best")})
        return ctx.state

    def _decide(self, ctx: Ctx, step: int, max_steps: int) -> dict:
        best = ctx.state.get("best")
        pool = [(k, v.get("judge_rae")) for k, v in ctx.state["plans"].items() if isinstance(v, dict)]
        skills_doc = "\n".join(f"  - {n}: {s['desc']}" for n, s in SKILLS.items())
        system = (
            "You are the MANAGER of an AutoML pipeline for molecular pEC50 regression (metric RAE, lower=better). "
            "Each step you pick ONE skill to run next, given the state. The skill descriptions encode hard-won "
            "lessons — follow them. Goal: minimize the stacked Set-1 judge RAE. "
            'Reply ONLY JSON: {"skill": <name|"finish">, "args": {..}, "reason": ".."}.'
        )
        user = (
            f"Step {step}/{max_steps}. Skills:\n{skills_doc}\n\n"
            f"State: pool has {len(pool)} members; best stacked judge RAE = {best['judge_rae'] if best else 'none'}.\n"
            f"Recent log: {json.dumps(ctx.state['log'][-3:])}\n\n"
            "Pick the next skill that will most lower the stacked RAE (remember: fine-tuning decorrelated "
            "foundations is the lever; stack to measure; finish when further gains are unlikely)."
        )
        try:
            out = self.call_json(system, user)
            return out if isinstance(out, dict) else {"skill": "finish", "reason": "bad LLM output"}
        except Exception as e:
            # LLM unavailable (e.g., 429 rate-limit on the free model): degrade gracefully, don't crash.
            # If we haven't fine-tuned yet, do the lever; else stack; else finish with what we have.
            print(f"[manager] LLM decide failed ({type(e).__name__}); heuristic fallback")
            has_ft = any(str(v.get("model")) == "finetune" for v in ctx.state["plans"].values() if isinstance(v, dict))
            if not has_ft:
                return {"skill": "finetune", "reason": "fallback: fine-tune the decorrelated lever"}
            if ctx.state.get("best") is None:
                return {"skill": "stack", "reason": "fallback: stack the pool"}
            return {"skill": "finish", "reason": "fallback: have a stacked result, stop"}
