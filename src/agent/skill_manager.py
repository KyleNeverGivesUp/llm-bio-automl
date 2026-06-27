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
    brief_path: Path | None = None             # competition description the Setup agent reads
    collect_only: bool = False                 # fine-tune: reuse predictions/ instead of GPU (Mac testing)
    state: dict = field(default_factory=lambda: {"plans": {}, "best": None, "log": [], "setup": None})


# --- skill executors (wrap existing code) ----------------------------------- #
def _skill_setup(ctx: Ctx, args: dict) -> tuple[str, str]:
    from src.agent.setup_agent import SetupAgent
    brief = (Path(ctx.brief_path).read_text(encoding="utf-8")
             if ctx.brief_path and Path(ctx.brief_path).exists()
             else "Predict pEC50 (metric RAE) from a molecule's SMILES string.")
    agent = SetupAgent()
    report = agent.run(instruction=brief, data_dir=ctx.data_dir, out_path=ctx.run_dir / "setup_report.json")
    ctx.state["setup"] = report                      # downstream reads schema/metric from here
    s = report.get("schema", {})
    return (f"task={report.get('task', {}).get('type')} metric={report.get('metric')} "
            f"schema(smiles={s.get('smiles_col')}, target={s.get('target_col')}) status={report.get('status')}"), agent.source


def _skill_retrieve(ctx: Ctx, args: dict) -> tuple[str, str]:
    from src.agent.retrieval_agent import RetrievalAgent
    agent = RetrievalAgent()
    result = agent.run(ctx.state.get("setup") or {}, top_k=int(args.get("top_k", 12)),
                       out_path=ctx.run_dir / "retrieval_result.json")
    ctx.state["candidates"] = result["selected"]          # downstream run reads these
    picks = ", ".join(f"{s.get('ref', '?').split('/')[-1]}:{s.get('mode')}" for s in result["selected"][:5])
    return f"searched {result['n_candidates']} models (task-guided) -> selected: {picks}", agent.source


def _skill_design_menu(ctx: Ctx, args: dict) -> tuple[str, str]:
    from src.agent.menu_designer import MenuDesigner
    from src.cv_runner import run_plan_cv
    n = int(args.get("n", 4))
    prior = [{"plan_id": k, **{x: v.get(x) for x in ("featurizer", "model", "params", "judge_rae")}}
             for k, v in ctx.state["plans"].items() if isinstance(v, dict)]
    designer = MenuDesigner()
    plans = designer.propose(n, prior, exclude=set(ctx.state["plans"]), use_llm=True, log_path=None)
    src = getattr(designer, "source", "llm")
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
    return f"ran {done}/{len(plans)} menu plans (frozen featurizers — the weak baseline ~0.62)", src


# Fine-tune epochs per templated backbone (from finetune_designer.BACKBONE_FACTS).
_FT_EPOCHS = {"chemeleon": 50, "unimol": 15}
# Validated decorrelated combo used when retrieval produced no candidates.
_RUN_FALLBACK = [{"ref": "chemeleon", "family": "graph", "mode": "finetune"},
                 {"ref": "unimol", "family": "3d", "mode": "finetune"}]


def _frozen_featurizer(base: str, family: str, ref: str) -> tuple[str | None, dict]:
    """Map a candidate to a frozen-embedding featurizer (None = can't run frozen here)."""
    if base == "chemeleon":
        return "chemeleon_embedding", {}
    if family == "smiles":
        return "chemberta_embedding", {"skill_ref": ref if "/" in ref else "DeepChem/ChemBERTa-77M-MTR"}
    if family == "descriptor":
        return "rdkit_descriptors", {}
    return None, {}                                   # 3d/multiview frozen: no featurizer wired -> skip


def _skill_run(ctx: Ctx, args: dict) -> tuple[str, str]:
    """Execute the retrieval-selected candidates by their mode: finetune (template) or frozen (featurizer).

    This replaces the old hardcoded {chemeleon, unimol} finetune skill — the models + modes now come
    from `ctx.state['candidates']` (retrieval). Falls back to the validated decorrelated combo if empty.
    """
    import subprocess
    from src.cv_runner import run_plan_cv
    from src.finetune_runner import TEMPLATES as FT_TEMPLATES, FineTunePlan, build_command, collect_results
    from src.schemas import MenuPlan
    repo = ctx.data_dir.parent.parent
    cands = ctx.state.get("candidates") or _RUN_FALLBACK
    ran, skipped = 0, []
    for c in cands:
        ref = str(c.get("ref", "")); base = ref.split("/")[-1].lower()
        family, mode = c.get("family", ""), c.get("mode", "frozen")
        pid = f"{mode}_{base}"
        if pid in ctx.state["plans"]:
            continue
        try:
            if mode == "finetune" and base in FT_TEMPLATES:
                p = FineTunePlan(backbone=base, epochs=_FT_EPOCHS.get(base, 50), label=f"ft_{base}")
                out_dir = (repo / "predictions") if ctx.collect_only else (Path("/tmp") / p.plan_id)
                if not ctx.collect_only:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    subprocess.run(build_command(p, repo, ctx.data_dir, out_dir), check=True)
                pdir = collect_results(p, out_dir=out_dir, plans_root=ctx.run_dir / "plans",
                                       folds_json=ctx.folds_json, train_csv=ctx.data_dir / "train.csv")
            else:                                       # frozen: embedding featurizer + sklearn head
                feat, params = _frozen_featurizer(base, family, ref)
                if feat is None:
                    skipped.append(f"{base}({mode}): no template/featurizer")
                    continue
                p = MenuPlan(plan_id=pid, name=pid, featurizer=feat, model="ridge", params=params)
                pdir = ctx.run_dir / "plans" / pid
                run_plan_cv(p, ctx.train_df, ctx.test_df, ctx.folds, out_dir=pdir,
                            cache_dir=ctx.data_dir.parent / "featurizer_cache", refit_full=True)
            jr = judge_csv(pdir / "test_predictions.csv")["rae"]
            ctx.state["plans"][pid] = {"dir": str(pdir), "judge_rae": jr,
                                       "featurizer": f"{mode}:{base}", "model": mode, "params": {}}
            ran += 1
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{base}: {type(e).__name__}")
    verb = "ran (collect-only)" if ctx.collect_only else "ran"
    return f"{verb} {ran} candidate(s)" + (f"; skipped {skipped}" if skipped else ""), "deterministic"


def _skill_stack(ctx: Ctx, args: dict) -> tuple[str, str]:
    """Ridge-stack the pool, with judge-aware backward elimination (drop members that hurt the judge).

    The LLM's selection can over-include weak members (e.g. a frozen SMILES model). Backward
    elimination greedily drops the member whose removal most lowers the Set-1 judge RAE, until no
    drop helps — so the autonomous ensemble self-prunes to the truly-helpful subset. Members are
    already run (predictions cached), so each re-stack is just a ridge refit + judge (milliseconds).
    """
    members = [(k, Path(v["dir"])) for k, v in ctx.state["plans"].items()
               if isinstance(v, dict) and v.get("judge_rae") is not None and v["judge_rae"] < 0.95]
    if len(members) < 1:
        return "nothing to stack yet", "deterministic"
    out_root = ctx.run_dir / "stacks"

    def stack_rae(subset: list, tag: str) -> float:
        d = out_root / f"ens_{tag}"
        aggregate([p for _, p in subset], d)
        return judge_csv(d / "ensemble" / "test_predictions.csv")["rae"]

    cur = list(members)
    best_rae = stack_rae(cur, "all")
    full_rae, full_n = best_rae, len(cur)
    dropped: list[str] = []
    improved = True
    while improved and len(cur) > 1:                 # keep >=1; in practice stops at the decorrelated core
        improved = False
        drop_i, drop_rae = None, best_rae
        for i in range(len(cur)):
            r = stack_rae(cur[:i] + cur[i + 1:], f"d{len(dropped)}_{i}")
            if r < drop_rae - 1e-6:                  # removing member i improves the judge
                drop_rae, drop_i = r, i
        if drop_i is not None:
            dropped.append(cur.pop(drop_i)[0]); best_rae = drop_rae; improved = True

    aggregate([p for _, p in cur], ctx.run_dir / "ensemble")   # final ensemble = surviving subset
    rae = judge_csv(ctx.run_dir / "ensemble" / "ensemble" / "test_predictions.csv")["rae"]
    prev = ctx.state["best"]["judge_rae"] if ctx.state["best"] else None
    if prev is None or rae < prev:
        ctx.state["best"] = {"judge_rae": rae, "n_members": len(cur), "members": [k for k, _ in cur]}
    msg = (f"stacked {full_n}->{len(cur)} members -> judge RAE {rae:.4f}"
           + (f" (pruned {dropped}; full was {full_rae:.4f})" if dropped else ""))
    return msg, "deterministic"


SKILLS = {
    "setup": {"executor": _skill_setup,
              "desc": "Read the competition brief + data dir → infer task schema (smiles/target cols), "
                      "metric, train/test files; deterministic validation + leak guard. Runs FIRST."},
    "retrieve": {"executor": _skill_retrieve,
                 "desc": "TASK-GUIDED model search: an LLM reads the setup task and plans what to search "
                         "(families + HF terms), runs a live HF search, and ranks candidates with a mode "
                         "(frozen=any model / finetune=templated only). LESSON: the strong, decorrelated "
                         "families are graph (CheMeleon) + 3D (Uni-Mol); SMILES transformers are weak here."},
    "design_menu": {"executor": _skill_design_menu,
                    "desc": "Propose & run frozen featurizer×sklearn plans. LESSON: this is the WEAK baseline "
                            "(caps ~RAE 0.62 / rank 84) — useful only as cheap diverse stack members."},
    "run": {"executor": _skill_run,
            "desc": "Execute the retrieval-selected candidates by their mode: finetune (templated graph/3D "
                    "foundations — the performance lever, 0.62->0.57) or frozen (embedding + sklearn head). "
                    "Reads ctx.state['candidates']; falls back to the validated graph+3D finetune combo. "
                    "Run this after retrieve to add members to the pool."},
    "stack": {"executor": _skill_stack,
              "desc": "Ridge-stack the pool on OOF, judge on Set-1. LESSON: decorrelated members lower RAE; "
                      "correlated ones don't. Run after adding members to measure progress."},
}


class SkillManager(LLMJsonAgent):
    name = "skill_manager"

    def run(self, ctx: Ctx, max_steps: int = 6) -> dict:
        log_path = ctx.run_dir / "stage_log.jsonl"

        def record(entry: dict) -> None:
            ctx.state["log"].append(entry)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")

        # Step 0 — Setup ALWAYS runs first (mirrors AIBuildAI: Setup precedes the loop).
        result, ssrc = SKILLS["setup"]["executor"](ctx, {})
        print(f"[step 0] setup  | skill-source={ssrc}\n           {result}")
        record({"step": 0, "skill": "setup", "manager_decision": "forced-first", "skill_source": ssrc, "result": result})
        if (ctx.state.get("setup") or {}).get("status") != "ok":
            print("[manager] setup did not validate — aborting before the loop")
            return ctx.state

        for step in range(1, max_steps + 1):
            d = self._decide(ctx, step, max_steps)
            skill, args, reason, dsrc = d.get("skill"), d.get("args", {}), d.get("reason", ""), d.get("_source", "llm")
            if skill == "finish" or skill not in SKILLS:
                print(f"[step {step}] FINISH  (manager-decision={dsrc})  {reason[:70]}")
                record({"step": step, "skill": "finish", "manager_decision": dsrc, "reason": reason})
                break
            result, ssrc = SKILLS[skill]["executor"](ctx, args or {})
            print(f"[step {step}] {skill}  | manager-decision={dsrc} | skill-source={ssrc}\n           {result}")
            record({"step": step, "skill": skill, "manager_decision": dsrc, "skill_source": ssrc,
                    "result": result, "best": ctx.state.get("best")})
        return ctx.state

    def _decide(self, ctx: Ctx, step: int, max_steps: int) -> dict:
        best = ctx.state.get("best")
        pool = [(k, v.get("judge_rae")) for k, v in ctx.state["plans"].items() if isinstance(v, dict)]
        skills_doc = "\n".join(f"  - {n}: {s['desc']}" for n, s in SKILLS.items() if n != "setup")  # setup already ran
        setup = ctx.state.get("setup") or {}
        task_line = (f"Task (from setup): type={setup.get('task', {}).get('type')}, "
                     f"metric={setup.get('metric')}, target={setup.get('schema', {}).get('target_col')}.")
        system = (
            "You are the MANAGER of an AutoML pipeline for molecular pEC50 regression (metric RAE, lower=better). "
            "Each step you pick ONE skill to run next, given the state. The skill descriptions encode hard-won "
            "lessons — follow them. Goal: minimize the stacked Set-1 judge RAE. "
            'Reply ONLY JSON: {"skill": <name|"finish">, "args": {..}, "reason": ".."}.'
        )
        user = (
            f"Step {step}/{max_steps}. {task_line}\nSkills:\n{skills_doc}\n\n"
            f"State: pool has {len(pool)} members; best stacked judge RAE = {best['judge_rae'] if best else 'none'}.\n"
            f"Recent log: {json.dumps(ctx.state['log'][-3:])}\n\n"
            "Pick the next skill that will most lower the stacked RAE (remember: fine-tuning decorrelated "
            "foundations is the lever; stack to measure; finish when further gains are unlikely)."
        )
        try:
            out = self.call_json(system, user)
            if isinstance(out, dict):
                out["_source"] = "llm"
                return out
            return {"skill": "finish", "reason": "bad LLM output", "_source": "fallback"}
        except Exception as e:
            # LLM unavailable (e.g., 429 rate-limit on the free model): degrade gracefully, don't crash.
            # If we haven't fine-tuned yet, do the lever; else stack; else finish with what we have.
            print(f"[manager] LLM decide failed ({type(e).__name__}); heuristic fallback")
            has_members = any(v.get("judge_rae") is not None for v in ctx.state["plans"].values() if isinstance(v, dict))
            if not has_members:
                return {"skill": "run", "reason": "fallback: run the decorrelated lever", "_source": "fallback"}
            if ctx.state.get("best") is None:
                return {"skill": "stack", "reason": "fallback: stack the pool", "_source": "fallback"}
            return {"skill": "finish", "reason": "fallback: have a stacked result, stop", "_source": "fallback"}
