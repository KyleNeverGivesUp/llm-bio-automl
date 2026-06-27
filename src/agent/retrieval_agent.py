"""Retrieval Agent (LLM-driven, task-guided, local-first + self-growing) — the model-search node.

After Setup tells us the task, this agent searches for suitable models *on purpose*, local-first:
  ① LLM PLANS the search from the task — which representation families + HF search terms suit it;
  ② search the LOCAL library first (`skills/models/registry.json`);
  ③ LLM JUDGES satisfaction — is the local library enough, or should we search online?
  ④ if not satisfied → live HuggingFace search, then WRITE-BACK new finds to the local library
     (so the next similar task is served locally — the library grows / "learns");
  ⑤ LLM RANKS the candidates for THIS task and recommends a mode — frozen (any model) or
     finetune (only models with a verified template; code enforces this).

Principle: every JUDGMENT is the LLM's (plan / satisfaction / rank), each with a deterministic
fallback; the EXECUTION (search the registry, HF search, write-back) is plain code.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agent import model_registry
from src.agent.LLM_base import LLMJsonAgent
from src.agent.hf_retrieval import DEFAULT_QUERIES, discover_models

# Models we have a verified FINE-TUNE template for. Anything else is frozen-only.
TEMPLATED = {"chemeleon": "graph", "unimol": "3d"}


class RetrievalAgent(LLMJsonAgent):
    name = "retrieval"
    source: str = "unknown"

    def run(self, setup_report: dict, top_k: int = 12, out_path: str | Path | None = None) -> dict:
        task = self._task_brief(setup_report)
        plan = self._plan(task)                                       # ① LLM: families + queries
        families = plan.get("families") or []

        local_hits = model_registry.search(families)                 # ② local-first
        sat = self._judge_satisfaction(task, families, local_hits)   # ③ LLM: satisfied?

        went_online, n_added = False, 0
        if not sat.get("satisfied"):
            # task-specific LLM queries + generic sweep, so a narrow query set still yields finds to cache
            queries = list(dict.fromkeys((plan.get("queries") or []) + list(DEFAULT_QUERIES)))
            hf = discover_models(queries=queries, top_k=top_k)
            n_added = model_registry.add([self._to_entry(c) for c in hf])   # ④ write-back (grow library)
            went_online = True
            local_hits = model_registry.search(families) or model_registry.load()

        candidates = [self._mark(m) for m in local_hits]
        ranked = self._rank(task, candidates, plan)                  # ⑤ LLM: select + mode

        result = {"task": task, "search_plan": plan, "satisfaction": sat,
                  "went_online": went_online, "n_added_to_library": n_added,
                  "n_candidates": len(candidates), "selected": ranked, "source": self.source}
        if out_path:
            Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _task_brief(self, setup_report: dict) -> dict:
        t = (setup_report or {}).get("task", {})
        return {"type": t.get("type"), "domain": t.get("domain"), "summary": t.get("summary"),
                "metric": (setup_report or {}).get("metric"),
                "target": (setup_report or {}).get("schema", {}).get("target_col")}

    # ① LLM plans WHAT to search for ----------------------------------------------------------- #
    def _plan(self, task: dict) -> dict:
        system = (
            "You plan a pretrained-model search for an AutoML pipeline. Given the task, decide what KINDS "
            "of models suit it and what to search for on HuggingFace. For families, use ONLY values from "
            "this fixed vocabulary: graph, 3d, smiles, descriptor, multiview (these match the local library "
            'tags). Reply ONLY JSON: {"queries":[..4-7 HF terms..],"families":[..from the vocabulary..],"rationale":..}'
        )
        user = (f"Task: {json.dumps(task)}\n\nPropose HF search queries and the representation families "
                "(from the fixed vocabulary: graph, 3d, smiles, descriptor, multiview) worth trying for THIS "
                "task. Prefer DECORRELATED families (they stack better). Return ONLY JSON.")
        try:
            out = self.call_json(system, user)
            if isinstance(out, dict) and out.get("queries"):
                self.source = "llm"
                return out
        except Exception as e:  # noqa: BLE001
            print(f"[retrieval] search-plan LLM failed ({e}); default sweep")
        self.source = "fallback"
        return {"queries": list(DEFAULT_QUERIES), "families": ["graph", "3d", "smiles"],
                "rationale": "fallback: generic molecular sweep"}

    # ③ LLM judges whether the local library is enough ----------------------------------------- #
    def _judge_satisfaction(self, task: dict, families: list[str], local_hits: list[dict]) -> dict:
        try:
            system = ("You decide if the LOCAL model library is sufficient for the task, or we should "
                      'search online for more. Reply ONLY JSON: {"satisfied":true|false,"reason":..}')
            user = (f"Task: {json.dumps(task)}\nNeeded families: {families}\n"
                    f"Local models: {json.dumps([{'ref': m.get('ref'), 'family': m.get('family')} for m in local_hits])}\n\n"
                    "Is the local library enough to build a strong solution (good coverage of the needed "
                    "decorrelated families)? Or should we search online for more? Return ONLY JSON.")
            out = self.call_json(system, user)
            if isinstance(out, dict) and "satisfied" in out:
                out["source"] = "llm"
                return out
        except Exception as e:  # noqa: BLE001
            print(f"[retrieval] satisfaction LLM failed ({e}); deterministic family-coverage fallback")
        covered = {str(m.get("family", "")).lower() for m in local_hits}
        need = {f.lower() for f in (families or [])}
        return {"satisfied": bool(local_hits) and need.issubset(covered),
                "reason": f"family coverage {sorted(covered)} vs needed {sorted(need)}", "source": "fallback"}

    # ⑤ LLM ranks for the task + mode; code enforces the template rule -------------------------- #
    def _rank(self, task: dict, candidates: list[dict], plan: dict) -> list[dict]:
        try:
            system = (
                "You select pretrained models for an AutoML task. For each chosen model recommend a mode: "
                "'finetune' (ONLY if has_template=true) or 'frozen' (embeddings; any model). Prefer a few "
                'DECORRELATED families. Reply ONLY JSON: {"selected":[{"ref":..,"family":..,"mode":..,"reason":..}]}'
            )
            user = (f"Task: {json.dumps(task)}\nSearch plan: {json.dumps(plan)}\n"
                    f"Candidates: {json.dumps(candidates, default=str)[:3500]}\n\n"
                    "Pick the best few (decorrelated families) and a mode for each. Return ONLY JSON.")
            out = self.call_json(system, user)
            sel = out.get("selected") if isinstance(out, dict) else None
            if sel:
                return [self._enforce_template(s, candidates) for s in sel]
        except Exception as e:  # noqa: BLE001
            print(f"[retrieval] rank LLM failed ({e}); deterministic rank")
        return [{"ref": ref, "family": fam, "mode": "finetune",
                 "reason": "validated fallback: has fine-tune template, decorrelated family"}
                for ref, fam in TEMPLATED.items()]

    # --- helpers (deterministic) -------------------------------------------------------------- #
    @staticmethod
    def _to_entry(cand) -> dict:
        d = cand.to_dict()
        base = d["ref"].split("/")[-1].lower()
        return {"ref": d["ref"], "family": d.get("family", "unknown"),
                "has_template": base in TEMPLATED, "downloads": d.get("downloads"),
                "source": "hf", "tags": d.get("tags", [])[:8]}

    @staticmethod
    def _mark(m: dict) -> dict:
        base = str(m.get("ref", "")).split("/")[-1].lower()
        return {**m, "has_template": m.get("has_template", base in TEMPLATED)}

    @staticmethod
    def _enforce_template(sel: dict, candidates: list[dict]) -> dict:
        """Code guard: only models with a verified template may be 'finetune'; else force 'frozen'."""
        ref = str(sel.get("ref", ""))
        base = ref.split("/")[-1].lower()
        has_tpl = base in TEMPLATED or any(c.get("ref") == ref and c.get("has_template") for c in candidates)
        if sel.get("mode") == "finetune" and not has_tpl:
            sel["mode"] = "frozen"
            sel["reason"] = (str(sel.get("reason", "")) + " [forced frozen: no template]").strip()
        sel.setdefault("mode", "frozen")
        return sel
