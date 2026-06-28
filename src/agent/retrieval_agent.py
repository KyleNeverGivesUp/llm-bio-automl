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
        # Every intermediate step is landed in the result (→ retrieval_result.json) so the whole
        # discover→filter→select process is inspectable/debuggable, not just the final picks.
        def brief(ms):  # compact view of a model list for the artifact
            return [{"ref": m.get("ref"), "family": m.get("family"),
                     "source": m.get("source"), "has_template": m.get("has_template")} for m in ms]

        task = self._task_brief(setup_report)
        plan = self._plan(task)                                          # ① LLM: queries + families
        families = plan.get("families") or []

        local_before = model_registry.search(families)                  # ② local-first hits (pre-online)
        sat = self._judge_satisfaction(task, families, local_before)    # ③ LLM: satisfied?

        went_online, n_added, online_pulled = False, 0, []
        if not sat.get("satisfied"):
            # task-specific LLM queries + generic sweep, so a narrow query set still yields finds to cache
            queries = list(dict.fromkeys((plan.get("queries") or []) + list(DEFAULT_QUERIES)))
            found = discover_models(queries=queries, top_k=top_k)       # ④ multi-source: HF + GitHub + Zenodo
            online_pulled = [self._to_entry(c) for c in found]          # (was misnamed 'hf' — it's all 3 sources)
            n_added = model_registry.add(online_pulled)                 #    write-back (grow library)
            went_online = True

        local_hits = model_registry.search(families) or model_registry.load()
        candidates = [self._mark(m) for m in local_hits]                # ⑤ full candidate pool (pre-select)
        ranked = self._rank(task, candidates, plan)                     # ⑥ LLM: select + mode

        result = {
            "task": task,
            "search_plan": plan,                                        # ①
            "local_hits_before_online": brief(local_before),           # ②
            "satisfaction": sat,                                        # ③
            "went_online": went_online,
            "online_pulled": brief(online_pulled),                     # ④ what this run pulled from outside
            "n_added_to_library": n_added,
            "queries_used": (queries if went_online else []),
            "candidates": brief(candidates),                           # ⑤ full pool the LLM ranked from
            "n_candidates": len(candidates),
            "selected": ranked,                                        # ⑥ final picks
            "source": self.source,
        }
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
            "You plan a pretrained-model search for an AutoML pipeline (searches HuggingFace + GitHub + "
            "Zenodo). Given the task, decide what KINDS of models suit it AND name specific models to find. "
            "For families, use ONLY this fixed vocabulary: graph, 3d, smiles, descriptor, multiview. "
            'Reply ONLY JSON: {"queries":[..],"families":[..from vocabulary..],"rationale":..}'
        )
        user = (
            f"Task: {json.dumps(task)}\n\n"
            "Propose search queries + representation families (graph/3d/smiles/descriptor/multiview) for THIS "
            "task; prefer DECORRELATED families (they stack better). For queries, include the NAMES of specific "
            "strong pretrained molecular foundation models you know that fit this task — from YOUR knowledge "
            "(the best ones often live on GitHub/Zenodo, not HuggingFace, so naming them is how we find them). "
            "CRITICAL: each query must be SHORT — ONE model name or ONE concept, 1-3 words (e.g. 'ChemProp', "
            "'Uni-Mol', 'GROVER', 'MolE'), NOT a long combined string. Give ~8-12 such short queries. "
            "Return ONLY JSON."
        )
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
        lib = d.get("library", "")
        source = lib if lib in ("github", "zenodo") else "hf"   # real origin, not hardcoded
        return {"ref": d["ref"], "family": d.get("family", "unknown"),
                "has_template": base in TEMPLATED, "downloads": d.get("downloads"),
                "source": source, "tags": d.get("tags", [])[:8]}

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
