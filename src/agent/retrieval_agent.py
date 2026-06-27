"""Retrieval Agent (LLM-driven, task-guided) — the model-search node (after Setup).

Now that Setup tells us the task, this agent searches for suitable models *on purpose*:
  1. an LLM PLANS the search from the task — what representation families + HF search terms suit
     a molecular pEC50 regression (graph / 3D / SMILES …);
  2. deterministic code EXECUTES a live HuggingFace search with those terms + adds the local
     templated registry (models we can actually fine-tune);
  3. an LLM RANKS the candidates for THIS task and recommends a mode — **frozen** (extract
     embeddings; works for any model) or **finetune** (only models with a verified template).

Our principle holds: the LLM proposes (search plan, ranking, mode); deterministic code verifies
(runs the search, and ENFORCES that only templated models may be 'finetune').
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.LLM_base import LLMJsonAgent
from src.agent.hf_retrieval import DEFAULT_QUERIES, discover_models

# Models we have a verified FINE-TUNE template for. Anything else can only be used FROZEN
# (extract embeddings + sklearn head) — fine-tuning a new backbone needs a new template.
TEMPLATED = {
    "chemeleon": {"family": "graph", "note": "graph D-MPNN foundation; fine-tuned single ~0.59"},
    "unimol":    {"family": "3d",    "note": "3D geometry foundation; fine-tuned single ~0.62, decorrelated from graph"},
}


class RetrievalAgent(LLMJsonAgent):
    name = "retrieval"
    source: str = "unknown"

    def run(self, setup_report: dict, top_k: int = 12, out_path: str | Path | None = None) -> dict:
        task = self._task_brief(setup_report)
        plan = self._plan(task)                                   # 1. LLM plans the search
        hf = discover_models(queries=plan.get("queries") or list(DEFAULT_QUERIES), top_k=top_k)  # 2. execute
        candidates = self._with_templated(hf)
        ranked = self._rank(task, candidates, plan)              # 3. LLM ranks + frozen/finetune (code-enforced)
        result = {"task": task, "search_plan": plan, "n_discovered": len(candidates),
                  "selected": ranked, "source": self.source}
        if out_path:
            Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _task_brief(self, setup_report: dict) -> dict:
        t = (setup_report or {}).get("task", {})
        return {"type": t.get("type"), "domain": t.get("domain"), "summary": t.get("summary"),
                "metric": (setup_report or {}).get("metric"),
                "target": (setup_report or {}).get("schema", {}).get("target_col")}

    # 1. LLM plans WHAT to search for, from the task ------------------------------------------- #
    def _plan(self, task: dict) -> dict:
        system = (
            "You plan a pretrained-model search for an AutoML pipeline. Given the task, decide what "
            "KINDS of models suit it and what to search for on HuggingFace. Reply ONLY JSON: "
            '{"queries":[..4-7 HF search terms..],"families":[..e.g. "graph","3d","smiles"..],"rationale":..}'
        )
        user = (
            f"Task: {json.dumps(task)}\n\n"
            "Propose HF search queries and the representation families worth trying for THIS task. "
            "Prefer DECORRELATED families (they stack better). Return ONLY the JSON object."
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

    # 2. add the templated (fine-tunable) local models so ranking can prefer/flag them --------- #
    def _with_templated(self, hf) -> list[dict]:
        out = []
        present = set()
        for c in hf:
            d = c.to_dict()
            base = d["ref"].split("/")[-1].lower()
            d["has_template"] = base in TEMPLATED
            present.add(base)
            out.append(d)
        for ref, meta in TEMPLATED.items():
            if ref not in present:
                out.append({"ref": ref, "family": meta["family"], "has_template": True,
                            "note": meta["note"], "downloads": None, "score": None})
        return out

    # 3. LLM ranks for the task + recommends mode; code enforces the template rule ------------- #
    def _rank(self, task: dict, candidates: list[dict], plan: dict) -> list[dict]:
        try:
            system = (
                "You select pretrained models for an AutoML task. For each chosen model recommend a mode: "
                "'finetune' (ONLY if has_template=true — we have a verified training template) or 'frozen' "
                "(extract embeddings, works for any model). Prefer a few DECORRELATED families over many of "
                'one. Reply ONLY JSON: {"selected":[{"ref":..,"family":..,"mode":"frozen|finetune","reason":..}]}'
            )
            user = (
                f"Task: {json.dumps(task)}\nSearch plan: {json.dumps(plan)}\n"
                f"Candidates: {json.dumps(candidates, default=str)[:3500]}\n\n"
                "Pick the best few models (decorrelated families) and a mode for each. Return ONLY JSON."
            )
            out = self.call_json(system, user)
            sel = out.get("selected") if isinstance(out, dict) else None
            if sel:
                return [self._enforce_template(s, candidates) for s in sel]
        except Exception as e:  # noqa: BLE001
            print(f"[retrieval] rank LLM failed ({e}); deterministic rank")
        # fallback: the templated decorrelated pair (finetune) — our validated default
        return [{"ref": ref, "family": meta["family"], "mode": "finetune",
                 "reason": "validated fallback: has fine-tune template, decorrelated family"}
                for ref, meta in TEMPLATED.items()]

    @staticmethod
    def _enforce_template(sel: dict, candidates: list[dict]) -> dict:
        """Code guard: only models with a verified template may be 'finetune'; else force 'frozen'."""
        ref = str(sel.get("ref", ""))
        base = ref.split("/")[-1].lower()
        has_tpl = base in TEMPLATED or any(c.get("ref") == ref and c.get("has_template") for c in candidates)
        if sel.get("mode") == "finetune" and not has_tpl:
            sel["mode"] = "frozen"
            sel["reason"] = (str(sel.get("reason", "")) + " [forced frozen: no fine-tune template]").strip()
        sel.setdefault("mode", "frozen")
        return sel
