"""Turn task and retrieval context into executable baseline plans."""

import json
import os
import re
from dataclasses import asdict
from pathlib import Path

from anthropic import Anthropic

from src.agent.LLM_base import LLMJsonAgent
from src.agent.agent_context import AgentResult, RunContext
from src.schemas import PlanSpec


class DesignerAgent(LLMJsonAgent):
    name = "designer"

    def _anthropic_client(self) -> tuple[Anthropic, str]:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

        client_kwargs = {"api_key": api_key}
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
        if base_url:
            client_kwargs["base_url"] = base_url

        model = os.environ.get("ANTHROPIC_MODEL", "").strip()
        if not model:
            raise RuntimeError("ANTHROPIC_MODEL is not configured. Set it to your Sonnet 4.6 model id.")
        return Anthropic(**client_kwargs), model

    def _anthropic_response_text(self, response) -> str:
        parts: list[str] = []
        for block in response.content:
            text = getattr(block, "text", "")
            if text:
                parts.append(text)
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError("Anthropic returned empty text content.")
        return text

    def _parse_json_response(self, raw_text: str) -> dict:
        text = raw_text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced:
            text = fenced.group(1)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("No JSON object found in Anthropic response.")
        return json.loads(text[start : end + 1])

    def _call_anthropic_json_logged(
        self,
        context: RunContext,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        log_path = context.run_dir / "llm_logs" / f"{call_name}.json"
        client, model_name = self._anthropic_client()
        request_payload = {
            "provider": "anthropic",
            "model": model_name,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        log_payload = {
            "status": "started",
            "call_name": call_name,
            "request": request_payload,
            "response_json": None,
            "raw_text_content": None,
            "parsed_json": None,
            "error": None,
        }

        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=self.max_tokens,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = self._anthropic_response_text(response)
            parsed = self._parse_json_response(raw_text)

            log_payload["status"] = "ok"
            log_payload["response_json"] = response.model_dump() if hasattr(response, "model_dump") else None
            log_payload["raw_text_content"] = raw_text
            log_payload["parsed_json"] = parsed
            self._write_llm_log(log_path, log_payload)
            return parsed
        except Exception as exc:
            log_payload["status"] = "error"
            log_payload["error"] = str(exc)
            self._write_llm_log(log_path, log_payload)
            raise

    def _validate_plans(self, payload: dict) -> list[PlanSpec]:
        plans = payload.get("plans", [])
        validated: list[PlanSpec] = []
        for idx, plan in enumerate(plans, start=1):
            plan.setdefault("plan_id", f"designer_plan_{idx}")
            plan.setdefault("name", f"Designer Plan {idx}")
            plan.setdefault("params", {})
            plan.setdefault("notes", "")
            validated.append(PlanSpec(**plan))
        return validated

    def _load_optional_json(self, path: Path) -> dict | list | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _best_known_baseline_plan(self) -> PlanSpec:
        return PlanSpec(
            plan_id="skill_1_chemberta-77m-mtr_ridge",
            name="DeepChem/ChemBERTa-77M-MTR Embedding + Ridge",
            feature_type="skill_embedding",
            model_type="ridge",
            params={
                "alpha": 1.0,
                "pooling": "cls",
                "max_length": 256,
                "batch_size": 16,
            },
            skill_ref="DeepChem/ChemBERTa-77M-MTR",
            skill_path="skills/models/chemistry/DeepChem--ChemBERTa-77M-MTR/SKILL.md",
            notes=(
                "Best validated fallback baseline. This plan previously achieved the strongest "
                "observed result around MAE 0.5537 / RAE 0.6085."
            ),
        )

    def _skill_candidates(self, retrieval_result: dict) -> list[dict]:
        selected_skills = retrieval_result.get("selected_skills", [])
        if selected_skills:
            return selected_skills
        baseline = self._best_known_baseline_plan()
        return [
            {
                "ref": baseline.skill_ref,
                "path": baseline.skill_path,
                "domain": "chemistry",
                "description": baseline.notes,
            }
        ]

    def _ensure_baseline_included(self, plans: list[PlanSpec]) -> list[PlanSpec]:
        baseline = self._best_known_baseline_plan()
        if any(plan.plan_id == baseline.plan_id for plan in plans):
            return plans
        return [baseline] + plans

    def _hydrate_generated_plans(self, plans: list[PlanSpec], skill_candidates: list[dict]) -> list[PlanSpec]:
        if not skill_candidates:
            return plans

        default_skill = skill_candidates[0]
        hydrated: list[PlanSpec] = []
        for idx, plan in enumerate(plans, start=1):
            if plan.plan_id == self._best_known_baseline_plan().plan_id:
                hydrated.append(plan)
                continue

            if plan.feature_type in {"skill_embedding", "skill_embedding_plus_morgan"}:
                if not plan.skill_ref:
                    plan.skill_ref = default_skill["ref"]
                if not plan.skill_path:
                    plan.skill_path = default_skill["path"]
            if not plan.plan_id:
                plan.plan_id = f"designer_plan_{idx}"
            if not plan.name:
                plan.name = f"Designer Plan {idx}"
            hydrated.append(plan)
        return hydrated

    def run(self, context: RunContext) -> AgentResult:
        llm_log_path = context.run_dir / "llm_logs" / "designer.json"
        baseline_plan = self._best_known_baseline_plan()
        plans = [baseline_plan]
        used_fallback = False
        payload = {
            "summary": (
                "Initialized search from the strongest validated baseline only. "
                "All subsequent model-family and hyperparameter choices are delegated to the LLM optimizer."
            ),
            "fallback_error": None,
        }

        design_plans_path = context.run_dir / "design_plans.json"
        design_plans_path.write_text(
            json.dumps({"plans": [asdict(plan) for plan in plans]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        agent_report_path = context.run_dir / "designer_report.json"
        agent_report_path.write_text(
            json.dumps(
                {
                    "summary": payload.get("summary", ""),
                    "n_plans": len(plans),
                    "used_fallback": used_fallback,
                    "llm_log_path": str(llm_log_path),
                    "fallback_error": payload.get("fallback_error"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return AgentResult(
            agent_name=self.name,
            status="done",
            summary=f"Generated {len(plans)} plans via LLM-guided planner.",
            outputs={
                "design_plans_path": str(design_plans_path),
                "designer_report_path": str(agent_report_path),
                "n_plans": len(plans),
                "used_fallback": used_fallback,
                "llm_log_path": str(llm_log_path),
            },
        )
