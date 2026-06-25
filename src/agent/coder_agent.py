"""Execute supported design plans and persist their metrics."""

import json

from src.agent.LLM_base import BaseAgent
from src.agent.agent_context import AgentResult, RunContext
from src.data_utils import load_activity_test, load_activity_train
from src.runner import run_plan
from src.schemas import PlanSpec


class CoderAgent(BaseAgent):
    name = "coder"

    def run(self, context: RunContext) -> AgentResult:
        train_df = load_activity_train(context.data_dir)
        test_df = load_activity_test(context.data_dir)

        payload = json.loads((context.run_dir / "design_plans.json").read_text(encoding="utf-8"))
        plans = [PlanSpec(**plan) for plan in payload["plans"]]

        supported_plans = [
            plan
            for plan in plans
            if (
                plan.feature_type == "morgan_fingerprint"
                and plan.model_type in {"ridge", "elastic_net", "random_forest", "xgboost"}
            )
            or (
                plan.feature_type in {"skill_embedding", "skill_embedding_plus_morgan"}
                and plan.model_type in {"ridge", "elastic_net", "random_forest", "xgboost"}
                and plan.skill_ref
            )
        ]

        skipped_plans = [plan.plan_id for plan in plans if plan not in supported_plans]

        result_paths: list[str] = []
        for plan in supported_plans:
            output_dir = context.run_dir / "plans" / plan.plan_id
            run_plan(plan, train_df, test_df, str(output_dir))
            result_paths.append(str(output_dir / "metrics.json"))

        return AgentResult(
            agent_name=self.name,
            status="done",
            summary=f"Executed {len(supported_plans)} supported plans.",
            outputs={
                "result_paths": result_paths,
                "n_supported_plans": len(supported_plans),
                "skipped_plan_ids": skipped_plans,
            },
        )
