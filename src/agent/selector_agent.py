"""Rank plan metrics and select the current best candidate."""

import json
from pathlib import Path

from src.agent.LLM_base import LLMJsonAgent
from src.agent.agent_context import AgentResult, RunContext
from src.selector import load_all_metrics, sort_metrics, write_best_plan, write_leaderboard


class SelectorAgent(LLMJsonAgent):
    name = "selector"

    def __init__(self, selection_mode: str) -> None:
        super().__init__()
        self.selection_mode = selection_mode

    def _load_baseline_plan_ids(self, context: RunContext) -> set[str]:
        design_plans_path = context.run_dir / "design_plans.json"
        payload = json.loads(Path(design_plans_path).read_text(encoding="utf-8"))
        return {plan["plan_id"] for plan in payload["plans"]}

    def run(self, context: RunContext) -> AgentResult:
        metrics_list = load_all_metrics(str(context.run_dir))
        if self.selection_mode == "baseline":
            baseline_plan_ids = self._load_baseline_plan_ids(context)
            metrics_list = [item for item in metrics_list if item["plan_id"] in baseline_plan_ids]
        llm_log_path = context.run_dir / "llm_logs" / f"selector_{self.selection_mode}.json"

        ranked_metrics = sort_metrics(metrics_list, primary_metric=context.primary_metric)
        if not ranked_metrics:
            return AgentResult(
                agent_name=self.name,
                status="failed",
                summary="No metrics found for selection.",
                outputs={},
            )

        ranking_metric_used = "rae" if context.primary_metric == "RAE" else context.primary_metric
        write_leaderboard(metrics_list, context.run_dir / "leaderboard.json", primary_metric=context.primary_metric)

        system_prompt = (
            "You are a SelectorAgent for a biological ML AutoML workflow. "
            "Select the best candidate from ranked experiment results and return JSON only."
        )
        user_prompt = f"""
Selection mode: {self.selection_mode}
Primary metric requested: {context.primary_metric}
Ranking metric currently used by the execution layer: {ranking_metric_used}

Candidates are already sorted best-to-worst by the execution layer:
{json.dumps(ranked_metrics, indent=2)}

To preserve parity with the existing manual pipeline, choose the plan that is best according to the ranking metric currently used by the execution layer.

Respond with JSON exactly in this shape:
{{
  "selected_plan_id": "<one of the candidate plan_id values>",
  "reason": "<short reason>"
}}
"""

        used_fallback = False
        try:
            decision = self.call_json_logged(context, f"selector_{self.selection_mode}", system_prompt, user_prompt)
            selected_plan_id = decision["selected_plan_id"]
            best_plan = next(item for item in ranked_metrics if item["plan_id"] == selected_plan_id)
        except Exception as exc:
            used_fallback = True
            decision = {
                "selected_plan_id": ranked_metrics[0]["plan_id"],
                "reason": "Fallback to deterministic top-ranked candidate.",
                "fallback_error": str(exc),
            }
            best_plan = ranked_metrics[0]

        write_best_plan(best_plan, context.run_dir / "best_plan.json")
        report_path = context.run_dir / f"{self.selection_mode}_selection_report.json"
        report_path.write_text(
            json.dumps(
                {
                    **decision,
                    "used_fallback": used_fallback,
                    "llm_log_path": str(llm_log_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return AgentResult(
            agent_name=self.name,
            status="done",
            summary=f"Selected {best_plan['plan_id']} as best {self.selection_mode}.",
            outputs={
                "best_plan": best_plan,
                "leaderboard_path": str(context.run_dir / "leaderboard.json"),
                "best_plan_path": str(context.run_dir / "best_plan.json"),
                "selection_report_path": str(report_path),
                "used_fallback": used_fallback,
                "llm_log_path": str(llm_log_path),
            },
        )
