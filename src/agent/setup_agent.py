"""Validate inputs and write the initial task specification."""

from src.agent.LLM_base import BaseAgent
from src.agent.agent_context import AgentResult, RunContext
from ..data_utils import validate_activity_dataset, write_dataset_report
from ..task_spec_builder import build_pxr_activity_task_spec, write_task_spec


class SetupAgent(BaseAgent):
    name = "setup"

    def run(self, context: RunContext) -> AgentResult:
        run_dir = context.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        dataset_report = validate_activity_dataset(context.data_dir)
        dataset_report_path = run_dir / "dataset_report.json"
        write_dataset_report(dataset_report, dataset_report_path)
        if not dataset_report["valid"]:
            return AgentResult(
                agent_name=self.name,
                status="failed",
                summary="Dataset validation failed.",
                outputs={
                    "dataset_report_path": str(dataset_report_path),
                    "errors": dataset_report["errors"],
                },
            )

        task_spec = build_pxr_activity_task_spec(context.data_dir)
        task_spec_path = run_dir / "task_spec.json"
        write_task_spec(task_spec, task_spec_path)

        return AgentResult(
            agent_name=self.name,
            status="done",
            summary="Validated dataset and wrote task spec.",
            outputs={
                "dataset_report_path": str(dataset_report_path),
                "task_spec_path": str(task_spec_path),
            },
        )
