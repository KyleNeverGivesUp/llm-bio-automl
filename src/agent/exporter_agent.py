"""Export the best selected plan into final deliverables."""

from src.agent.LLM_base import BaseAgent
from src.agent.agent_context import AgentResult, RunContext
from src.exporter import export_best_submission


class ExporterAgent(BaseAgent):
    name = "exporter"

    def run(self, context: RunContext) -> AgentResult:
        final_report = export_best_submission(run_id=context.run_id, data_dir=context.data_dir)
        return AgentResult(
            agent_name=self.name,
            status="done",
            summary="Exported final submission and report.",
            outputs=final_report,
        )
