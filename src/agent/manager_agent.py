"""Orchestrate the end-to-end AutoML agent pipeline."""

import json
import logging
from dataclasses import asdict

from src.agent.LLM_base import BaseAgent
from src.agent.coder_agent import CoderAgent
from src.agent.designer_agent import DesignerAgent
from src.agent.exporter_agent import ExporterAgent
from src.agent.retrieval_agent import RetrievalAgent
from src.agent.selector_agent import SelectorAgent
from src.agent.setup_agent import SetupAgent
from src.agent.tuner_agent import TunerAgent
from src.agent.agent_context import AgentResult, RunContext
from src.schemas import RunState


logger = logging.getLogger(__name__)


class ManagerAgent(BaseAgent):
    name = "manager"
    tuning_rounds = 5

    def __init__(self) -> None:
        self.setup_agent = SetupAgent()
        self.retrieval_agent = RetrievalAgent()
        self.designer_agent = DesignerAgent()
        self.coder_agent = CoderAgent()
        self.baseline_selector_agent = SelectorAgent(selection_mode="baseline")
        self.tuner_agent = TunerAgent()
        self.final_selector_agent = SelectorAgent(selection_mode="overall")
        self.exporter_agent = ExporterAgent()

    def _write_run_state(self, run_state: RunState, context: RunContext) -> None:
        output_file = context.run_dir / "run_state.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(asdict(run_state), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def run(self, context: RunContext) -> AgentResult:
        context.run_dir.mkdir(parents=True, exist_ok=True)
        run_state = RunState(run_id=context.run_id, current_stage="init")
        self._write_run_state(run_state, context)

        logger.info("ManagerAgent starting run_id=%s", context.run_id)

        run_state.current_stage = "setup"
        self._write_run_state(run_state, context)
        setup_result = self.setup_agent.run(context)
        if setup_result.status != "done":
            return AgentResult(
                agent_name=self.name,
                status="failed",
                summary="Setup agent failed.",
                outputs={"setup_result": setup_result.outputs},
            )
        run_state.task_spec_path = setup_result.outputs["task_spec_path"]
        self._write_run_state(run_state, context)

        run_state.current_stage = "retrieval"
        self._write_run_state(run_state, context)
        retrieval_result = self.retrieval_agent.run(context)

        run_state.current_stage = "designer"
        self._write_run_state(run_state, context)
        designer_result = self.designer_agent.run(context)
        run_state.plan_paths = [designer_result.outputs["design_plans_path"]]
        self._write_run_state(run_state, context)

        run_state.current_stage = "coder"
        self._write_run_state(run_state, context)
        coder_result = self.coder_agent.run(context)
        run_state.result_paths = coder_result.outputs["result_paths"]
        self._write_run_state(run_state, context)

        run_state.current_stage = "select_best_baseline"
        self._write_run_state(run_state, context)
        baseline_selection = self.baseline_selector_agent.run(context)
        run_state.best_plan_path = baseline_selection.outputs["best_plan_path"]
        self._write_run_state(run_state, context)

        tuner_round_results: list[dict] = []
        final_selection = baseline_selection
        for round_idx in range(1, self.tuning_rounds + 1):
            run_state.current_stage = f"tuner_round_{round_idx}"
            self._write_run_state(run_state, context)
            tuner_result = self.tuner_agent.run(context, iteration=round_idx, total_iterations=self.tuning_rounds)
            tuner_round_results.append(tuner_result.outputs)

            run_state.current_stage = f"select_best_overall_round_{round_idx}"
            self._write_run_state(run_state, context)
            final_selection = self.final_selector_agent.run(context)
            run_state.best_plan_path = final_selection.outputs["best_plan_path"]
            self._write_run_state(run_state, context)

        run_state.current_stage = "exporter"
        self._write_run_state(run_state, context)
        exporter_result = self.exporter_agent.run(context)

        run_state.current_stage = "completed"
        self._write_run_state(run_state, context)

        return AgentResult(
            agent_name=self.name,
            status="done",
            summary="Completed full LLM-agent PXR pipeline.",
            outputs={
                "setup": setup_result.outputs,
                "retrieval": retrieval_result.outputs,
                "designer": designer_result.outputs,
                "coder": coder_result.outputs,
                "baseline_selection": baseline_selection.outputs,
                "tuner_rounds": tuner_round_results,
                "tuner": tuner_round_results[-1] if tuner_round_results else {},
                "final_selection": final_selection.outputs,
                "exporter": exporter_result.outputs,
                "run_state_path": str(context.run_dir / "run_state.json"),
            },
        )
