"""Run the manager agent as a simple CLI entrypoint."""

import logging

from src.agent.manager_agent import ManagerAgent
from src.agent.agent_context import RunContext
from src.constants import RUN_ID


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main() -> None:
    context = RunContext(run_id=RUN_ID)
    manager = ManagerAgent()
    result = manager.run(context)

    print("llm agent pipeline completed:")
    print(f"  status: {result.status}")
    print(f"  summary: {result.summary}")
    if "final_selection" in result.outputs:
        print(f"  best_plan_id: {result.outputs['final_selection']['best_plan']['plan_id']}")
    if "exporter" in result.outputs:
        print(f"  submission_path: {result.outputs['exporter']['submission_path']}")


if __name__ == "__main__":
    main()
