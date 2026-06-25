from src.agent.agent_context import RunContext
from src.agent.retrieval_agent import RetrievalAgent
from src.agent.setup_agent import SetupAgent
from src.constants import RUN_ID


if __name__ == "__main__":
    run_context = RunContext(run_id=RUN_ID)

    setup_result = SetupAgent().run(run_context)
    print("setup status:", setup_result.status)
    print("task_spec_path:", setup_result.outputs.get("task_spec_path"))

    retrieval_result = RetrievalAgent().run(run_context)
    print("retrieval status:", retrieval_result.status)
    print("retrieval_result_path:", retrieval_result.outputs.get("retrieval_result_path"))
    print("used_fallback:", retrieval_result.outputs.get("used_fallback"))
    print("decision:")
    print(retrieval_result.outputs.get("decision"))
