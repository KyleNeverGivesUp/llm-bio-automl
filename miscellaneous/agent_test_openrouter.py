"""Smoke-test the OpenRouter JSON response path and logging."""

import json
from datetime import datetime

from src.agent.LLM_base import LLMJsonAgent
from src.agent.agent_context import AgentResult, RunContext


class SmokeTestAgent(LLMJsonAgent):
    name = "smoke_test"

    def run(self, context: RunContext) -> AgentResult:
        system_prompt = (
            "You are a smoke test assistant for an OpenRouter integration. "
            "Return valid JSON only."
        )
        user_prompt = """
Respond with JSON exactly in this shape:
{
  "ok": true,
  "provider": "openrouter",
  "message": "short success message",
  "model_echo": "brief text naming the model family if known"
}
"""

        log_call_name = "openrouter_smoketest"
        log_path = context.run_dir / "llm_logs" / f"{log_call_name}.json"

        try:
            parsed = self.call_json_logged(context, log_call_name, system_prompt, user_prompt)
            summary = {
                "status": "ok",
                "run_id": context.run_id,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "parsed_response": parsed,
                "llm_log_path": str(log_path),
            }
            return AgentResult(
                agent_name=self.name,
                status="done",
                summary="OpenRouter smoke test succeeded.",
                outputs=summary,
            )
        except Exception as exc:
            summary = {
                "status": "error",
                "run_id": context.run_id,
                "model": self.config.model,
                "base_url": self.config.base_url,
                "error": str(exc),
                "llm_log_path": str(log_path),
            }
            return AgentResult(
                agent_name=self.name,
                status="failed",
                summary="OpenRouter smoke test failed.",
                outputs=summary,
            )


def make_smoketest_run_id() -> str:
    return f"openrouter_smoketest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main() -> None:
    context = RunContext(run_id=make_smoketest_run_id())
    agent = SmokeTestAgent()
    result = agent.run(context)

    report_path = context.run_dir / "smoketest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.outputs, indent=2, ensure_ascii=False), encoding="utf-8")

    print("openrouter smoke test completed:")
    print(f"  status: {result.status}")
    print(f"  model: {result.outputs['model']}")
    print(f"  base_url: {result.outputs['base_url']}")
    print(f"  report_path: {report_path}")
    print(f"  llm_log_path: {result.outputs['llm_log_path']}")
    if result.status == "done":
        print(f"  parsed_response: {json.dumps(result.outputs['parsed_response'], ensure_ascii=False)}")
    else:
        print(f"  error: {result.outputs['error']}")


if __name__ == "__main__":
    main()
