"""Expose public agent types without eager imports that create cycles."""

from src.agent.agent_context import AgentResult, RunContext

__all__ = ["AgentResult", "ManagerAgent", "RunContext"]


def __getattr__(name: str):
    if name == "ManagerAgent":
        from src.agent.manager_agent import ManagerAgent

        return ManagerAgent
    raise AttributeError(name)
