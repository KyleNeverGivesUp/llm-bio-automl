"""Define shared context and result types for agents."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    run_id: str
    data_dir: str = "data/pxr_activity"
    primary_metric: str = "RAE"

    @property
    def run_dir(self) -> Path:
        return Path("outputs") / self.run_id


@dataclass
class AgentResult:
    agent_name: str
    status: str
    summary: str
    outputs: dict[str, Any] = field(default_factory=dict)
