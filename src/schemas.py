from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json

@dataclass
class MenuPlan:
    """A single modeling recipe for the Approach-1 menu (M1+).

    Forward-looking replacement for the legacy ``PlanSpec`` (whose
    ``feature_type``/``model_type`` naming is still wired into the
    soon-to-be-replaced runner/tuner/agent code). New M1 modules
    (``cv_runner`` etc.) consume ``MenuPlan``; the two converge in M3
    when the legacy pipeline is deleted.

    ``featurizer`` is a key in the featurizer registry (``src/featurizers.py``),
    ``model`` a key in the model registry (``src/models.py``). ``params`` carries
    hyperparameters for both. ``seeds`` drives multi-seed averaging (M2) — the
    OOF/test predictions are averaged across these seeds inside each fold.
    """
    plan_id: str
    name: str
    featurizer: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=lambda: [42])
    skill_ref: str | None = None      # e.g. "DeepChem/ChemBERTa-77M-MTR" for embedding featurizers
    skill_path: str | None = None
    notes: str = ""


@dataclass
class FoldSpec:
    """Frozen scaffold-CV assignment loaded from ``folds.json`` (see cv_split.py)."""
    strategy: str
    n_folds: int
    seed: int
    fold_of_row: list[int]            # fold_of_row[i] = fold index (0..n_folds-1) of train row i

    @classmethod
    def from_json(cls, path: str | Path) -> "FoldSpec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assignments = payload["assignments"]
        # assignments is {row_index_str: fold_int}; rebuild a dense, row-ordered list
        fold_of_row = [int(assignments[str(i)]) for i in range(len(assignments))]
        return cls(
            strategy=payload.get("strategy", "scaffold"),
            n_folds=int(payload["n_folds"]),
            seed=int(payload.get("seed", 42)),
            fold_of_row=fold_of_row,
        )


@dataclass
class TaskSpec:
    challenge_name: str
    task_title: str
    task_description: str
    target_column: str
    primary_metric: str
    submission_columns: list[str]
    data_dir: str

@dataclass
class TaskInference:
    task_domain: str
    task_modality: str
    task_type: str
    reason: str

@dataclass
class PlanSpec:
    plan_id: str
    name: str
    feature_type: str
    model_type: str
    params: dict[str, Any] = field(default_factory=dict) # model params
    skill_ref: str | None = None
    skill_path: str | None = None
    notes: str = ""


@dataclass
class RunState:
    run_id: str
    task_spec_path: str | None = None
    current_stage: str = "init"
    plan_paths: list[str] = field(default_factory=list)
    result_paths: list[str] = field(default_factory=list)
    best_plan_path: str | None = None

def to_dict(obj):
    return asdict(obj)

def write_json(data: dict, output_path: str) -> None:
    Path(output_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )