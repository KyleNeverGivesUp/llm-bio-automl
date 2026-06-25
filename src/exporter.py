import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.constants import RUN_ID

# Packages whose versions decide whether a result reproduces (featurization +
# modeling stack). Recorded in the submission manifest.
_MANIFEST_PACKAGES = [
    "rdkit", "scikit-learn", "xgboost", "numpy", "pandas",
    "scipy", "torch", "transformers",
]


def _pkg_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in _MANIFEST_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def build_manifest(run_dir: Path, source_label: str, extra: dict | None = None) -> dict:
    """Reproducibility manifest: everything needed to regenerate this submission —
    code version, package versions, python, the fold spec, and the run artifacts
    that produced the predictions (leaderboard + ensemble report when present)."""
    run_dir = Path(run_dir)
    manifest = {
        "source": source_label,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "packages": _pkg_versions(),
    }
    for name in ("leaderboard.json", "ensemble_report.json"):
        path = run_dir / name
        if path.exists():
            manifest[name.replace(".json", "")] = load_json(path)
    if extra:
        manifest.update(extra)
    return manifest


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: dict, output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_submission(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    required_pred_cols = {"SMILES", "Molecule Name", "pEC50"}
    missing_pred_cols = required_pred_cols - set(pred_df.columns)
    if missing_pred_cols:
        raise ValueError(f"Prediction file missing columns: {sorted(missing_pred_cols)}")

    submission = pred_df[["SMILES", "Molecule Name", "pEC50"]].copy()

    if len(submission) != len(test_df):
        raise ValueError(
            f"Submission row count {len(submission)} does not match test row count {len(test_df)}"
        )

    if submission["SMILES"].isna().any() or (submission["SMILES"].astype(str).str.strip() == "").any():
        raise ValueError("Submission contains empty SMILES values.")

    if submission["Molecule Name"].isna().any() or (
        submission["Molecule Name"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("Submission contains empty Molecule Name values.")

    if submission["pEC50"].isna().any():
        raise ValueError("Submission contains empty pEC50 values.")

    return submission


def write_submission(submission_df: pd.DataFrame, output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_file, index=False)


def write_final_report(report: dict, output_path: str | Path) -> None:
    write_json(report, output_path)


def export_best_submission(run_id: str, data_dir: str = "data/pxr_activity") -> dict:
    run_dir = Path("outputs") / run_id
    best_plan_path = run_dir / "best_plan.json"
    best_plan = load_json(best_plan_path)

    if "metrics_path" in best_plan:
        predictions_path = Path(best_plan["metrics_path"]).parent / "test_predictions.csv"
    else:
        predictions_path = run_dir / "plans" / best_plan["plan_id"] / "test_predictions.csv"

    test_path = Path(data_dir) / "test.csv"
    test_df = pd.read_csv(test_path)
    pred_df = pd.read_csv(predictions_path)

    submission_df = build_submission(test_df, pred_df)

    submission_path = run_dir / "submission.csv"
    final_report_path = run_dir / "final_report.json"

    write_submission(submission_df, submission_path)

    final_report = {
        "run_id": run_id,
        "challenge_name": "openadmet/pxr-challenge",
        "track": "activity",
        "submission_path": str(submission_path),
        "best_plan": best_plan,
        "test_predictions_path": str(predictions_path),
        "n_submission_rows": int(len(submission_df)),
        "submission_columns": list(submission_df.columns),
    }
    write_final_report(final_report, final_report_path)

    return final_report


def export_submission(
    predictions_path: str | Path,
    run_dir: str | Path,
    source_label: str,
    data_dir: str = "data/pxr_activity",
    extra_manifest: dict | None = None,
) -> dict:
    """Build & validate ``submission.csv`` from any predictions CSV (e.g. the
    ensemble's) and write a reproducibility manifest alongside it.

    This is the M2 export path: it does not assume a single ``best_plan`` — it
    takes whichever predictions we chose to submit and records exactly how they
    were produced.
    """
    run_dir = Path(run_dir)
    test_df = pd.read_csv(Path(data_dir) / "test.csv")
    pred_df = pd.read_csv(predictions_path)

    submission_df = build_submission(test_df, pred_df)
    submission_path = run_dir / "submission.csv"
    write_submission(submission_df, submission_path)

    manifest = build_manifest(
        run_dir,
        source_label,
        extra={
            "predictions_path": str(predictions_path),
            "submission_path": str(submission_path),
            "n_submission_rows": int(len(submission_df)),
            "submission_columns": list(submission_df.columns),
            **(extra_manifest or {}),
        },
    )
    write_json(manifest, run_dir / "submission_manifest.json")
    return manifest


if __name__ == "__main__":
    report = export_best_submission(run_id=RUN_ID)
    print("submission exported:")
    print(f"  run_id: {report['run_id']}")
    print(f"  submission_path: {report['submission_path']}")
    print(f"  best_plan_id: {report['best_plan']['plan_id']}")
