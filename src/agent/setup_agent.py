"""Setup Agent (LLM-driven) — env/data intake, the first node (mirrors AIBuildAI's Setup).

Given the **competition description** + the data directory, an LLM reads the brief and the actual
CSV headers and INFERS the task schema: which file is train/test/submission, which column is the
SMILES / target / id, the metric, and row counts. It writes a `setup_report.json` (data paths +
inferred schema + status) that downstream stages read.

Boundary (our principle "LLM proposes, deterministic code verifies"):
- the LLM only INFERS + REPORTS; deterministic code then VALIDATES every claim against the real
  files (columns exist, SMILES parse with RDKit) and refuses to pass a setup it can't verify;
- it NEVER touches pip/conda/the environment (read-only import checks only);
- it NEVER selects the held-out judge (`phase1_unblinded.csv`, the Set-1 labels) as train/test —
  a hard deterministic guard, so the LLM cannot cause a leak.

Falls back to the known PXR schema if the LLM is unavailable, like the other agents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.agent.LLM_base import LLMJsonAgent

# The Set-1 judge — must never be used as train/test (leak guard). Kept out of the LLM's candidates.
JUDGE_FILE = "phase1_unblinded.csv"

# Validated fallback schema (the real PXR layout) if the LLM is down / returns junk.
_FALLBACK = {
    "task": {"name": "OpenADMET PXR Activity", "type": "regression", "domain": "molecular property"},
    "data": {"train_file": "train.csv", "test_file": "test.csv", "submission_file": "sample_submission.csv"},
    "schema": {"smiles_col": "SMILES", "target_col": "pEC50", "id_col": "Molecule Name"},
    "metric": "RAE",
}


class SetupAgent(LLMJsonAgent):
    name = "setup"
    source: str = "unknown"   # "llm" | "fallback"

    def run(self, instruction: str, data_dir: str | Path, out_path: str | Path | None = None) -> dict:
        data_dir = Path(data_dir)
        evidence = self._gather_evidence(data_dir)
        try:
            inferred = self._llm_infer(instruction, evidence)
            self.source = "llm"
        except Exception as e:  # noqa: BLE001
            print(f"[setup] LLM infer failed ({e}); using validated fallback")
            inferred = dict(_FALLBACK)
            self.source = "fallback"

        report = self._validate(inferred, data_dir)
        report["source"] = self.source
        if out_path:
            Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    # --- evidence: list CSVs + their columns/row-count (+ 2 sample rows, EXCEPT the judge) ----- #
    def _gather_evidence(self, data_dir: Path) -> list[dict]:
        ev = []
        for csv in sorted(data_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv, nrows=2)
                n = sum(1 for _ in open(csv, encoding="utf-8")) - 1  # row count w/o header
                item = {"file": csv.name, "columns": list(df.columns), "n_rows": n}
                if csv.name != JUDGE_FILE:                       # never expose judge label values
                    item["sample_rows"] = df.head(2).to_dict("records")
                else:
                    item["note"] = "HELD-OUT JUDGE (Set-1 labels) — never use as train/test"
                ev.append(item)
            except Exception:  # noqa: BLE001
                ev.append({"file": csv.name, "error": "unreadable"})
        return ev

    def _llm_infer(self, instruction: str, evidence: list[dict]) -> dict:
        system = (
            "You are the SETUP agent of an AutoML pipeline. Read the competition description and the "
            "data-directory evidence (each CSV's columns, row count, sample rows), then INFER the task "
            "setup. Do NOT write code or install anything. One file is a HELD-OUT JUDGE — never pick it "
            "as train/test. Reply with ONLY this JSON:\n"
            '{"task":{"name":..,"type":"regression|classification","domain":..,"summary":..},'
            '"data":{"train_file":..,"test_file":..,"submission_file":..},'
            '"schema":{"smiles_col":..,"target_col":..,"id_col":..},'
            '"metric":..,"reason":..}'
        )
        user = (
            f"Competition description:\n{instruction[:7000]}\n\n"   # long enough to include the Scoring section
            f"Data directory evidence (JSON):\n{json.dumps(evidence, default=str)[:4000]}\n\n"
            "Identify: the train file (has the target column), the test file (no target — blind), the "
            "submission template, and the SMILES / target / id column names. For the metric, use the "
            "competition's PRIMARY ranking metric exactly as named in the Scoring section (do NOT assume "
            "a default like RMSE). Return ONLY the JSON object."
        )
        out = self.call_json(system, user)
        if not isinstance(out, dict) or "schema" not in out or "data" not in out:
            raise ValueError("LLM setup did not return the expected JSON shape")
        return out

    # --- deterministic validation: every LLM claim checked against the real files --------------- #
    def _validate(self, inferred: dict, data_dir: Path) -> dict:
        checks: dict = {}
        data = inferred.get("data", {})
        schema = inferred.get("schema", {})
        train_f, test_f = data.get("train_file"), data.get("test_file")

        # leak guard: train/test may never be the judge
        if JUDGE_FILE in (train_f, test_f):
            return {**inferred, "status": "failed",
                    "validation": {"leak_guard": f"refused {JUDGE_FILE} as train/test"}}

        def cols(name):
            p = data_dir / name if name else None
            return list(pd.read_csv(p, nrows=1).columns) if p and p.exists() else None

        train_cols, test_cols = cols(train_f), cols(test_f)
        checks["train_exists"] = train_cols is not None
        checks["test_exists"] = test_cols is not None

        smi, tgt, idc = schema.get("smiles_col"), schema.get("target_col"), schema.get("id_col")
        checks["smiles_col_in_train"] = bool(train_cols and smi in train_cols)
        checks["target_col_in_train"] = bool(train_cols and tgt in train_cols)
        checks["target_absent_in_test"] = bool(test_cols is not None and tgt not in test_cols)  # blind-test sanity

        # SMILES actually parse?
        parse_rate = None
        if checks["train_exists"] and checks["smiles_col_in_train"]:
            from rdkit import Chem
            s = pd.read_csv(data_dir / train_f, usecols=[smi]).iloc[:200, 0].astype(str)
            parse_rate = round(sum(Chem.MolFromSmiles(x) is not None for x in s) / len(s), 3)
        checks["smiles_parse_rate"] = parse_rate

        n_train = pd.read_csv(data_dir / train_f).shape[0] if checks["train_exists"] else None
        n_test = pd.read_csv(data_dir / test_f).shape[0] if checks["test_exists"] else None

        ok = (checks["train_exists"] and checks["test_exists"] and checks["smiles_col_in_train"]
              and checks["target_col_in_train"] and (parse_rate or 0) >= 0.95)

        # read-only environment status (no install — just whether key libs import)
        env = {}
        for lib in ("rdkit", "pandas", "numpy", "sklearn"):
            try:
                __import__(lib); env[lib] = True
            except Exception:  # noqa: BLE001
                env[lib] = False

        return {
            "task": inferred.get("task", {}),
            "data": {"data_dir": str(data_dir), **data, "n_train": n_train, "n_test": n_test},
            "schema": schema,
            "metric": inferred.get("metric"),
            "held_out_judge": JUDGE_FILE,
            "environment": env,
            "validation": checks,
            "status": "ok" if ok else "failed",
            "reason": inferred.get("reason", ""),
        }
