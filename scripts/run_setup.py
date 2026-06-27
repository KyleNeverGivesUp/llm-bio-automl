"""Run the LLM-driven Setup agent and print the setup report (for verification).

  uv run python scripts/run_setup.py
  uv run python scripts/run_setup.py --brief docs/CHALLENGE_BRIEF.md --data-dir data/pxr_activity

The LLM reads the competition description + the data-dir evidence and INFERS the task schema;
deterministic code then validates it. Output: outputs/setup/setup_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent.setup_agent import SetupAgent   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", default="docs/CHALLENGE_BRIEF.md", help="competition description the LLM reads")
    ap.add_argument("--data-dir", default="data/pxr_activity")
    args = ap.parse_args()

    instruction = Path(args.brief).read_text(encoding="utf-8")
    out_dir = Path("outputs/setup"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "setup_report.json"

    agent = SetupAgent()
    report = agent.run(instruction=instruction, data_dir=args.data_dir, out_path=out_path)

    print(f"\n=== SETUP REPORT (source = {report.get('source')}, status = {report.get('status')}) ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwritten to: {out_path}")


if __name__ == "__main__":
    main()
