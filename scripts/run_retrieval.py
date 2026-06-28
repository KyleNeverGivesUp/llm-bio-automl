"""Run the task-guided LLM Retrieval agent and print the result (for verification).

  uv run python scripts/run_setup.py        # first, produces outputs/setup/setup_report.json
  uv run python scripts/run_retrieval.py     # then this reads it and searches for models

The LLM plans the search FROM the task, runs a live HF search, and ranks candidates with a
frozen/finetune mode. Output: outputs/retrieval/retrieval_result.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent.retrieval_agent import RetrievalAgent   # noqa: E402


def main() -> None:
    setup_path = Path("outputs/setup/setup_report.json")
    if not setup_path.exists():
        print("run scripts/run_setup.py first (need outputs/setup/setup_report.json)")
        return
    setup_report = json.loads(setup_path.read_text(encoding="utf-8"))

    out_dir = Path("outputs/retrieval"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_result.json"

    result = RetrievalAgent().run(setup_report, top_k=12, out_path=out_path)
    print(f"!!!!! {result}")

    sat = result["satisfaction"]
    print(f"\n=== RETRIEVAL (source = {result['source']}) ===")
    print(f"① search plan (LLM): families={result['search_plan'].get('families')}")
    print(f"                     queries={result['search_plan'].get('queries')}")
    print(f"③ satisfied? {sat.get('satisfied')} [{sat.get('source')}] — {sat.get('reason','')[:70]}")
    print(f"④ went online: {result['went_online']}; added to local library: {result['n_added_to_library']}")
    print(f"⑤ selected ({result['n_candidates']} candidates considered):")
    for s in result["selected"]:
        print(f"   - {s.get('ref'):<42} family={s.get('family','?'):<8} mode={s.get('mode'):<9} {s.get('reason','')[:55]}")
    print(f"\nwritten to: {out_path}   |   local library: skills/models/registry.json")


if __name__ == "__main__":
    main()
