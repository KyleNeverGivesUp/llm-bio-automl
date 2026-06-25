"""Score a broad-trained submission against the Set-1 judge (the 253 analog labels).

This is the honest read on where we stand — broad scaffold-CV is NOT a reliable
proxy (it said 0.543; the judge says 0.633). Point it at any 513-row prediction
file (submission.csv or a plan's test_predictions.csv).

Usage:
    uv run python -m scripts.judge outputs/m1_menu/ensemble/test_predictions.csv
    uv run python -m scripts.judge outputs/m1_menu/submission.csv outputs/m1_menu/plans/*/test_predictions.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.analog_judge import judge_csv


def main(argv: list[str]) -> None:
    paths = argv or ["outputs/m1_menu/ensemble/test_predictions.csv"]
    print(f"{'RAE':>7}  {'MAE':>6}  {'R2':>6}  n   file")
    rows = []
    for p in paths:
        try:
            s = judge_csv(p)
            rows.append((s["rae"], p, s))
        except Exception as e:  # keep going across a glob of files
            print(f"{'ERR':>7}  {'':>6}  {'':>6}      {p}  ({type(e).__name__}: {e})")
    for rae, p, s in sorted(rows, key=lambda r: r[0]):
        print(f"{s['rae']:7.4f}  {s['mae']:6.4f}  {s['r2']:6.3f}  {s['n_judged']}  {Path(p).as_posix()}")


if __name__ == "__main__":
    main(sys.argv[1:])
