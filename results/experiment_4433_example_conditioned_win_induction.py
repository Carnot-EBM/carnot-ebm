"""Runner for Exp 4433 example-conditioned held-out win induction."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from carnot import experiment_4433_example_conditioned_win_induction as exp  # noqa: E402


def main(root: Path = ROOT) -> int:
    exp.run(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
