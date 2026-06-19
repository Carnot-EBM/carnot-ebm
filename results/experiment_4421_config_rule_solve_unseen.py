"""Runner for Exp 4421 unseen config-rule solve."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from carnot import experiment_4421_config_rule_solve_unseen as exp


def main(root: Path = ROOT) -> int:
    exp.run(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
