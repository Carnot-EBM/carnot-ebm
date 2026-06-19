"""Runner for Exp 4422 glyph-rewrite perception."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from carnot import experiment_4422_glyph_rewrite_perception as exp


def main(root: Path = ROOT) -> int:
    exp.run(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
