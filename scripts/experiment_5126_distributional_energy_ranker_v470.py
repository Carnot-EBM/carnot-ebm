"""CLI wrapper for Exp 5126 distributional-energy ranker."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct execution guard
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_5126_distributional_energy_ranker_v470 as experiment  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return experiment.main(argv)


if __name__ == "__main__":  # pragma: no cover - direct execution guard
    raise SystemExit(main())
