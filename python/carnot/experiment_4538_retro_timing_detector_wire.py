"""Run Exp 4538 retro timing-data detector wiring artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
import sys


if __package__ in {None, ""}:  # pragma: no cover - direct script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot.reporting.retro_timing_detector_wire_4538 import run  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    output_path = run(REPO_ROOT)
    print(json.dumps(json.loads(output_path.read_text(encoding="utf-8")), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by requested command
    raise SystemExit(main())
