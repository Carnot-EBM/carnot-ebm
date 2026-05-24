#!/usr/bin/env python3
"""Run Exp 3016 SOTA repair rerun with the acceptance controller.

The heavy lifting lives in ``carnot.eval`` so tests can exercise the artifact
contract without importing this thin CLI wrapper.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot.eval.sota_repair_rerun_with_acceptance_controller import main


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
