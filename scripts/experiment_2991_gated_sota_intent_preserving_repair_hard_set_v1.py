#!/usr/bin/env python3
"""Run Exp 2991 gated SOTA intent-preserving hard-set repair.

The implementation follows the repository experiment-template convention:
resolve repo-local SOTA GGUFs first, write a terminal JSON deliverable under
``results/``, and never promote smoke-only evidence as a headline result.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot.eval.gated_sota_intent_preserving_repair_hard_set import main


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
