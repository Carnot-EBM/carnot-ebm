#!/usr/bin/env python3
"""Run Exp 3003 gated SOTA repair metamorphic false-accept rerun.

The script follows the repository experiment-template convention: resolve the
repo root, require local SOTA/cache provenance before promotion, write a
terminal JSON artifact under ``results/``, and never promote smoke-only model
evidence as a headline repair result.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot.eval.gated_sota_repair_metamorphic_false_accept_rerun import main


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
