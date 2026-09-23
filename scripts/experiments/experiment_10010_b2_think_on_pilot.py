#!/usr/bin/env python
"""Entry point for Experiment 10010 (REQ-ARC-WMTE-10010): the B2 think-ON induction pilot.

All logic lives in `python/carnot/experiment_10010_b2_think_on_pilot.py`.

    # CPU only, no server, no GPU; writes only under the scratch directory given.
    python scripts/experiments/experiment_10010_b2_think_on_pilot.py --dry-run --output-dir /tmp/x

    # The GPU run (GPU 1 only). Resumable: finished windows are skipped on restart.
    python scripts/experiments/experiment_10010_b2_think_on_pilot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import from THIS checkout's python/ tree. A worktree run that imported the main
# checkout's copy would measure code other than the code under review.
_PYTHON_DIR = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.experiment_10010_b2_think_on_pilot import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
