from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from carnot.experiment_4330_arc_adapter_free_discovery_sweep_shallow_tail import main


if __name__ == "__main__":
    main()
