"""REQ-REPORT-8490: authenticate the objective adapter before any released-label learning."""

import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
print("[exp8490] start inference_substrate_class=no_model_load current_llm_calls=0", flush=True)
_root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_root / "python"), str(_root)]

from carnot.reporting.brier_online_runner_8490 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
