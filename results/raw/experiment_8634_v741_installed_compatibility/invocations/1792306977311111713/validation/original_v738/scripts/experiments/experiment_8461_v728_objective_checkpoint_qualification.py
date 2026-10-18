"""REQ-REPORT-8461: constructed objective qualification invokes no language model."""

import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
print("[exp8461] start inference_substrate_class=no_model_load current_llm_calls=0", flush=True)
_root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_root / "python"), str(_root)]

from carnot.reporting.objective_checkpoint_runner_8461 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
