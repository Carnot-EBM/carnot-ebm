"""REQ-REPORT-8464: complete disk service timing invokes no language model."""

import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
print("[exp8464] start inference_substrate_class=no_model_load current_llm_calls=0", flush=True)
_root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_root / "python"), str(_root)]

from carnot.reporting.persistence_localization_runner_8464 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
