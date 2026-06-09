import sys
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT / "python"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    from carnot.phase3.p01_trained_energy_reranker import _Verifiers
    verifiers = _Verifiers()
    _ = verifiers.ising.energy("2 + 2 = 4")
    print("Substrate loaded successfully.")
except Exception as e:
    print(f"Failed to load substrate: {e}")
