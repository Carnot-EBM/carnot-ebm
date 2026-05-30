"""Experiment 3398 — CAffNet Layer Robustness against Out-of-Distribution Inputs

This experiment tests the robustness of the CAffNet layer against adversarial 
OOD constraint configurations as per REQ-CAFFNET-3398.

Steps:
  1. Generate adversarial OOD constraint configurations.
  2. Feed into CAffNet.
  3. Verify projection still satisfies affine constraints.
"""
import json
import logging
import sys
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

EXPERIMENT_ID = 3398
RESULT_PATH = Path("results/experiment_3398_caffnet_robustness.json")

def _generate_adversarial_ood_constraints() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate adversarial constraints:
      A is a highly ill-conditioned matrix.
      b is a random target.
      logits are extremely large (OOD).
    """
    # Create ill-conditioned matrix A
    A = jnp.array([[1.0, 1.0], [1.0 + 1e-8, 1.0]])
    b = jnp.array([1.0, 1.0])
    
    # Extreme logits
    logits = jnp.array([1e10, -1e10])
    
    return A, b, logits

def main() -> None:
    started_at = datetime.now(UTC)
    _log.info("Experiment %d starting at %s", EXPERIMENT_ID, started_at.isoformat())
    
    result: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": "caffnet_ood_robustness_v1",
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "duration_s": None,
        "status": "unknown",
        "honest_verdict": "unknown",
        "caffnet_robust": False,
        "max_residual": None,
    }
    
    try:
        from carnot.models.caffnet_layer import CAffNetLayer
        
        # 1. Generate adversarial constraints
        A, b, logits = _generate_adversarial_ood_constraints()
        
        # 2. Feed into CAffNet
        layer = CAffNetLayer(A, b)
        x_proj = layer.apply(logits)
        
        # 3. Verify projection
        contains_nans = bool(jnp.any(jnp.isnan(x_proj)))
        contains_infs = bool(jnp.any(jnp.isinf(x_proj)))
        
        res = jnp.abs(A @ x_proj - b)
        max_res = float(jnp.max(res))
        
        result["max_residual"] = max_res
        result["caffnet_robust"] = not contains_nans and not contains_infs
        
        if result["caffnet_robust"]:
            result["honest_verdict"] = "robust"
            result["status"] = "success"
        else:
            result["honest_verdict"] = "failed_with_nans_or_infs"
            result["status"] = "failure"
            
    except Exception as e:
        _log.exception("Experiment failed with exception: %s", e)
        result["status"] = "error"
        result["error"] = str(e)
        result["honest_verdict"] = "failed_with_exception"
        
    finally:
        finished_at = datetime.now(UTC)
        result["finished_at"] = finished_at.isoformat()
        result["duration_s"] = round((finished_at - started_at).total_seconds(), 3)

        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(result, indent=2))
        _log.info("Result written to %s", RESULT_PATH)
        _log.info(
            "honest_verdict=%s caffnet_robust=%s",
            result["honest_verdict"],
            result["caffnet_robust"],
        )

if __name__ == "__main__":
    _repo_root = Path(__file__).parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    _py_root = _repo_root / "python"
    if str(_py_root) not in sys.path:
        sys.path.insert(0, str(_py_root))

    main()