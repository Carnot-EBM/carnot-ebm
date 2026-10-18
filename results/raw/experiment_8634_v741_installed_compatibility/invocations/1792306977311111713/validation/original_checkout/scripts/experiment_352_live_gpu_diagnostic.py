#!/usr/bin/env python3
"""Experiment 352: Live GPU diagnostic — identify the exact failure layer.

**Researcher summary:**
    Experiments 340, 341, 346, and 347 all ran in simulated mode despite
    CARNOT_FORCE_LIVE=1 being set.  Both RTX 3090s were idle for two full
    milestones.  This experiment runs the live GPU diagnostic to identify
    exactly WHICH layer is failing:

        Layer 1 — cuda_visible:     nvidia-smi accessible and returning GPUs
        Layer 2 — torch_cuda:       torch.cuda.is_available() returns True
        Layer 3 — model_loadable:   AutoTokenizer can load each model within 30s

    The diagnostic never raises and is CI-safe.  If running in a real GPU
    environment, it will identify the first failed layer and report it in the
    artifact.  Future benchmark experiments can check this artifact before
    attempting live inference.

**Output:** results/experiment_352_live_gpu_diagnostic.json

Spec: REQ-INFRA-014, SCENARIO-INFRA-014, SCENARIO-INFRA-015
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Bootstrap: ensure repo root is on sys.path so scripts.* and carnot.* resolve.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.live_gpu_diagnostic import diagnose_live_gpu  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 352
EXP_TITLE = "Live GPU diagnostic — identify failure layer"
DELIVERABLE = "results/experiment_352_live_gpu_diagnostic.json"

MODEL_IDS = [
    "Qwen/Qwen3.5-0.8B",
    "google/gemma-4-E4B-it",
]


def main() -> None:
    """Run Experiment 352: live GPU diagnostic."""
    tmpl = ExperimentTemplate(EXP_ID, EXP_TITLE, DELIVERABLE)
    tmpl.setup()

    _log.info("Exp 352: Running live GPU diagnostic for models: %s", MODEL_IDS)

    # Run the full diagnostic — never raises, always returns a result.
    diag = diagnose_live_gpu(MODEL_IDS)

    # Build check lists for the artifact.
    checks_passed: list[str] = []
    checks_failed: list[str] = []

    if diag.cuda_visible:
        checks_passed.append("cuda_visible")
    else:
        checks_failed.append("cuda_visible")

    if diag.torch_available:
        checks_passed.append("torch_cuda")
    else:
        checks_failed.append("torch_cuda")

    if diag.model_loadable:
        checks_passed.append("model_loadable")
    else:
        checks_failed.append("model_loadable")

    _log.info("Diagnostic result: is_live_capable=%s", diag.is_live_capable)
    _log.info("  cuda_visible:          %s", diag.cuda_visible)
    _log.info("  torch_cuda:            %s", diag.torch_available)
    _log.info("  carnot_force_live_set: %s", diag.carnot_force_live_set)
    _log.info("  model_loadable:        %s", diag.model_loadable)
    if diag.failure_reason:
        _log.info("  failure_reason: %s", diag.failure_reason)

    status = "success" if diag.is_live_capable else "blocked"

    artifact = tmpl.build_result(
        {
            "schema": "carnot.live_gpu_diagnostic.v1",
            "is_live_capable": diag.is_live_capable,
            "failure_reason": diag.failure_reason,
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
            "cuda_visible": diag.cuda_visible,
            "torch_available": diag.torch_available,
            "carnot_force_live_set": diag.carnot_force_live_set,
            "model_loadable": diag.model_loadable,
            "models_checked": MODEL_IDS,
        },
        status=status,
    )

    output_path = _REPO_ROOT / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Exp 352: artifact written to %s", output_path)

    if not diag.is_live_capable:
        _log.error(
            "Exp 352: Live GPU NOT available. Failure: %s\n"
            "  Fix this before running Exps 340/341/346/347 with CARNOT_FORCE_LIVE=1.",
            diag.failure_reason,
        )
    else:
        _log.info("Exp 352: All diagnostic checks passed — live GPU inference is ready.")


if __name__ == "__main__":
    main()
