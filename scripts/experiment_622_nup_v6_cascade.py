#!/usr/bin/env python3
"""Experiment 622: NUP v6 Tier 0c Cascade Wire-In.

**Researcher summary:**
    NUP Probe v6 (Exp 608, AUC=0.964) was validated as Tier 0c-ready but was NOT wired
    into VerifyRepairPipeline.  This experiment deploys it: a NUPProbeV4 instance is
    constructed, passed to VerifyRepairPipeline via the new nup_probe parameter, and
    exercised on 100 synthetic responses (50 'correct', 50 'incorrect').

    Deliverable fields:
        nup_v6_wired       — confirms the pipeline accepted the probe without error
        n_tested           — 100 (50 correct + 50 incorrect)
        cascade_latency_ms — mean wall-clock ms per verify() call with probe active
        nup_skip_rate      — fraction of calls where Tier 0c short-circuited
        latency_ok         — True when cascade_latency_ms < 5.0 (REQ-VERIFY-147)
        honest_verdict     — 'nup_deployed_latency_ok' or 'nup_deployed_latency_high'

Spec: REQ-VERIFY-146, REQ-VERIFY-147,
      SCENARIO-VERIFY-177, SCENARIO-VERIFY-178, SCENARIO-VERIFY-179
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# apply_env_autofix MUST be called before any JAX/GPU import
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

EXP_ID = 622
EXP_TITLE = "NUP v6 Tier 0c Cascade Wire-In"
DELIVERABLE = "results/experiment_622_nup_v6_cascade.json"

# Synthetic corpus: 50 well-constrained (correct) steps and 50 free-form (incorrect) steps.
# We do NOT need live GPU data — this is a latency + integration smoke-test.
_CORRECT_RESPONSES = [
    f"2 + {i} = {2 + i}. Therefore the total is {2 + i}." for i in range(50)
]
_INCORRECT_RESPONSES = [
    (
        f"The quantum flux of particle {i} interacts with the holomorphic manifold "
        f"suggesting a non-trivial topology shift of order {i * 3.14:.2f} radians, "
        f"which fundamentally alters the eigenvalue decomposition of the Hamiltonian."
    )
    for i in range(50)
]
_ALL_RESPONSES = _CORRECT_RESPONSES + _INCORRECT_RESPONSES  # 100 total


def main() -> None:
    """Wire NUP Probe v6 into VerifyRepairPipeline and measure cascade latency."""
    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=25)
    _watchdog.start()

    tmpl = ExperimentTemplate(
        EXP_ID,
        EXP_TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Load NUPProbeV4 — use Exp 608 saved weights if present, else random init.
    # Random-init weights are sufficient for the latency / integration test
    # because we are NOT measuring classification accuracy here.
    # ------------------------------------------------------------------
    from carnot.pipeline.nup_probe_v4 import NUPProbeV4  # noqa: PLC0415

    v6_weights_path = _REPO_ROOT / "results" / "nup_probe_v6.safetensors"
    nup_probe = NUPProbeV4(energy_dim=32, random_seed=42)

    if v6_weights_path.exists():
        _log.info("Loading Exp 608 v6 weights from %s", v6_weights_path)
        try:
            import safetensors.numpy as st  # noqa: PLC0415
            tensors = st.load_file(str(v6_weights_path))
            if "weights" in tensors and "bias" in tensors:
                nup_probe._weights = list(tensors["weights"].flatten().tolist())
                nup_probe._bias = float(tensors["bias"].flatten()[0])
                _log.info("v6 weights loaded successfully (energy_dim=%d)", len(nup_probe._weights))
            else:
                _log.warning("Unexpected safetensors keys; using random-init weights")
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not load v6 weights (%s); using random-init", exc)
    else:
        _log.info("v6 weights not found at %s; using random-init for latency test", v6_weights_path)

    # ------------------------------------------------------------------
    # Construct VerifyRepairPipeline with nup_probe wired in (REQ-VERIFY-146-1).
    # model=None → verify-only mode (no LLM needed for this test).
    # ------------------------------------------------------------------
    from carnot.pipeline.verify_repair import VerifyRepairPipeline  # noqa: PLC0415

    pipeline = VerifyRepairPipeline(
        model=None,
        nup_probe=nup_probe,
        nup_probe_threshold=0.5,
    )
    _log.info("VerifyRepairPipeline constructed with NUP Probe v6 as Tier 0c")

    # ------------------------------------------------------------------
    # Run 100 verify() calls, measuring wall-clock latency per call.
    # ------------------------------------------------------------------
    latencies_ms: list[float] = []
    n_skipped = 0  # calls where Tier 0c short-circuited

    for resp in _ALL_RESPONSES:
        t0 = time.perf_counter()
        result = pipeline.verify(question="What is the answer?", response=resp, domain=None)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        if getattr(result, "mode", None) == "NUP_PROBE_FAST_PATH":
            n_skipped += 1

    n_tested = len(_ALL_RESPONSES)
    cascade_latency_ms = sum(latencies_ms) / len(latencies_ms)
    nup_skip_rate = n_skipped / n_tested
    latency_ok = cascade_latency_ms < 5.0

    _log.info(
        "Results: n_tested=%d, cascade_latency_ms=%.3f, nup_skip_rate=%.3f, latency_ok=%s",
        n_tested,
        cascade_latency_ms,
        nup_skip_rate,
        latency_ok,
    )

    honest_verdict = "nup_deployed_latency_ok" if latency_ok else "nup_deployed_latency_high"

    artifact = tmpl.build_result(
        {
            "nup_v6_wired": True,
            "n_tested": n_tested,
            "cascade_latency_ms": cascade_latency_ms,
            "nup_skip_rate": nup_skip_rate,
            "latency_ok": latency_ok,
            "honest_verdict": honest_verdict,
        },
        schema="carnot.nup_v6_cascade.v1",
        status="success",
    )
    writer = AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE))
    writer.write(artifact)
    _log.info("Artifact written to %s", DELIVERABLE)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
