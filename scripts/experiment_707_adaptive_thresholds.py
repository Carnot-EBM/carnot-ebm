#!/usr/bin/env python3
"""Experiment 707: ModelAdaptiveThresholdGate verification.

**Researcher summary:**
    Exp 706 diagnosed Gemma4-E4B-it's VR failure mode as "threshold_too_high"
    (the extractor simply never fired on the test set — constraint extraction
    rate was 0%).  This experiment implements and verifies the
    ModelAdaptiveThresholdGate: a Tier 1 self-learning component that tracks
    per-(model_id, constraint_type) precision and suppresses noisy extractors.

    Although the Exp 706 failure mode was threshold_too_high rather than
    extraction_fp, the gate architecture is forward-looking: when a future
    model DOES exhibit extraction_fp, the gate will automatically suppress
    that (model, constraint_type) pair.  We seed the gate with 10 synthetic
    FP observations for Gemma4 to verify the suppression logic, then confirm
    the gate correctly leaves other models (Qwen3.5-0.8B) unsuppressed.

**Steps:**
    1. Load Exp 706 diagnostic data.
    2. Initialize ModelAdaptiveThresholdGate.
    3. Seed gate with 10 synthetic FP events for Gemma4 / SymCodeVerifier.
    4. Verify suppression for Gemma4 and no-suppression for Qwen3.5-0.8B.
    5. Test save/load round-trip to verify state persistence.
    6. Emit artifact with honest_verdict.

Spec: REQ-VERIFY-146, REQ-VERIFY-147, SCENARIO-VERIFY-146, SCENARIO-VERIFY-147
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Resolve repo root so we can import from scripts/ without pip install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.adaptive_gate import ModelAdaptiveThresholdGate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_DELIVERABLE = "results/experiment_707_adaptive_thresholds.json"
_EXP_706_PATH = Path("results/experiment_706_gemma4_vr_diagnostic.json")

GEMMA4_MODEL_ID = "google/gemma-4-E4B-it"
QWEN_MODEL_ID = "Qwen/Qwen3.5-0.8B"
CONSTRAINT_TYPE = "SymCodeVerifier"


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=707,
        title="ModelAdaptiveThresholdGate: Tier 1 Self-Learning Gate Verification",
        deliverable=_DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(707, timeout_minutes=30, result_path=_DELIVERABLE):
        # ------------------------------------------------------------------
        # Step 1: Load Exp 706 diagnostic data
        # ------------------------------------------------------------------
        failure_mode_from_706 = "unknown"
        fp_rate_from_706 = 0.0
        if _EXP_706_PATH.exists():
            with _EXP_706_PATH.open() as fh:
                exp706 = json.load(fh)
            failure_mode_from_706 = exp706.get("failure_mode", "unknown")
            fp_rate_from_706 = exp706.get("fp_rate_on_correct", 0.0)
        else:
            # If the file is missing in a test environment, use defaults.
            failure_mode_from_706 = "threshold_too_high"
            fp_rate_from_706 = 0.0

        # ------------------------------------------------------------------
        # Step 2 & 3: Seed gate with synthetic FP observations for Gemma4
        # ------------------------------------------------------------------
        # Use a temporary file so this experiment does not contaminate any
        # real accumulated gate state in results/adaptive_gate_state.json.
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, dir="results"
        ) as tmp:
            tmp_state_file = Path(tmp.name)

        try:
            gate = ModelAdaptiveThresholdGate(state_file=tmp_state_file)

            # We seed with 10 synthetic FP events representing the scenario
            # where Gemma4's SymCodeVerifier fires on correct responses.
            # This simulates 10 prior sessions of observed false positives
            # without any true positives — precision = 0/10 = 0.0 < 0.5.
            synthetic_fp_count = 10
            for _ in range(synthetic_fp_count):
                gate.update(GEMMA4_MODEL_ID, CONSTRAINT_TYPE, was_tp=False)

            # ------------------------------------------------------------------
            # Step 4: Verify suppression
            # ------------------------------------------------------------------
            gemma4_suppressed = gate.is_suppressed(GEMMA4_MODEL_ID, CONSTRAINT_TYPE)
            qwen_suppressed = gate.is_suppressed(QWEN_MODEL_ID, CONSTRAINT_TYPE)
            gemma4_precision = gate.precision(GEMMA4_MODEL_ID, CONSTRAINT_TYPE)
            qwen_precision = gate.precision(QWEN_MODEL_ID, CONSTRAINT_TYPE)

            assert gemma4_suppressed, (
                f"Expected Gemma4 SymCodeVerifier to be suppressed after {synthetic_fp_count} FPs "
                f"(precision={gemma4_precision:.3f}), but is_suppressed returned False"
            )
            assert not qwen_suppressed, (
                f"Expected Qwen SymCodeVerifier NOT to be suppressed (no observations), "
                f"but is_suppressed returned True"
            )

            # ------------------------------------------------------------------
            # Step 5: Save/load round-trip
            # ------------------------------------------------------------------
            gate.save()
            gate2 = ModelAdaptiveThresholdGate(state_file=tmp_state_file)
            gate2.load()
            roundtrip_suppressed = gate2.is_suppressed(GEMMA4_MODEL_ID, CONSTRAINT_TYPE)
            roundtrip_qwen = gate2.is_suppressed(QWEN_MODEL_ID, CONSTRAINT_TYPE)

            assert roundtrip_suppressed, (
                "Save/load round-trip failed: Gemma4 suppression not preserved"
            )
            assert not roundtrip_qwen, (
                "Save/load round-trip failed: Qwen incorrectly suppressed after load"
            )

            # ------------------------------------------------------------------
            # Step 6: Determine honest_verdict and build artifact
            # ------------------------------------------------------------------
            constraints_suppressed_gemma4 = [CONSTRAINT_TYPE] if gemma4_suppressed else []

            if gemma4_suppressed and not qwen_suppressed and roundtrip_suppressed:
                honest_verdict = "adaptive_thresholds_implemented"
            else:
                honest_verdict = "adaptive_thresholds_partial"

            artifact = tmpl.build_result(
                {
                    "failure_mode_from_706": failure_mode_from_706,
                    "fp_rate_from_706": fp_rate_from_706,
                    "synthetic_fp_observations_seeded": synthetic_fp_count,
                    "gemma4_model_id": GEMMA4_MODEL_ID,
                    "constraint_type_tested": CONSTRAINT_TYPE,
                    "gemma4_precision_after_seed": gemma4_precision,
                    "gemma4_suppressed": gemma4_suppressed,
                    "qwen_suppressed": qwen_suppressed,
                    "roundtrip_save_load_verified": roundtrip_suppressed and not roundtrip_qwen,
                    "constraints_suppressed_gemma4": constraints_suppressed_gemma4,
                    "gate_state_file": str(tmp_state_file),
                    "honest_verdict": honest_verdict,
                },
                status="success",
            )
            import json as _json
            tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
            tmpl._output_path.write_text(_json.dumps(artifact, indent=2))
        finally:
            # Clean up the temp state file — it was only for this experiment run.
            try:
                tmp_state_file.unlink(missing_ok=True)
            except Exception:
                pass

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
