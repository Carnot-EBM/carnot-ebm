#!/usr/bin/env python3
"""Exp 3396: CAS Diffusion — Compress-Add-Smooth updates for constraint templates.

**Researcher summary:**
    arXiv:2604.00067 proposes the CAS recursion for bounded memory updates.
    This experiment validates that CASConstraintUpdater correctly applies
    CAS steps to ConstraintTemplateLibrary observation banks:

    1. Loads a ConstraintTemplateLibrary with built-in templates.
    2. Applies a sequence of CAS updates with synthetic observations.
    3. Verifies that:
       - Memory stays bounded (no count exceeds max_count).
       - Active templates are correctly maintained after updates.
       - Decay behaves geometrically (SCENARIO-CAS-001).
       - New observations above min_frequency activate templates
         (SCENARIO-CAS-002).

    CPU-only, fully deterministic — no GPU required.

    Output: results/experiment_3396_cas_diffusion.json

Spec: REQ-CAS-001, REQ-CAS-001-1, REQ-CAS-001-2, REQ-CAS-001-3,
      REQ-CAS-001-4, SCENARIO-CAS-001, SCENARIO-CAS-002
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "python"))

from experiment_template import ExperimentTemplate  # noqa: E402

from carnot.pipeline.cas_constraint_update import CASConstraintUpdater  # noqa: E402
from carnot.pipeline.constraint_template_library import (  # noqa: E402
    ConstraintTemplateLibrary,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "qwen3.5-0.8b"
N_CAS_STEPS = 10
COMPRESS_FACTOR = 0.9
SMOOTH_ALPHA = 0.05
SMOOTH_TARGET = 0.0
MAX_COUNT = 100.0
INITIAL_COUNT = 80.0  # set high so decay is clearly observable


def _make_library() -> ConstraintTemplateLibrary:
    lib = ConstraintTemplateLibrary()
    lib.register_builtin_templates()
    return lib


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=3396,
        title="CAS Diffusion — Compress-Add-Smooth updates for constraint templates",
        deliverable="results/experiment_3396_cas_diffusion.json",
        requires_gpu=False,
    )
    tmpl.setup()

    updater = CASConstraintUpdater(
        compress_factor=COMPRESS_FACTOR,
        smooth_alpha=SMOOTH_ALPHA,
        smooth_target=SMOOTH_TARGET,
        max_count=MAX_COUNT,
    )

    # -------------------------------------------------------------------------
    # Step 1: Verify SCENARIO-CAS-001 — decay reduces high counts geometrically
    # -------------------------------------------------------------------------
    decay_lib = _make_library()
    decay_lib._observations[("carry_check", MODEL_ID)] = INITIAL_COUNT

    print(f"Initial count: {INITIAL_COUNT}")
    count_trace = [INITIAL_COUNT]
    for _ in range(N_CAS_STEPS):
        updater.compress(decay_lib)
        count_trace.append(decay_lib._observations[("carry_check", MODEL_ID)])

    final_decay_count = count_trace[-1]
    expected_decay = INITIAL_COUNT * (COMPRESS_FACTOR ** N_CAS_STEPS)
    print(f"After {N_CAS_STEPS} compress steps: {final_decay_count:.4f}")
    print(f"Expected (analytical): {expected_decay:.4f}")
    decay_error = abs(final_decay_count - expected_decay)

    # -------------------------------------------------------------------------
    # Step 2: Verify SCENARIO-CAS-002 — new observations survive above threshold
    # -------------------------------------------------------------------------
    add_lib = _make_library()  # carry_check min_frequency=5
    # Add 10 observations — should stay above min_frequency=5 after one CAS step
    post_add = updater.cas_update(add_lib, {("carry_check", MODEL_ID): 10.0})
    post_add_count = post_add.get(("carry_check", MODEL_ID), 0.0)
    active_after_add = add_lib.get_active_templates(MODEL_ID)
    carry_active = any(t.pattern_key == "carry_check" for t in active_after_add)

    print(f"Post-CAS count for carry_check: {post_add_count:.4f}")
    print(f"carry_check active after CAS add: {carry_active}")

    # -------------------------------------------------------------------------
    # Step 3: Verify memory stays bounded under N_CAS_STEPS with large injections
    # -------------------------------------------------------------------------
    bounded_lib = _make_library()
    # Inject 50 observations each step — combined with compression and smooth cap
    max_observed = 0.0
    for _ in range(N_CAS_STEPS):
        result = updater.cas_update(bounded_lib, {("carry_check", MODEL_ID): 50.0})
        current = result.get(("carry_check", MODEL_ID), 0.0)
        if current > max_observed:
            max_observed = current

    bounded = max_observed <= MAX_COUNT
    print(f"Max observed count over {N_CAS_STEPS} bounded steps: {max_observed:.4f}")
    print(f"All counts bounded by max_count={MAX_COUNT}: {bounded}")

    # -------------------------------------------------------------------------
    # Step 4: Verify a previously active template deactivates after decay
    # -------------------------------------------------------------------------
    deactivate_lib = _make_library()
    deactivate_lib._observations[("carry_check", MODEL_ID)] = 6.0
    fast_updater = CASConstraintUpdater(
        compress_factor=0.5, smooth_alpha=0.0, max_count=MAX_COUNT
    )
    fast_updater.compress(deactivate_lib)
    fast_updater.compress(deactivate_lib)
    deactivated_count = deactivate_lib._observations[("carry_check", MODEL_ID)]
    still_active = any(
        t.pattern_key == "carry_check"
        for t in deactivate_lib.get_active_templates(MODEL_ID)
    )
    print(f"Count after 2× 0.5-compress steps: {deactivated_count:.4f}")
    print(f"carry_check still active: {still_active} (expected False)")

    # -------------------------------------------------------------------------
    # Acceptance gates
    # -------------------------------------------------------------------------
    decay_ok = decay_error < 1e-6
    add_ok = carry_active and post_add_count >= 5.0
    bounded_ok = bounded
    deactivate_ok = not still_active

    all_gates = decay_ok and add_ok and bounded_ok and deactivate_ok

    honest_verdict = (
        "complete: CAS updates verified — decay geometric, add activates templates, "
        "memory bounded, deactivation confirmed"
        if all_gates
        else (
            "complete: CAS updates PARTIALLY verified — "
            f"decay_ok={decay_ok} add_ok={add_ok} "
            f"bounded_ok={bounded_ok} deactivate_ok={deactivate_ok}"
        )
    )

    artifact = tmpl.build_result(
        {
            "compress_factor": COMPRESS_FACTOR,
            "smooth_alpha": SMOOTH_ALPHA,
            "smooth_target": SMOOTH_TARGET,
            "max_count": MAX_COUNT,
            "n_cas_steps": N_CAS_STEPS,
            "initial_count": INITIAL_COUNT,
            "model_id": MODEL_ID,
            # SCENARIO-CAS-001: decay verification
            "decay_count_trace": count_trace,
            "final_decay_count": final_decay_count,
            "expected_decay_count": expected_decay,
            "decay_absolute_error": decay_error,
            "decay_geometric_ok": decay_ok,
            # SCENARIO-CAS-002: add verification
            "post_cas_add_count": post_add_count,
            "carry_check_active_after_add": carry_active,
            "add_above_threshold_ok": add_ok,
            # Memory bounded check
            "max_count_observed_during_bounded_steps": max_observed,
            "memory_bounded_ok": bounded_ok,
            # Deactivation check
            "deactivated_count": deactivated_count,
            "deactivate_ok": deactivate_ok,
            # Gate summary
            "all_acceptance_gates_passed": all_gates,
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        status="success" if all_gates else "partial",
        schema="carnot.cas_diffusion.v1",
        honest_verdict=honest_verdict,
    )

    output_path = REPO_ROOT / "results" / "experiment_3396_cas_diffusion.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact written to {output_path}")
    print(f"honest_verdict: {honest_verdict}")
    print(f"all_acceptance_gates_passed: {all_gates}")


if __name__ == "__main__":
    main()
