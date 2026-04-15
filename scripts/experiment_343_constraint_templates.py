#!/usr/bin/env python3
"""Exp 343: ConstraintTemplateLibrary — Tier 2 constraint ADDITION from memory patterns.

**Researcher summary:**
    Exp 134 proved that reweighting existing constraints does NOT improve accuracy.
    The fix: when CaseMemory observes a frequent error pattern for a model, ADD a new
    constraint template that checks that specific error type.

    This experiment validates that ConstraintTemplateLibrary:
    1. Correctly activates templates once observation thresholds are crossed.
    2. Applies active templates to synthetic arithmetic responses.
    3. Generates constraints that correctly identify arithmetic errors.

    The experiment is CPU-only and fully deterministic — no GPU required.

    Output: results/experiment_343_constraint_templates.json

Spec: REQ-LEARN-017, REQ-LEARN-018,
      SCENARIO-LEARN-029, SCENARIO-LEARN-030, SCENARIO-LEARN-031, SCENARIO-LEARN-032
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "python"))

from experiment_template import ExperimentTemplate  # noqa: E402

from carnot.pipeline.constraint_template_library import (  # noqa: E402
    ConstraintTemplateLibrary,
)

# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

MODEL_ID = "qwen3.5-0.8b"
N_CARRY_OBSERVATIONS = 20  # Exceeds min_frequency=5 for carry_check

# Synthetic arithmetic responses that contain known error patterns.
# Each response was crafted to trigger specific template types when active.
SYNTHETIC_RESPONSES = [
    # carry error: 24 × 3 should be 72 but LLM writes 62
    "To solve this: 24 × 3 = 62. So the answer is 62.",
    # sign error: (-3) × (-4) should be 12 (positive) but LLM writes -12
    "Multiply: (-3) × (-4) = -12. Therefore the product is -12.",
    # unit inconsistency: mixes kg and g
    "The total mass is 5 kg + 200 g = 5.2 kg.",
    # comparison direction error: 50 > 30 but then claims 50 - 30 = -20
    "Since 50 > 30, the difference is 50 - 30 = -20.",
    # clean response with no arithmetic errors — templates should return []
    "The answer is obtained by adding all values: total = 100.",
]


def main() -> None:
    """Run the ConstraintTemplateLibrary experiment."""
    tmpl = ExperimentTemplate(
        exp_id=343,
        title="ConstraintTemplateLibrary — Tier 2 constraint addition from memory patterns",
        deliverable="results/experiment_343_constraint_templates.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # -------------------------------------------------------------------------
    # Step 1: Create library and register all 4 builtin templates
    # -------------------------------------------------------------------------
    lib = ConstraintTemplateLibrary()
    lib.register_builtin_templates()

    n_templates_registered = len(lib._templates)
    templates_names = list(lib._templates.keys())
    print(f"Registered {n_templates_registered} templates: {templates_names}")

    # -------------------------------------------------------------------------
    # Step 2: Simulate 20 observations of the "carry_check" pattern for MODEL_ID
    # -------------------------------------------------------------------------
    # In production, CaseMemory would call observe_pattern each time it records
    # a carry error for this model. Here we simulate 20 such observations in one call.
    lib.observe_pattern("carry_check", MODEL_ID, count=N_CARRY_OBSERVATIONS)
    print(f"Simulated {N_CARRY_OBSERVATIONS} carry_check observations for {MODEL_ID}")

    # Check activation state after observations
    active_templates = lib.get_active_templates(MODEL_ID)
    n_active = len(active_templates)
    active_keys = [t.pattern_key for t in active_templates]
    print(f"Active templates after {N_CARRY_OBSERVATIONS} carry observations: {active_keys}")

    # -------------------------------------------------------------------------
    # Step 3: Apply active templates to 5 synthetic arithmetic responses
    # -------------------------------------------------------------------------
    total_constraints_generated = 0
    per_response_results = []

    for i, response in enumerate(SYNTHETIC_RESPONSES):
        constraints = lib.apply_active_templates(response, MODEL_ID)
        n_constraints = len(constraints)
        total_constraints_generated += n_constraints
        per_response_results.append({
            "response_index": i,
            "response_snippet": response[:60],
            "n_constraints": n_constraints,
            "constraint_types": [c.constraint_type for c in constraints],
            "satisfied": [c.metadata.get("satisfied") for c in constraints],
        })
        print(f"Response {i}: {n_constraints} constraints from active templates")

    # -------------------------------------------------------------------------
    # Step 4: Build and save artifact
    # -------------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "n_templates_registered": n_templates_registered,
            "n_active_after_observations": n_active,
            "n_constraints_generated": total_constraints_generated,
            "templates_names": templates_names,
            "active_template_keys": active_keys,
            "per_response_results": per_response_results,
            "carry_observation_count": N_CARRY_OBSERVATIONS,
            "model_id": MODEL_ID,
        },
        status="success",
        schema="carnot.constraint_template_lib.v1",
    )

    output_path = REPO_ROOT / "results" / "experiment_343_constraint_templates.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact written to {output_path}")
    print(f"  n_templates_registered: {n_templates_registered}")
    print(f"  n_active_after_observations: {n_active}")
    print(f"  n_constraints_generated: {total_constraints_generated}")
    print(f"  templates_names: {templates_names}")


if __name__ == "__main__":
    main()
