import os

spec_path = "openspec/capabilities/self-learning/spec.md"
content = """

---

## REQ-FR11-1681: ETS Policy Evaluation in FR-11 Loop

**Statement:** The FR-11 self-learning loop MUST replace RLHF with Energy-Term Transition Probabilities (ETS) for policy evaluation.
- The self-learning promotion function MUST incorporate test-time energy scaling based on ETS.
- Test-time compute MUST scale with measured energy uncertainty.

**Acceptance criteria:**
- `EtsPolicyEvaluator.scale_test_time_compute(base_compute, uncertainty)` returns scaled compute proportional to uncertainty.
- `EtsPolicyEvaluator.promote_policy(candidate_policy, transition_probabilities, uncertainty)` returns a promotion decision incorporating energy scaling.
- Experiment 1681 confirms ETS promotion works and writes output to `results/experiment_1681_ets_policy.json`.

**Spec traces:** Exp 1681, FR-11 Tier 1

---

## SCENARIO-FR11-1681: Compute Scales with Uncertainty

**Given** an `EtsPolicyEvaluator` and a base compute limit
**When** test-time energy uncertainty is high
**Then** test-time compute is scaled up proportionally
**And** promotion decision accounts for ETS.
"""

with open(spec_path, "a") as f:
    f.write(content)

print(f"Appended spec to {spec_path}")
