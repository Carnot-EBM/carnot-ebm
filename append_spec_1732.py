with open("openspec/capabilities/verification/spec.md", "a") as f:
    f.write("""

### REQ-VERIFY-1732: Behavioral Entanglement Reweighting for k=16 Ensemble

The repository shall replicate the de-entangled reweighting algorithm on Carnot's k=16 setup (arXiv:2604.07650) to measure lift on the adversarial corpus.
- Implement the reweighting algorithm to adjust weights for the 16 verifiers based on failure covariance.
- Evaluate on the adversarial test corpus to measure accuracy lift.
- Document the new weights and pass rates in `docs/research-notes/k16_reweighting.md`.
- Produce a terminal JSON artifact `results/experiment_1732_k16_reweighting.json` with status, accuracy_lift_pct, and honest_verdict.

### SCENARIO-VERIFY-1732: De-entangled Reweighting Evaluation
Given a k=16 verifier ensemble and an adversarial test corpus,
When the de-entangled reweighting algorithm adjusts weights based on failure covariance,
Then it records the new weights, evaluates accuracy lift, and writes the required artifacts.
""")
