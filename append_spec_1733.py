spec_update = """

### REQ-VERIFY-1733: NLA PWA Formal Verifier

The repository shall provide an NLA PWA Formal Verification abstraction that:
- abstracts the NLA core activation (ReLU) into Piecewise Affine (PWA) pieces;
- formulates the safety property (e.g., minimum confidence bound via max MSE for a given input radius) as a Z3 script/MILP formulation;
- computes a strict theoretical upper bound on MSE via interval arithmetic;
- runs deterministically on CPU;
- writes `results/experiment_1733.json` with `status="complete"`, `pwa_abstraction_generated=true`, the theoretical bound, and an honest verdict.

### SCENARIO-VERIFY-1733: NLA PWA Abstraction Verification
Given a trained or randomly initialized NLA MinimalSAE and a target input radius,
When the PWA abstraction and formal bounds are generated,
Then a valid Z3 script (.smt2 format) is produced with the MSE property assertion,
And a positive theoretical upper bound is calculated using interval arithmetic,
And the JSON artifact records `pwa_abstraction_generated=true` and `status="complete"`.
"""

with open("openspec/capabilities/verification/spec.md", "a") as f:
    f.write(spec_update)
