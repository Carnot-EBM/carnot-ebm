import sys

with open("openspec/capabilities/kan/spec.md", "a") as f:
    f.write("""
## REQ-KAN-1840: PWA KAN Abstraction

The KAN capability MUST implement Piecewise Affine (PWA) abstractions for nonlinear KAN units to support MILP verification (arXiv:2602.06737).

**Rationale:**
    Formal verification of KAN units requires converting continuous, nonlinear spline functions into piecewise-linear components that can be encoded in an MILP solver. The abstraction must compute affine bounds (lower and upper) for each linear segment.

**Acceptance criteria:**
    - `python/carnot/verify/pwa_kan.py` exposes a PWA abstraction function that converts 1D splines to piecewise-linear segments with bounds.
    - Each segment contains bounds information.
    - `results/experiment_1840_pwa_kan.json` is generated correctly.
    - Tests verify the logic and achieve 100% coverage, referencing `REQ-KAN-1840`.

### SCENARIO-KAN-1840: Spline to PWA conversion

Given a nonlinear KAN unit (e.g. spline callable),
When the PWA abstraction is applied,
Then it computes piecewise-linear approximations with affine bounds for each segment, tests pass with 100% coverage, and `results/experiment_1840_pwa_kan.json` is written.
""")
