with open("openspec/capabilities/kan/spec.md", "a") as f:
    f.write("""
## REQ-KAN-1859: S2KAN Z3 Verification

The KAN capability MUST connect S2KAN symbolic primitives to Z3 for formal verification.

**Rationale:**
    Verification tiers must be formally proven via Z3/MILP. A transpiler script must convert S2KAN primitives to Z3 constraints to verify consistency over a bounded input domain.

**Acceptance criteria:**
    - A script transpiles S2KAN layer into Z3 constraints.
    - Consistency over a bounded input domain is formally proven using Z3.
    - Test coverage for the new code is 100%.
    - `results/experiment_1859_z3_verify.json` is generated upon success.

### SCENARIO-KAN-1859: Z3 Transpilation and Verification

Given the `s2kan.py` primitives and a Z3 transpilation script,
When S2KAN operations are converted to Z3 constraints and verified for a bounded domain,
Then the proof completes successfully, tests pass with 100% coverage, and `results/experiment_1859_z3_verify.json` is written.
""")
