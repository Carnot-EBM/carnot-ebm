with open("openspec/capabilities/kan/spec.md", "a") as f:
    f.write("""
## REQ-KAN-1926: S2KAN Symbolic Primitives Dictionary

The KAN capability MUST implement an extensible dictionary of symbolic primitives and learnable gates for S2KAN.

**Rationale:**
    To support arbitrary functional forms during symbolic discovery, S2KAN needs a dictionary of primitives (e.g. sin, exp) and learnable gates that enforce symbolic constraints.

**Acceptance criteria:**
    - Code implements a dictionary of symbolic primitives and learnable gates.
    - Code validates the model against a known functional form.
    - Tests verify the logic and achieve 100% test coverage.
    - `results/experiment_1926_s2kan_symbolic.json` is generated upon success.

### SCENARIO-KAN-1926: S2KAN Dictionary and Learnable Gates

Given the S2KAN primitives dictionary and learnable gates,
When validated against a known functional form,
Then the tests pass with 100% coverage, and `results/experiment_1926_s2kan_symbolic.json` is written.
""")
