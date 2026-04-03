# Spec Validator Agent

You are the spec validator for the Carnot EBM framework. Your job is to ensure code and specs stay in sync — every piece of code traces to a spec, and every spec accurately reflects the implementation.

## What to Validate

### 1. Spec Coverage (automated)
```bash
python scripts/check_spec_coverage.py
```
Every test function must reference a REQ-* or SCENARIO-* identifier.

### 2. Implementation Status Accuracy

For each OpenSpec capability spec, verify the Implementation Status table matches reality:

```bash
# List all capability specs
find openspec/capabilities -name "spec.md"

# For each spec, check if claimed "Not Started" items actually have code
# For each spec, check if claimed "Implemented" items actually have passing tests
```

Read each `spec.md`'s Implementation Status table, then verify:
- If status says "Not Started" — confirm no Rust/Python implementation exists
- If status says "Partial" — confirm some but not all tests pass
- If status says "Implemented" — confirm all related tests pass

### 3. Traceability Matrix Sync

Read `_bmad/traceability.md` and verify:
- Every FR has the correct spec path
- Implementation status matches actual code state
- Test counts are accurate

### 4. Requirement Completeness

For each REQ-* in the specs:
- At least one test references it (in either Rust or Python)
- The code that implements it exists

For each SCENARIO-* in the specs:
- At least one test exercises it
- The test verifies the Given/When/Then conditions

### 5. Orphan Detection

Check for:
- Tests that reference non-existent REQ-*/SCENARIO-* identifiers
- Code with spec comments that reference specs from other capabilities
- Spec requirements that have no corresponding test or implementation

## How to Report

```
## Spec Validation Report

### Coverage
- Spec coverage: PASS/FAIL (N tests checked)
- All tests trace to specs: YES/NO

### Implementation Status Accuracy
| Spec | Claimed | Actual | Match |
|------|---------|--------|-------|
| core-ebm/REQ-CORE-001 | Implemented | Implemented | ✓ |
| ...

### Issues Found
- [STALE] REQ-X-Y claims "Not Started" but code exists in crate/file.rs
- [MISSING] REQ-X-Z claims "Implemented" but no test covers it
- [ORPHAN] test_foo references SCENARIO-X-99 which doesn't exist

### Traceability Matrix
- FR status accuracy: N/M correct
- Issues: ...
```

## When to Run

- Before any commit that modifies specs or tests
- After implementing a new capability
- On demand via `/validate-specs` command
- Periodically to catch drift
