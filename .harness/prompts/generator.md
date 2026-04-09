# Generator Agent — TDD Implementation

## Task

Implement code changes following strict TDD against OpenSpec requirements. Write tests first, then implement, targeting 100% coverage in both Rust and Python.

## Inputs

- Architect design: `openspec/capabilities/{capability}/design.md`
- Spec requirements: `openspec/capabilities/{capability}/spec.md`
- Stories: `epics/stories/{story-id}.md`
- Generator handoffs from prior sprints: `.harness/handoffs/generator-*.md`

## Process

### Phase 1: Preparation

1. **Read the spec and design** — Understand every REQ-* and SCENARIO-* you must satisfy.
2. **Read existing code** — Understand the current state of affected crates and Python modules.
3. **Think harder before implementation** — Use extended thinking to plan:
   - Which files will be created or modified?
   - What is the test-first order? (test the leaf dependencies first)
   - Are there shared types in `carnot-core` that need updating first?
   - What PyO3 bindings are needed in `carnot-python`?

### Phase 2: Test First (Rust)

4. **Write Rust tests** — In `crates/carnot-{crate}/src/` or `tests/`:
   - Every test function must have a comment referencing `REQ-*` or `SCENARIO-*`
   - Use `#[test]` for unit tests, integration tests in `tests/` directory
   - Tests must initially fail (red phase)

5. **Run tests to confirm failure**:
   ```bash
   cargo test --workspace --exclude carnot-python
   ```

### Phase 3: Test First (Python)

6. **Write Python tests** — In `tests/python/`:
   - Every test function must have a docstring or comment referencing `REQ-*` or `SCENARIO-*`
   - Use pytest fixtures, parametrize where appropriate
   - Tests must initially fail

7. **Run tests to confirm failure**:
   ```bash
   pytest tests/python --cov=python/carnot --cov-report=term-missing
   ```

### Phase 4: Implement

8. **Implement Rust code** — Satisfy the failing Rust tests.
9. **Implement Python code** — Satisfy the failing Python tests.
10. **Build PyO3 bindings** if needed:
    ```bash
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python
    ```

### Phase 5: Verify

11. **Run all checks**:
    ```bash
    cargo test --workspace --exclude carnot-python
    cargo fmt --all -- --check
    cargo clippy --workspace --exclude carnot-python -- -D warnings
    pytest tests/python --cov=python/carnot --cov-report=term-missing --cov-fail-under=100
    ruff check python/ tests/
    ruff format --check python/ tests/
    mypy python/carnot
    python scripts/check_spec_coverage.py
    ```

12. **Verify coverage**:
    ```bash
    cargo tarpaulin --workspace --exclude carnot-python --fail-under 100
    ```

### Phase 6: Handoff

13. **Write handoff artifact** — Before session ends, always write `.harness/handoffs/generator-{topic}-{date}.md` with:
    - What was implemented (files created/modified)
    - What tests were added (with REQ-*/SCENARIO-* mappings)
    - Current test results (pass/fail counts)
    - Coverage numbers (Rust and Python)
    - Any deviations from the design doc (with rationale)
    - Remaining work if session limit was reached

14. **Update ops docs**:
    - `ops/changelog.md` — what was done
    - `ops/status.md` — current state

## Output

- Implementation code in `crates/` and `python/carnot/`
- Tests in `crates/*/tests/` and `tests/python/`
- Updated specs: `openspec/capabilities/*/spec.md` (Implementation Status section)
- Handoff: `.harness/handoffs/generator-{topic}-{date}.md`

## Constraints

- Never skip the red phase — tests must fail before implementation
- Never use `#[allow(dead_code)]` or `# type: ignore` without documented justification
- Every public function must have a doc comment (Rust) or docstring (Python) explaining it in verbose layman terms, not terse research shorthand
- All embedded secrets must use SOPS encryption, never plaintext
- 100% line and branch coverage required in both languages
- Spec coverage: every REQ-* and SCENARIO-* must have at least one test
- No orphan tests (tests that don't trace to any spec requirement)
