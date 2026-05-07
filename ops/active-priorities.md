# Active Priorities

Run date: `20260507`

Active priority count: `7`

## 1. Scope-reduction execution

- Active index id: `scope_reduction_execution`
- Source entries: SCOPE REDUCTION MILESTONE
- Next action: Finish the .112 scope-reduction tasks and block new variant expansion until the signal/noise, priority, lineage, claim, hardware, and comparator audits land.

## 2. Repair runtime and validation-context gate

- Active index id: `repair_runtime_and_validation_context_gate`
- Source entries: Repair-Loop Validation-Error-as-Context Fix
- Next action: Repair the local SOTA GGUF runtime first, then run only the validation-error-as-context salvage test before preserving or retiring the repair-executor lineage.

## 3. Paper integrity and claim narrowing

- Active index id: `paper_integrity_and_claim_narrowing`
- Source entries: Paper Integrity Audit, Paper-v6 Related Work Overhaul
- Next action: Keep publication hold active until critical figure, hardware-claim, related-work, and anchored-claim issues are reconciled to measured artifacts.

## 4. Verifier orthogonality and phase gates

- Active index id: `verifier_orthogonality_and_phase_gates`
- Source entries: Verifier Joint-Orthogonality Audit, Phase Prototype + Empirical Validation + Adversarial Check Discipline
- Next action: Measure verifier joint overlap before k-count claims or scale-up, and keep each phase behind prototype, empirical, and adversarial gates.

## 5. Planning and artifact lifecycle hygiene

- Active index id: `planning_and_artifact_lifecycle_hygiene`
- Source entries: Failure-Ledger v2 + Planner Discipline, artifact_not_updated_past_bootstrap Pattern, Auto-Populate prior_failures
- Next action: Treat prior-failure coverage, STEP-0 artifacts, and terminal artifact finalization as one operational hygiene lane rather than separate mandatory priorities.

## 6. Test memory safety guardrails

- Active index id: `test_memory_safety_guardrails`
- Source entries: Watchdog Insufficient for Single-Test Catastrophic Load, Pytest Worker Memory Watchdog
- Next action: Keep the pytest RSS watchdog and RLIMIT_AS cap active; do not create new memory watchdog lineages unless fresh failures bypass both guards.

## 7. Hardware portfolio narrowing

- Active index id: `hardware_portfolio_narrowing`
- Source entries: Phase-2 Hardware Story Re-Scope
- Next action: Carry the FPGA proof-of-concept caveats into the .112 hardware portfolio narrowing and cap active production hardware tracks.
