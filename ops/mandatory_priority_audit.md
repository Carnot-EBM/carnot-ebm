# Mandatory Priority Audit

Run date: `20260507`

Statuses: `keep`, `consolidate`, `superseded`, `parked`, `retire`.

| # | source line | marker | priority | status | active index | rationale |
|---|---:|---|---|---|---|---|
| 1 | 201 | NEW 2026-05-06 (20:00Z) | Repair-Loop Validation-Error-as-Context Fix (compatible with .111 scope reduction) | keep | repair_runtime_and_validation_context_gate | Still valid as the single scoped repair-executor salvage gate after local SOTA runtime is fixed. |
| 2 | 244 | NEW 2026-05-06 (16:30Z) | SCOPE REDUCTION MILESTONE (.111 — preempts all other priorities) | keep | scope_reduction_execution | Controlling .112 governance directive; keep active until the scope-reduction milestone closes. |
| 3 | 347 | NEW 2026-05-06 (15:30Z) | trace2skill + Skillify Testing Rigor (.112-.116 series) | parked | - | Useful operational follow-up, but explicitly deferred until active scope reduction settles what stays. |
| 4 | 416 | NEW 2026-05-06 (12:00Z) | LARQL Decoupled-Attention Substrate Prototype (.111-.115 series) | parked | - | Strategic substrate work, but parked behind hardware narrowing and comparator cite/retire decisions. |
| 5 | 498 | NEW 2026-05-03 (22:55Z) | Verifier Joint-Orthogonality Audit (.96 mandatory) | keep | verifier_orthogonality_and_phase_gates | Still publication-blocking and scale-up-blocking for any k-verifier headline claim. |
| 6 | 563 | NEW 2026-05-03 (21:55Z) | Paper-v6 Related Work Overhaul (.94 or .95 mandatory) | consolidate | paper_integrity_and_claim_narrowing | Fold into the paper integrity and anchored-claims narrowing lane instead of tracking separately. |
| 7 | 623 | NEW 2026-05-03 (20:35Z) | NRGPT Frozen-Prefix Evaluation (optional, .95 or .96) | parked | - | The entry labels itself optional; it should not consume the active mandatory priority budget. |
| 8 | 657 | NEW 2026-05-03 (19:50Z) | CRITICAL — Pre-Commit `staged_files_only` is Causing Silent Data Loss | superseded | planning_and_artifact_lifecycle_hygiene | Superseded by Exp 1216 and the batching-check exemption/fail-forward hook changes. |
| 9 | 749 | NEW 2026-05-03 (19:40Z) | Phase-5 Intermediate-Scale Derisking (.96/.97) | parked | - | Valid future scale-up risk, but parked until the .112 scope and paper claim set are smaller. |
| 10 | 794 | NEW 2026-05-03 (13:55Z) | Retro Task Boundary Too Tight (artifact_not_updated_past_bootstrap) | superseded | planning_and_artifact_lifecycle_hygiene | Superseded by the Exp 1215 retro STEP-0 and max-turns pattern. |
| 11 | 825 | NEW 2026-05-03 (13:05Z) | Auto-Populate prior_failures from Failure-Ledger at Plan Time | superseded | planning_and_artifact_lifecycle_hygiene | Superseded by the shipped conductor_priors_autofill workflow and its tests. |
| 12 | 854 | NEW 2026-05-03 (06:33Z) | artifact_not_updated_past_bootstrap Pattern (5 .92 Retirements) | consolidate | planning_and_artifact_lifecycle_hygiene | Keep the issue only as part of the broader artifact lifecycle hygiene lane. |
| 13 | 883 | NEW 2026-05-02 (22:50Z) | Watchdog Insufficient for Single-Test Catastrophic Load — Need prlimit/cgroup Preemptive Cap | consolidate | test_memory_safety_guardrails | Consolidate with the pytest memory watchdog and RLIMIT_AS guardrail status. |
| 14 | 910 | NEW 2026-05-02 (21:35Z) | Pytest Worker Memory Watchdog — Stop the Recurring Load-Spike Pattern | consolidate | test_memory_safety_guardrails | Consolidate into one test memory safety guardrail instead of two mandatory entries. |
| 15 | 936 | NEW 2026-05-02 (20:05Z) | GRPO v5 Routing Bug — Re-propose with claude/opus | retire | - | Retire as a standalone priority; GRPO/VPRM lineages are under .112 consolidation/retirement. |
| 16 | 955 | NEW 2026-05-02 (18:50Z) | Paper Integrity Audit — 18 Issues Block Publication | keep | paper_integrity_and_claim_narrowing | Still active because publication remains blocked until the critical evidence issues close. |
| 17 | 1073 | NEW 2026-05-02 (06:40Z) | Seed IQ Verified — Active-Inference Phase 4 Track (3 candidate tasks) | parked | - | Strategic context is preserved, but active-inference expansion is parked during scope reduction. |
| 18 | 1300 | NEW 2026-05-02 (06:25Z) | EBT/ARC-AGI-3 Paradigm-Shift Tasks (4 candidate tasks) | parked | - | Paradigm-shift ideas remain research context, not current active mandatory work. |
| 19 | 1404 | NEW 2026-05-02 | Phase-3 Thinking-Mode Composition (4 candidate tasks) | parked | - | Inference-mode expansion is parked until scope reduction and current paper claims are narrowed. |
| 20 | 1501 | NEW 2026-05-01 | Failure-Ledger v2 + Planner Discipline (5 STRUCTURAL FIXES + 3 PLANNER-PROMPT DELTAS) | consolidate | planning_and_artifact_lifecycle_hygiene | Partly shipped and now tracked as planning/artifact lifecycle hygiene rather than five separate fixes. |
| 21 | 1619 | NEW 2026-05-01 | LLM Failure Exemplar Corpus + Goodfire Silico Comparison | parked | - | Useful benchmark curation, but not mandatory while scope-reduction tasks are closing. |
| 22 | 1677 | Carry-forward | Carry-forward from .85 (operator-retired tasks the .86 planner | superseded | planning_and_artifact_lifecycle_hygiene | Historical carry-forward batch; later milestones either executed or reclassified these tasks. |
| 23 | 1694 | NEW 2026-04-30 | Phase Prototype + Empirical Validation + Adversarial Check Discipline (5 LOAD-BEARING TASKS) | keep | verifier_orthogonality_and_phase_gates | Still active as the project-wide gate against architecture-heavy, evidence-light expansion. |
| 24 | 1770 | REVISED 2026-04-30 | Phase-2 Hardware Story Re-Scope (HIGH PRIORITY — paper-shaping) | consolidate | hardware_portfolio_narrowing | Consolidate into the .112 hardware portfolio narrowing decision. |
