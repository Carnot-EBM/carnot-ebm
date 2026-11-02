# V748 equal-budget relational verifier audit

Exp8729 executes the two schedules sealed by qualified Exp8726. Each schedule
uses five calls per world and verifier variant, without short circuit. All120
original constructed worlds and six variants remain in the experiment:1440
world/variant/arm rows and7200 calls. The original90 development and30 evaluation
IDs are preserved. These are exposed worlds; the partition creates no new
independent evidence. No model is loaded and no text is generated.

The baseline schedule is truth, equivalent, invalid, unknown and deleted_records.
The targeted schedule is truth, invalid, entity_bijection, required_premise and
required_role. Expected labels come from the separate explicit-membership oracle.
A conjunction needs every premise, and missing evidence remains unknown. The
correct verifier, reject-all, accept-all, omitted-required-predicate,
reversed-arguments and missing-as-false variants are unchanged from Exp8715.

On the30 exposed evaluation worlds, the baseline detects120/150 defective
world/variant units; the targeted schedule detects130/150. The paired difference
is6.67 percentage points. A percentile bootstrap with10000 draws, clustered by
world, gives95 percent bounds of3.33 to10.00 percentage points. The fixed seed
is7488729. Seeds and variants do not multiply the30 evaluation world clusters.

The correct verifier rejects0/60 supported alternatives in each arm and
converts0/60 baseline and0/30 targeted unknown cases to false. Both degenerate
verifiers fail. Every missed defective unit retains its scheduled observations
in missed_defect_witnesses; no schedule was tuned to remove those misses.

Each arm executes3600 calls overall and900 on the evaluation worlds. Predicate
input work is3840 versus4800 overall and960 versus1200 on evaluation worlds.
This counts supplied predicates, including inputs to constant-return variants;
it does not count internal comparisons or instructions. Actual measured wall
time is retained separately in arm_rows and each unit's latency_s. This study
claims a matched-call detection gain and makes no equal-wall-time claim.

The result is oracle-dependent constructed evidence, so a qualified win is
circular_positive. verifier_audit_ready_score records a complete qualified audit;
cost_matched_detection_gain_score separately gates the utility result. Null or
false-rejection tradeoff cases still retain audit readiness when checks pass.
Independent generalization stays zero. No natural extraction, learned energy
moat, game solve, generation improvement or physical acceleration is established.

V747 remains unchanged: Exp8715's120 versus130 result used unequal budgets;
Exp8717's1.138x batching gain and1.005x same-batch native/Python comparison are
separate results; Exp8719 established exact signed32/24 parity, not speed. The
missing CUDA repair does not block this CPU branch. The binding authenticates
all fourteen V748 task objects, the visible table and independent canonical
digest separately from the original scientific methods.

The thin runner reuses the existing bounded lifecycle in an isolated namespace.
Sources, tests and inputs are frozen before execution. Raw invocation evidence
retains child exits, clocks, node IDs, JUnit and SQLite coverage. Unchanged
publication validators check cold replay, adversarial flags and strict row
consistency before atomic publication. Rehashed summary and primitive tampering,
missing evidence and unsafe private-output arguments are covered by focused
tests. Ops, status and traceability reconciliation remain conductor-owned.
