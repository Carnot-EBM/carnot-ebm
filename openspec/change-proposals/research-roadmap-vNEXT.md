# Research Roadmap V566: External Constraint Transfer and Reversible Self-Learning

**Milestone:** `2026.08.566`  
**Sequence:** 566  
**Planning date:** 2026-08-23  
**Execution manifest:** `research-roadmap-next.yaml`  
**Status:** proposed  
**Experiment range:** Exp6541-Exp6554 (14 tasks, four phases)

## Purpose

V566 restarts the external-transfer question at the smallest valid evidence
root. V565 did not disprove external transfer. It blocked before intake because
its source contract made unrelated OpenReview metadata mandatory. DRIFT-Bench
itself is public, content-pinnable, MIT-licensed, and executable through local
Z3 replay.

This milestone builds that external root, transfers Carnot's exact structural
router, measures a solver-guided inference-cost guard on current local models,
and turns transactional conflict memory into reversible continuous
self-learning. It keeps exact execution as release authority. Learned energy
may rank, route, allocate work, store verified conflicts, or abstain. It may
not certify itself or remove exact fallback.

## What V565 proved

| Evidence | Terminal finding | V566 consequence |
|---|---|---|
| Exp6527 evidence eligibility | V564's structural-router and conflict-memory rows are eligible after live recheck and duration correction. | Adopt those rows by path and hash. Do not depend on a retired task ID. |
| Exp6528 source/model/method contract | DRIFT provenance and model-cache checks were usable, but `v565_method_contract_ready_score=0.0` because OpenReview metadata was unavailable. | Replace broad discovery-channel fan-in with a direct-source contract. Make a channel mandatory only when the downstream task consumes it. |
| Exp6529 DRIFT intake | The conductor skipped the task after the Exp6528 gate failed. No intake artifact or fixture exists. | Build a fresh V566 fixture from the immutable DRIFT repository revision. Do not reuse the retired gate chain. |
| Exp6530 independent corpus audit | The audit ran and correctly reported the missing intake artifact, fixture, pinned source root, revision, license, and schema. | Keep the audit independent and always runnable. Use its readiness field as the external evidence root. |

V565 produced no external scientific comparison, no local-model cost result,
and no external continuous-learning result. Its blocked state is a broken
dependency contract, not a null result for those methods.

## The three largest gaps to the PRD vision

### Gap 1: there is no external, independently replayed constraint evidence root

Carnot's best router and conflict-memory results still come from internal
procedural data. The PRD asks for general constraint verification. The current
evidence does not show transfer across a new grammar, domain, lineage split, or
multi-turn state format.

V566 pins DRIFT-Bench commit
`d24cda4f59a6ee06bafe886f4724899a7ec94f1c`, preserves its chronological
problem structure, regenerates labels locally, seals base-problem lineages,
and runs an independent source, split, and exact-replay audit before any
learned comparison.

### Gap 2: learned energy has not shown useful behavior on current local models

The PRD vision needs energy-guided reasoning that complements exact checks.
Current positive evidence is solver-side branch ordering. It does not show how
current flagship local models react to exact structural difficulty or whether
a cheap exact guard can prevent inference-cost attacks.

V566 tests the tension between SMTrap (`2608.18921`) and Solver-Hard
(`2607.17047`). It controls SMT conflict count and proof-preserving surface
form separately across all three mandated GGUF families. It tests a bounded
exact-tool route as a cost guard. Solver conflict count is never a correctness
label, and model output is never release authority.

### Gap 3: self-learning is useful but not reversible or production-shaped

V564 showed chronological benefit from exact-admitted conflict memory. The
memory is still experiment-local. It does not distinguish temporarily obsolete
knowledge from permanently invalid knowledge, and it has not been tested on an
external stream behind the default-off `VerifyRepairPipeline` boundary.

V566 adds active, dormant, and retired states with asymmetric thresholds,
shadow reactivation, policy-gated retirement, restart, and rollback. It freezes
memory during each query and commits only after exact replay. Current cost,
retained-family performance, future exact-satisfying support, and unsafe reuse
are co-primary outcomes.

## Research findings incorporated

The dated review is in `research-references.md` under `V566 planner refresh -
2026-08-23`.

- **Reversible Forgetting (`2608.18177`)** supplies the hysteretic
  active/dormant/retired memory design for Phase 2.
- **SMTrap (`2608.18921`)** motivates the solver-conflict inference-cost stress
  and exact-tool guard in Phase 1.
- **Solver-Hard (`2607.17047`)** requires separate solver-hardness and surface
  strata. V566 will not assume that solver-hard means model-hard.
- **SemaPLC (`2608.18565`)** reinforces terminal executable receipts as the
  completion rule. Static checks alone cannot support a release claim.
- **Verification Autonomy Levels (`2608.19009`)** supplies useful vocabulary
  for candidate validity versus completeness. It is a reporting control, not
  scientific authority.
- **FormalTCS (`2608.20153`)** confirms that extraction remains a major long-term
  bottleneck. V566 does not start a separate Lean lineage before external CSP
  transfer is established.
- Current OpenReview, Hugging Face, Semantic Scholar, GitHub, Extropic, and Kona
  checks found no public executable EBM verifier or available thermodynamic
  device that should replace these experiments.

## Scientific invariants

1. **Direct sources only gate their consumers.** Discovery channels are
   advisory unless a task consumes their content.
2. **External means content-pinned.** Record source URL, commit, license, file
   hashes, local transformation hashes, and exact replay receipts.
3. **Rows precede aggregates.** Each comparison emits one row per problem,
   turn, model, surface, seed, arm, or condition.
4. **Exact authority stays separate.** Z3 validates assignments and conflicts.
   Learned routing, memory, and uncertainty cannot certify a result.
5. **No candidate deletion.** Guidance may reorder a complete candidate set.
   Native exact fallback stays reachable.
6. **Memory is transactional.** Freeze it within a query. Admit a write only
   after exact outcome validation and a refinement witness.
7. **Forgetting is reversible before retirement.** Dormancy and shadow
   reactivation precede any irreversible retirement decision.
8. **Future support is co-primary.** A current speed or token gain cannot hide
   reduced retained-family accuracy or future exact-satisfying support.
9. **No ARC re-solve.** The ARC task reads the live redirect ledger and changes
   only shared supervisor selection when outcome rows support it.
10. **No unchanged hardware probe.** GateMate receives no command without a new
    dated physical-state receipt newer than the previous continuity attempt.
11. **Closed verdict classes.** Every artifact declares `verdict_class` as one
    of `positive | circular_positive | null | blocked | disqualified | partial`.
12. **Blocked records explain the gate.** Every blocked artifact populates
    `gate_check_summary` with the failed check and observed value.

## Model policy

Two tasks perform headline LLM inference.

- Exp6546 uses `cached_sota_pair(gpu_indices=(0, 1))` and requires rows from
  `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`.
- Exp6550 uses the same three model families on a bounded chronological DRIFT
  slice. It may reduce unit count to preserve the preregistered timeout, but it
  may not replace the headline models with legacy small models.

The tasks load each `.gguf` path through llama.cpp. They do not call
`AutoTokenizer.from_pretrained()` on a GGUF repository ID. Qwen3.5-0.8B and
gemma-4-E4B-it may run CPU smoke tests only. Their rows cannot support a
headline result. If the required cache or GPU contract fails, the affected task
closes as blocked and records the exact precondition.

Formulaic dataset and replay tasks use Codex with `gpt-5.5`. Synthesis and
judgment tasks use the default Claude backend. The GateMate task and final
multi-file capstone use Opus because they carry hardware or coordination risk.

## Architecture after V566

```text
                    direct primary source contract
                                 |
                                 v
                 content-pinned DRIFT intake (Z3 replay)
                                 |
                                 v
                    independent external audit
                                 |
                 +---------------+----------------+
                 |               |                |
                 v               v                v
       structural headroom   SOTA cost stress   default-off
                 |           + exact cost guard  memory adapter
                 v               |                |
       Safety-Net transfer       |                v
          + exact fallback       |       hysteretic reversible
                 |               |          conflict memory
                 +-------+-------+                |
                         v                        v
                 independent transfer      chronological CSL
                       audit               + support/retention
                                                   |
                                                   v
                                         independent CSL audit

      ARC redirect-ledger closure ----+
      GateMate changed-state receipt --+--> independent V566 capstone

      RELEASE AUTHORITY: exact Z3 replay and executable receipts only
```

The learned layer proposes search order, routing, abstention, and verified
memory reuse. The exact layer checks every accepted result and remains
reachable after every learned decision.

## Phase 0: Repair the external evidence root

| Exp | Task | Primary output | Gate |
|---:|---|---|---|
| 6541 | Direct-source contract and V565 boundary | Immutable source, cache, split, and dependency contract | none |
| 6542 | Content-pinned DRIFT intake | Exact chronological fixture and row receipts | Exp6541 direct-source ready |
| 6543 | Independent corpus audit | Audited external evidence root | always runs |

Exp6541 directly addresses Exp6528. It makes the DRIFT repository, commit,
license, schema, and local solver the only hard source requirements for DRIFT
intake. OpenReview and other discovery channels remain dated advisory rows.
The task also imports Exp6527's eligible V564 boundaries by immutable path and
hash.

Exp6542 directly addresses the skipped Exp6529 scope with a new dependency
root. It creates a bounded balanced fixture across seating, scheduling, and
logic-grid domains. Base-problem lineages cannot cross train, development, and
held cells. Every admitted turn has a terminal Z3 receipt.

Exp6543 directly addresses Exp6530. It always runs and always writes
`external_constraint_corpus_audited_ready_score`. It recomputes source identity,
chronology, solver outcomes, split isolation, and transaction closure without
trusting intake aggregates.

## Phase 1: External transfer and local-model cost guarding

| Exp | Task | Primary comparison | Gate |
|---:|---|---|---|
| 6544 | External structural headroom | native, random, analytical, bounded refocus, one-shot enumeration | Exp6543 audited root |
| 6545 | External Safety-Net router | certified structural control versus learned router, abstention, exception table, exact fallback | Exp6544 held headroom |
| 6546 | SMT-cost stress and exact-tool guard | conflict strata, surface strata, three SOTA GGUFs, guarded versus unguarded cost | Exp6543 audited root |
| 6547 | Independent transfer audit | router value, shortcut tests, cost-guard calibration, exact equality | always runs |

Exp6544 must prove charged non-learned headroom before a learned router is
trained. Exp6545 keeps the exception table train-only and allows no held write.
It must report abstention calibration and exact fallback reachability.

Exp6546 uses SMT conflict count as a stress variable, not as ground truth for
model difficulty. Each logical instance receives proof-preserving surface
variants. The artifact reports token count, time, timeout, exact-tool dispatch,
and exact final validity by model and unit. The central claim is limited:
whether a preregistered cheap guard reduces bounded inference cost without
reducing exact completion.

Exp6547 independently recomputes both transfer lanes. It attacks family and
model identity, entity names, row order, solver-conflict leakage, exception
table contamination, timing fabrication, and aggregate-only conclusions.

## Phase 2: Reversible continuous self-learning

| Exp | Task | Primary output | Gate |
|---:|---|---|---|
| 6548 | Default-off production adapter | Transactional conflict memory behind `VerifyRepairPipeline` | Exp6543 audited root |
| 6549 | Hysteretic reversible memory | Active/dormant/retired controller with shadow reactivation | Exp6548 adapter ready |
| 6550 | Prospective chronological CSL | Scratch, frozen, transactional, one-threshold, hysteretic, same-query mutation | Exp6549 controller and Exp6543 root |
| 6551 | Independent CSL audit | Retention, future support, safety, restart, rollback, and dose audit | Exp6550 terminal comparison |

Exp6548 is default-off and preserves native behavior byte-for-byte when
disabled. Exp6549 compares no retirement, LRU, one-threshold, and hysteretic
state control under recurring regime changes. Only exact replay can admit,
dormant-reactivate, quarantine, or retire a conflict.

Exp6550 is the milestone's continuous self-learning experiment. It runs
chronologically. It freezes memory during each query, then commits after exact
verification. It measures current exact success, solver work, model tokens,
retained-family performance, future support, unsafe uses, capacity, churn,
restart equality, rollback equality, and same-query contamination.

Exp6551 independently replays every transition and recomputes the claimed
benefit from per-unit rows. A positive class requires exact answer equality,
zero unsafe writes and uses, positive charged benefit, retained-family floors,
future-support non-inferiority, and successful restart and rollback.

## Phase 3: Standing continuity and capstone

| Exp | Task | Primary output | Gate |
|---:|---|---|---|
| 6552 | ARC redirect-ledger generalization closure | Supported arm-selection refinement or honest no-firing closure | none |
| 6553 | GateMate changed-state continuity | One authorized action or zero-command blocked receipt | none |
| 6554 | Independent V566 capstone | Claim ledger, adoption boundary, and V567 handoff | always runs |

Exp6552 satisfies the ARC generalization floor through the live trajectory
supervisor redirect ledger. It does not solve a game. It treats an empty
post-REQ-ARC-WMTE-6640 outcome ledger as the valid result "no firings, nothing
to refine." It changes the curated selection order only when outcome rows
support a preregistered decision.

Exp6553 satisfies the non-terminal GateMate continuity slot. It first searches
for a dated operator-authored physical-state receipt newer than Exp6525. With no
receipt, it runs zero hardware commands and closes blocked. With a valid
receipt, it runs only the single authorized detect or flash action, stops after
the first result, and makes no speed or availability claim. KV260 and PolarFire
are terminal and need no milestone task.

Exp6554 always runs. It distinguishes blocked dependencies from null science,
recomputes every adopted headline from rows, preserves circularity and oracle
boundaries, and writes the next-state decision without changing historical
artifacts.

## Dependency graph

```text
Exp6541 direct source
   |
   v
Exp6542 DRIFT intake
   |
   v
Exp6543 independent audit
   |--------------------+----------------------+
   |                    |                      |
   v                    v                      v
Exp6544 headroom     Exp6546 cost guard     Exp6548 adapter
   |                    |                      |
   v                    |                      v
Exp6545 router          |                  Exp6549 reversible memory
   |                    |                      |
   +----------+---------+                      v
              v                            Exp6550 chronological CSL
         Exp6547 audit                         |
                                              v
                                         Exp6551 CSL audit

Exp6552 ARC ledger ---------------------------+
Exp6553 GateMate continuity ------------------+--> Exp6554 capstone
all terminal Phase 0-2 artifacts -------------+
```

Structured gates name only tasks in this roadmap. Each gated field is declared
verbatim in its upstream task's required artifact fields. Exp6543, Exp6547,
Exp6552, Exp6553, and Exp6554 always run so a failed chain still ends with a
diagnostic artifact.

## Prior-failure boundaries

| New task | Prior scope | What changes |
|---|---|---|
| Exp6541 | Exp6528 | Removes unrelated discovery channels from the hard DRIFT contract. |
| Exp6542 | Exp6529 | Uses Exp6541's direct source field instead of the retired V565 method-contract field. |
| Exp6543 | Exp6530 | Audits a new V566 artifact, fixture, and direct pinned source root. |
| Exp6545 | Exp6520 | Runs on an external family-blind fixture after Exp6527 corrected evidence eligibility; it adds abstention and independent audit. |
| Exp6552 | Exp6524 | Applies the 2026-08-22 no-firing closure rule and reads only post-schema live receipts. |
| Exp6553 | Exp6525 | Requires a receipt newer than Exp6525 and mechanically retires the unchanged attempt if the same verdict repeats. |

Each corresponding YAML task includes all four required `prior_failures`
fields and `retire_if_same_verdict: true`. No task requires a retired upstream
ID.

## Hardware requirements

| Resource | Tasks | Contract |
|---|---|---|
| CPU, RAM, disk, Z3 | Exp6541-Exp6545, Exp6547-Exp6554 | Pin solver and package versions. Use atomic shards. Record timeouts and terminal unit counts. |
| Dual RTX 3090 | Exp6546 and Exp6550 | Use cached GGUF paths through llama.cpp. Record GPU identity, VRAM, quantized file hashes, token counts, and wall time. No legacy model may supply headline rows. |
| GateMate A1-EVB-2M | Exp6553 only | Zero commands without a new physical receipt. One bounded authorized command with a valid receipt. No speedup or availability claim. |
| KV260 and PolarFire | none | Both have reached their defined terminal states. Preserve prior receipts; do not repeat smoke work. |
| Extropic Z1 / Kona | none | No authenticated local access or public reproducible runner exists. |

No task requires an FPGA redesign, NPU claim, thermodynamic execution claim, or
external paid model API.

## Milestone acceptance

V566 is scientifically positive only if all of these conditions hold:

1. The external corpus audit reaches
   `external_constraint_corpus_audited_ready_score=1.0`.
2. At least one Phase 1 lane produces row-supported charged value with exact
   fallback and no shortcut or authority violation.
3. The chronological self-learning experiment and independent audit show exact
   equality, zero unsafe memory operations, positive charged benefit,
   retained-family floors, future-support non-inferiority, and restart and
   rollback equality.
4. Every headline row uses a mandated SOTA GGUF where LLM inference is needed.
5. The capstone passes row consistency, adversarial verification, spec coverage,
   exclusion-manifest checks, and applicable end-to-end checks.

A blocked external root is an infrastructure result, not a null transfer
result. A solver-cost correlation without a guarded benefit is a null cost-guard
result. A current-task gain that reduces future support is partial or null, not
positive.

## Non-goals

- No model-generated answer becomes verification authority.
- No retry of retired finite-ID, grammar-only, or answer-level energy scoring.
- No public ARC game or level solve.
- No offline BFS or per-game ARC adapter claim.
- No FPGA redesign, unchanged GateMate probe, or hardware speedup claim.
- No Extropic, Kona, cloud-model, or paid API dependency.
- No claim that external transfer creates an oracle-distinct EBM verifier.

## Expected handoff

The milestone should leave four reusable assets:

1. a content-pinned, independently replayed external constraint fixture;
2. an audited external structural-router and exact-cost-guard result;
3. a default-off reversible conflict-memory controller with prospective
   continuous-learning receipts; and
4. a claim ledger that states which evidence can enter the PRD architecture
   and which lanes remain blocked, null, partial, or disqualified.
