# Carnot Research Roadmap V637: Mention Grounding and Recurring Memory

**Created:** 2026-09-12
**Milestone:** 2026.09.637
**Milestone title:** Mention-grounded verification, recurring constraint memory, and scored-path validation
**Status:** Planned; thirteen tasks, exp7233 through exp7245, in four phases.
**Execution file:** `research-roadmap.yaml` after conductor activation.

## Evidence Boundary

V636 produced terminal artifacts. Its held-out source branch did not run, and
four later artifacts are quarantined. V637 keeps those results as historical
diagnosis. It does not treat a numeric readiness field as authenticated
evidence.

The contract receipt is advisory. No scientific task depends on it. The
receipt checks plan identity, source versions, gate behavior, and artifact
classification. It invokes no model and changes no production default.

## Exact Task Contract

There are **13 tasks**, **exp7233 through exp7245**, in this exact order.
The Markdown parser and YAML parser read separate bytes. Titles, deliverables,
and gates below are literal contract values.

| Order | Task ID | Title | Deliverable | Structured gates |
|---|---|---|---|---|
| 1 | exp7233-contract | V637 source and execution contract receipt | results/experiment_7233_v637_contract.json | None |
| 2 | exp7234-arc-scored-dryrun | Local scored-stack ARC selfparse dry run | results/experiment_7234_v637_arc_scored_dryrun.json | None |
| 3 | exp7235-arc-path-audit | Independent ARC dry-run reachability and authority audit | results/experiment_7235_v637_arc_path_audit.json | None |
| 4 | exp7236-mention-fixture | Public mention compiler and sealed semantic fixture | results/experiment_7236_v637_mention_fixture.json | None |
| 5 | exp7237-mention-canary | Bounded Qwen3.8 mention-grounding canary | results/experiment_7237_v637_mention_canary.json | exp7236-mention-fixture.mention_fixture_ready_score == 1 |
| 6 | exp7238-mention-capture | Qwen3.8 held-out mention-grounding capture | results/experiment_7238_v637_mention_capture.json | exp7236-mention-fixture.mention_fixture_ready_score == 1; exp7237-mention-canary.mention_canary_ready_score == 1 |
| 7 | exp7239-semantic-audit | Independent mention fidelity and verification-value audit | results/experiment_7239_v637_semantic_audit.json | None |
| 8 | exp7240-recurrence-fixture | Feedback-validated archive memory prototype and recurrence stream | results/experiment_7240_v637_recurrence_fixture.json | None |
| 9 | exp7241-recurrence-learning | Prospective continuous learning with validated archive reuse | results/experiment_7241_v637_recurrence_learning.json | exp7240-recurrence-fixture.recurrence_fixture_ready_score == 1 |
| 10 | exp7242-recurrence-audit | Cold recurrence-memory causality and safety audit | results/experiment_7242_v637_recurrence_audit.json | None |
| 11 | exp7243-native-memory | Native archive-memory parity and complete cost measurement | results/experiment_7243_v637_native_memory.json | exp7240-recurrence-fixture.recurrence_fixture_ready_score == 1 |
| 12 | exp7244-board-disposition | GateMate changed-state review and graduated-board disposition | results/experiment_7244_v637_board_disposition.json | None |
| 13 | exp7245-capstone | V637 independent evidence matrix and branch decisions | results/experiment_7245_v637_capstone.json | None |

## Phase 1: Contract and Live-Path Checks

Exp7233 binds the plan and remains advisory. Exp7234 runs one local scored-path
dry run with no game adapter or submission. Exp7235 audits reachability and
authority without launching another model.

## Phase 2: Mention-Grounded Verification

Exp7236 creates public mention identifiers and a sealed semantic fixture.
Exp7237 checks bounded output quality. Exp7238 performs the held-out capture
only after both readiness gates pass. Exp7239 audits source fidelity and value
independently, including missing support and direction changes.

## Phase 3: Recurring Constraint Memory

Exp7240 creates an immutable recurrence stream and six-arm memory contract.
Exp7241 measures prospective learning after the fixture passes. Exp7242 audits
causality, retention, rollback, and safety even when learning value is null.

## Phase 4: Native Cost and Disposition

Exp7243 measures exact native parity and full boundary cost after fixture
readiness. Exp7244 records changed GateMate state and the dispositions of all
three attached boards. Exp7245 builds an independent evidence matrix. It keeps
contract completion separate from scientific value.

## Dependency Graph

```text
exp7236 -> exp7237
exp7236 -> exp7238
exp7237 -> exp7238
exp7240 -> exp7241
exp7240 -> exp7243
```

The arrows are the five structured gates. No arrow starts at Exp7233. Other
historical or audit inputs are authenticated evidence, not hidden gates.

## Execution Limits

Model work uses the required Qwen3.8 GGUF only after its task owns the needed
resources. CPU work uses exact solvers or deterministic replay. Board rows name
the actual venue. A blocked resource stays blocked and does not gain synthetic
evidence.

Each task prints progress at every numbered phase. Long native or child work
uses streamed output and truthful heartbeats. Tests use private output paths.
No task publishes, uploads, submits, or changes a production default.
