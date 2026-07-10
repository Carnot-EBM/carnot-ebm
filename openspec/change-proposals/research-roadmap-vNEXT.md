# Research Roadmap vNEXT - Milestone 2026.07.501

**Milestone title:** Live SOTA Schema Repair, Gate-Clean CSL, Sparse Scaling, Receipt Repair, and ARC Strategy Routing

**Planner date:** 2026-07-10
**Previous milestone:** 2026.07.500
**Task range:** Exp 5523-5535
**Pre-staged roadmap:** `research-roadmap-next.yaml`

## Inputs Read

Required repository inputs were read before planning:

1. `research-program.md`
2. `_bmad/prd.md`
3. `_bmad/architecture.md`
4. `ops/status.md`
5. `ops/changelog.md`
6. `research-complete.yaml`
7. `research-roadmap.yaml`
8. `openspec/change-proposals/`
9. `ops/conductor-log.md`
10. `research-references.md`
11. `research-hardware-wishlist.md`

Additional guardrails checked before writing the roadmap:

- `CLAUDE.md`
- `CODEX.md`
- `ops/exclusion_manifest.yaml`
- `ops/known-issues.md`
- `ops/arc_solve_registry.yaml`
- `scripts/experiment_template.py`
- `scripts/conductor_gates.py`
- `scripts/audit_roadmap_gates.py`
- `scripts/arc_levelup_guarantee_lint.py`
- `scripts/exclusion_manifest_lint.py`
- `ops/e2e-test-plan.md`

## Literature Refresh Incorporated

The planner performed a 2025-2026 refresh across arXiv, OpenReview public pages where reachable,
Hugging Face Papers, GitHub discovery, Semantic Scholar-style public routes for EBT `2507.02092` and
ARM-EBM `2512.15605`, Extropic writing, Logical Intelligence public posts, and Carnot's accumulated
reference history. Non-duplicate actionable items were appended to `research-references.md` under:

`## V501 Planner Refresh - 2026-07-10`

New planning consequences:

- **GAM** (`arXiv:2604.12285`) motivates a two-tier continuous self-learning fixture: event-progression
  graph for rapid updates, topic-associative graph for stable consolidation, and semantic-shift gates
  before promotion.
- **Compliance-grade LLMOps for schema-constrained local serving** (`arXiv:2605.11232`) reframes the
  Exp5513 SOTA failure as a workload/runtime issue as much as a model-quality issue. V501 records prompt
  prefix hashes, token budgets, truncation, grammar path, retry counts, and exact-validator handoff before
  scoring reasoning.
- **Metacognition and faithful uncertainty** (`arXiv:2605.01428`) sharpens the hard/soft panel: schema
  validity, exact correctness, uncertainty, abstention, and confident wrong answers must be separate
  fields. Missing rows are not credited as abstentions.
- Extropic TSU and Logical Intelligence Kona/Aleph remain architecture context only. There is no local SDK
  or reproducible baseline, so V501 continues the receipt-only hardware posture and deterministic-verifier
  authority.

## What 2026.07.500 Proved

The `.500` milestone completed its task range and closed with a useful, honest capstone. It proved several
small substrates while keeping broad claims blocked.

| Lane | Experiments | Finding |
|------|-------------|---------|
| Structured output fixtures | Exp 5512 | Deterministic hard/soft candidate fixtures were schema-valid and ready for exact-validator handoff. The live Qwen3.6 smoke row was still invalid. |
| Live SOTA hard/soft evidence | Exp 5513, 5514 | GPU offload worked for Qwen3.6, but the SOTA panel emitted schema-invalid or missing rows, so `sota_structured_panel_ready=false` and the energy sidecar stayed gated. |
| Continuous self-learning | Exp 5515, 5516, 5517 | The independent graph-memory fixture showed `heldout_delta=1.0` with stale-evidence rejection, but downstream gates read `None` because the conductor chose a later same-number sidecar artifact. Broad CSL claims stayed blocked. |
| Sparse repair | Exp 5518 | Exact-checked sparse repair descriptors matched exact-only success on tiny fixtures and beat random blocks, but speedup claims remained disallowed. |
| Hardware | Exp 5519 | Receipt-only continuity ran. CPU/CUDA parser rows were malformed, KV260 and GateMate identity stayed blocked, PolarFire was reachable, and `matched_timing_available=false`. |
| ARC live path | Exp 5520, 5521 | `sb26 L3` was a valid non-duplicate target and the live attempt used `solve_provenance=live_agent_self_discovery`, but it banked no level and repeated coordinates returned during the actual attempt. |
| Capstone | Exp 5522 | Final claims were bounded: no structured SOTA claim, no energy-sidecar headline, fixture-level CSL evidence only, sparse repair bounded, no hardware speedup, and `arc_registry_delta=0`. |

## Three Biggest Gaps To PRD Vision

1. **Local SOTA reasoning is blocked at the candidate-output interface.** The PRD asks for verifiable
   reasoning where LLMs propose and exact constraints decide. Carnot has exact fixtures and GPU offload,
   but flagship GGUF models still fail to provide reliable structured rows. V501 must isolate whether the
   fault is prompt format, grammar/runtime integration, truncation, JSON extraction, or semantic field
   mismatch, then rerun only behind a clean repair gate.

2. **Continuous self-learning has a scientific positive but an operational gate failure.** FR-11 requires
   learning from experience without metric leakage. Exp5515 produced the right scientific fields, but the
   conductor selected a later sidecar lacking those fields. V501 must make the CSL gate artifact canonical,
   then stress event/topic memory promotion, stale evidence, and negative transfer before any local-SOTA
   memory claim.

3. **Operational embodiments still do not add capability.** Sparse repair, hardware, and ARC all produced
   useful receipts but no broad capability delta: no sparse scale evidence, no matched hardware timing,
   and no new ARC level. V501 should scale sparse repair cautiously, fix receipt parsers, and change ARC
   action routing through strategy portfolios and repeated-coordinate suppression.

## Architecture For V501

```text
          research-program / PRD / architecture / .500 capstone / source refresh
                                      |
                                      v
                      +-------------------------------+
                      | V501 Transition + Source Delta |
                      +----------------+--------------+
                                       |
          +----------------------------+-----------------------------+
          |                            |                             |
          v                            v                             v
+----------------------+     +----------------------+       +----------------------+
| SOTA Schema Repair   |     | CSL Gate-Clean Loop  |       | Sparse + Hardware   |
| - failure taxonomy   |     | - canonical artifact |       | - multi-seed scale  |
| - grammar/repair     |     | - event/topic memory |       | - parser receipts   |
| - exact validation   |     | - residue stress     |       | - no speedup unless |
+----------+-----------+     +----------+-----------+       |   matched timing    |
           |                            |                   +----------+-----------+
           v                            v                              |
+----------------------+     +----------------------+                  |
| SOTA Hard/Soft v2    |     | SOTA CSL Memory v2   |                  |
| - mandated GGUFs     |     | - mandated GGUFs     |                  |
| - uncertainty split  |     | - no leakage claim   |                  |
| - no missing credit  |     | - negative transfer  |                  |
+----------+-----------+     +----------+-----------+                  |
           |                            |                              |
           +----------------------------+------------------------------+
                                        |
                                        v
                         +-------------------------------+
                         | ARC Strategy-Routed Live Path |
                         | - registry precheck           |
                         | - strategy portfolio          |
                         | - repeated-coordinate guard   |
                         | - live self-discovery only    |
                         +---------------+---------------+
                                         |
                                         v
                         +-------------------------------+
                         | Capstone / Spec Reconciliation|
                         +-------------------------------+
```

## Phase Plan

### Phase 0 - Transition And Source Freshness

**Goal:** Carry `.500` facts forward and verify that the new reference deltas are reflected in the
execution plan.

- `exp5523-transition-v501` archives `.500` close-state and records the exact claims that remain blocked.
- `exp5524-v501-source-delta-ingestion` performs the execution-time freshness check and appends only
  non-duplicate actionable findings.

### Phase 1 - Live SOTA Schema Repair

**Goal:** Turn the Exp5513 live GGUF schema failure into a classified, repairable interface problem before
spending more flagship runtime.

- `exp5525-sota-schema-failure-taxonomy` replays the structured-output path and separates prompt, grammar,
  truncation, extraction, JSON validity, field validity, and exact-validator mismatch.
- `exp5526-gated-sota-structured-repair-loop` runs only if the taxonomy is ready and tests a bounded
  repair loop: validator feedback, retry, extraction repair, and exact handoff.
- `exp5527-gated-sota-hard-soft-panel-v2` runs the mandated local SOTA GGUF panel only if the repair loop
  produces schema-valid candidate rows.

### Phase 2 - Continuous Self-Learning With Gate-Clean Artifacts

**Goal:** Preserve Exp5515's non-tautological positive while making downstream gating mechanically safe.

- `exp5528-csl-canonical-gate-artifact` emits a canonical CSL gate artifact and avoids same-number sidecars
  that would be newer than the primary artifact.
- `exp5529-gated-csl-event-topic-residue-stress` applies the GAM-inspired event/topic split and tests stale
  evidence, semantic-shift consolidation, and negative transfer.
- `exp5530-gated-sota-csl-memory-panel-v2` uses the mandated local SOTA GGUF models only after gate fields
  and residue stress are clean.

### Phase 3 - Constraint Scaling, Hardware Receipts, ARC Live Path, Capstone

**Goal:** Improve operational reach without claiming unearned speedups or offline solves.

- `exp5531-sparse-repair-scaleup-ci` scales the sparse-repair descriptor interface to larger fixtures and
  multi-seed confidence intervals, with exact fallback.
- `exp5532-hardware-receipt-parser-repeatability` repairs CPU/CUDA receipt parsers, records PolarFire/KV260
  reachability classes, and keeps `hardware_speedup_claim=false` unless matched timing exists.
- `exp5533-arc-strategy-routing-precheck` chooses a non-duplicate target and validates strategy routing plus
  repeated-coordinate suppression before a live attempt.
- `exp5534-gated-arc-strategy-routed-levelup` performs the required live ARC level-up attempt with
  `solve_provenance=live_agent_self_discovery`.
- `exp5535-v501-capstone-reconciliation` reconciles artifacts, specs, status, changelog, and claim
  boundaries.

## Dependency Graph

```text
exp5523 transition
  |
  v
exp5524 source delta
  |
  +--> exp5525 SOTA schema taxonomy
  |       |
  |       v
  |     exp5526 SOTA repair loop
  |       |
  |       v
  |     exp5527 SOTA hard/soft panel v2
  |
  +--> exp5528 CSL canonical gate artifact
  |       |
  |       v
  |     exp5529 CSL event/topic residue stress
  |       |
  |       v
  |     exp5530 SOTA CSL memory panel v2
  |
  +--> exp5531 sparse repair scale-up
  |
  +--> exp5532 hardware receipt parser repeatability
  |
  +--> exp5533 ARC strategy-routing precheck
          |
          v
        exp5534 ARC strategy-routed level-up
          |
          v
        exp5535 capstone
```

## Hardware Requirements

- **Dual RTX 3090 CUDA:** Required for headline-eligible local SOTA GGUF tasks. V501 prompts require the
  mandated model specs: `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`. Legacy small GGUFs may be used only as CPU smoke tests.
- **Local GGUF runtime:** Use repository helpers such as `cached_sota_pair()` and never call
  `AutoTokenizer.from_pretrained` on GGUF repositories.
- **KV260:** Use SSH, `xmutil`, and on-board UIO or equivalent authenticated board paths only. Host
  block-device probes are not valid board-state evidence.
- **PolarFire:** Reachability may be recorded, but speedup claims require matched workload hashes and
  board-local timing.
- **GateMate:** Identity/toolchain receipts are useful, but no hardware result can be credited without
  authenticated target evidence.
- **Extropic/TSU and Kona/Aleph:** Watch-only architecture context. No local execution path exists in this
  milestone.

## Guardrails And Expected Outcomes

- `research-roadmap.yaml` and `scripts/research_conductor.py` are protected and must not be modified by the
  planning task or by any experiment prompt.
- Every task that reattempts a previous blocked or bounded scope includes `prior_failures` with
  `retire_if_same_verdict: true`.
- Gated tasks include structured `gated_on` fields whose artifact fields are named in the upstream prompt's
  REQUIRED ARTIFACT FIELDS block.
- ARC solve credit is live-path only. Any ARC level-up task must include `solve_provenance` and count a new
  level only through `offline_reproduced=true` with `reproduced_levels>=1`.
- Hardware speedup remains false unless matched, board-authenticated timing exists.
- The expected success state for `.501` is not a broad PRD victory. It is a narrower executable handoff:
  schema-valid local SOTA rows or a precise taxonomy of why not; conductor-visible CSL gates; a real residue
  stress result; scaled sparse-repair evidence; cleaner hardware receipts; and one live ARC attempt with a
  changed action-routing mechanism.
