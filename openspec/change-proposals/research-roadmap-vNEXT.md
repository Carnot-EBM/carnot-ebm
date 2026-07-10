# Research Roadmap vNEXT - Milestone 2026.07.502

**Milestone title:** Adversarial-Clean SOTA Evidence, Non-Tautological CSL, Finite-State Constraints, Receipt Hygiene, and ARC Live-Path Recovery

**Planner date:** 2026-07-10
**Previous milestone:** 2026.07.501
**Task range:** Exp 5536-5549
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
Hugging Face Papers, GitHub discovery, Semantic Scholar public citation routes for EBT `2507.02092` and
ARM-EBM `2512.15605`, Extropic writing, Logical Intelligence public posts, and Carnot's accumulated
reference history. Non-duplicate actionable items were appended to `research-references.md` under:

`## V502 Planner Refresh - 2026-07-10`

New planning consequences:

- **LLM-FSM** (`arXiv:2602.07032`) motivates a deterministic finite-state exact fixture. This gives
  Carnot a compact, checkable substrate for structured hard/soft reasoning, sparse repair descriptors,
  and future hardware workloads.
- **Gram2Token** (OpenReview public listing) motivates a grammar-table preflight before another flagship
  GGUF schema-validity claim. V502 records backend availability and schema reachability, but does not
  claim decoding speedup unless a real runtime path is exercised.
- **ConstrainPrompt, CRV, XGrammar/llguidance, and JSONSchemaBench** remain the source family for the
  local SOTA structured-output lane. V502 focuses on adversarial-clean substrate and duration receipts
  before interpreting model quality.
- **Continual memory and retrieval-warmed reasoning work** sharpen the CSL lane: memory claims need
  non-identical event/topic metrics, retrieval ablations, stale-evidence rejection, negative-transfer
  checks, and no-weight-mutation receipts.
- **Extropic TSU and Logical Intelligence Kona/Aleph** remain architecture context only. There is no local
  executable TSU, Kona, or Aleph path, so they cannot serve as baselines or speedup evidence.

## What 2026.07.501 Proved

The `.501` milestone completed its task range and closed with a useful bounded capstone. It proved that
several repair paths are possible, but it also left important adversarial flags that must be resolved
before broad claims are defensible.

| Lane | Experiments | Finding |
|------|-------------|---------|
| Transition and source freshness | Exp 5523, 5524 | `.500` claims were carried forward and the source delta was recorded cleanly. |
| SOTA structured output | Exp 5525, 5526 | The schema-failure taxonomy and structured repair loop worked. Repair brought schema validity to `1.0` with exact handoff ready and no missing candidate rows after repair. |
| SOTA hard/soft panel | Exp 5527 | The panel artifact reported schema-valid exact rows, but adversarial verification flagged `DURATION_TOO_SHORT`. The hard/soft SOTA claim is not allowed until substrate and duration are repaired. |
| Continuous self-learning | Exp 5528, 5529, 5530 | The canonical CSL gate artifact was clean and a SOTA CSL memory panel ran, but the residue stress was flagged as tautological because event-only and topic-only scores were identical. Broad CSL claims remain blocked. |
| Sparse repair | Exp 5531 | Exact-checked sparse repair scale-up completed with no speedup claim. This lane is ready for richer finite-state descriptors. |
| Hardware receipts | Exp 5532 | Hardware receipt parsing was repaired enough to identify blockers, but the artifact carried compute-bound/GGUF markers and was flagged for methodology and duration. No hardware speedup claim is allowed. |
| ARC live path | Exp 5533, 5534 | The ARC target and live attempt preserved `solve_provenance=live_agent_self_discovery`, but both artifacts were flagged for too-short/no-seed/no-checksum hygiene, and the live level-up reproduced no new level. |
| Capstone | Exp 5535 | Final `.501` claims were bounded: structured SOTA repair was allowed, hard/soft SOTA and broad CSL remained blocked, sparse repair was bounded, hardware speedup was false, and `arc_registry_delta=0`. |

## Three Biggest Gaps To PRD Vision

1. **Verifiable local SOTA reasoning lacks adversarial-clean live evidence.** The PRD asks for LLMs to
   propose and exact constraints to decide. Carnot now has repairable structured rows, but Exp5527 was
   too short to support a live flagship claim. V502 must either produce authenticated live GGUF evidence
   with real duration/offload receipts or explicitly downgrade to no-live-quality-claim.

2. **Continuous self-learning has a clean gate but a tainted residue metric.** FR-11 requires learning
   from experience without metric leakage. Exp5528 proved the canonical gate path, but Exp5529's
   event/topic scores were identical and therefore not independent evidence. V502 must repair the metric,
   add retrieval-warmed ablations, and only then try cross-model local SOTA memory transfer.

3. **Operational embodiment is still receipt-heavy rather than capability-increasing.** Sparse repair has
   bounded exact evidence, hardware has no matched timing, and ARC has no new live banked level. V502
   should add a finite-state exact substrate, keep hardware receipt hygiene no-LLM and speedup-free unless
   matched timing appears, and clean the ARC live-path substrate before one gated level-up attempt.

## Architecture For V502

```text
 research-program / PRD / architecture / .501 capstone / source refresh / references
                                      |
                                      v
                    +--------------------------------------+
                    | V502 Transition + Source Delta       |
                    +------------------+-------------------+
                                       |
       +-------------------------------+-------------------------------+
       |                               |                               |
       v                               v                               v
+------------------------+     +-------------------------+     +------------------------+
| Adversarial-Clean SOTA |     | Continuous Self-Learning|     | Exact Constraints      |
| - duration/substrate   |     | - residue corrigendum   |     | - LLM-FSM fixture      |
| - grammar preflight    |     | - five-arm retrieval    |     | - sparse descriptors   |
| - live hard/soft v3    |     | - cross-model memory    |     | - exact fallback       |
+-----------+------------+     +-----------+-------------+     +-----------+------------+
            |                              |                               |
            +------------------------------+-------------------------------+
                                           |
                                           v
                           +-------------------------------+
                           | Hardware + ARC Hygiene        |
                           | - no-LLM receipt substrate    |
                           | - no speedup without timing   |
                           | - ARC no-LLM live precheck    |
                           | - one gated live level-up     |
                           +---------------+---------------+
                                           |
                                           v
                           +-------------------------------+
                           | Capstone / Spec Reconciliation|
                           +-------------------------------+
```

## Phase Plan

### Phase 0 - Transition And Source Freshness

**Goal:** Carry `.501` terminal evidence forward and make the execution plan current with new literature.

- `exp5536-transition-v502` records `.501` clean, flagged, and blocked claims, including Exp5527
  duration/substrate, Exp5529 tautology, Exp5532 hardware hygiene, and Exp5533/Exp5534 ARC hygiene.
- `exp5537-v502-source-delta-ingestion` confirms the V502 reference refresh and maps LLM-FSM and
  Gram2Token into concrete experiment hooks.

### Phase 1 - Adversarial-Clean SOTA And Finite-State Fixtures

**Goal:** Repair the local SOTA evidence boundary and add a deterministic exact constraint substrate.

- `exp5538-sota-panel-duration-substrate-corrigendum` reopens the Exp5527 claim boundary and either
  produces authenticated live GGUF duration/offload evidence or records a no-live-quality downgrade.
- `exp5539-gram2token-grammar-table-preflight` checks whether local grammar/table backends are available
  and schema-reachable before relying on grammar-constrained generation.
- `exp5540-gated-sota-hard-soft-live-panel-v3` runs only after the duration and grammar gates are clean.
  It uses the mandated flagship GGUF models with exact-validator authority and no missing-row credit.
- `exp5541-llm-fsm-exact-fixture` creates a deterministic finite-state exact fixture inspired by
  LLM-FSM, with schema, reference synthesis, and SAT/exact checks.

### Phase 2 - Continuous Self-Learning Without Tautology

**Goal:** Convert the `.501` CSL positives into non-tautological, retrieval-aware, conductor-visible
evidence.

- `exp5542-csl-residue-metric-independence-corrigendum` repairs the Exp5529 event/topic residue stress so
  event-only and topic-only metrics are distinct evidence families.
- `exp5543-gated-retrieval-warmed-csl-five-arm-ablation` runs only after the residue corrigendum is clean
  and compares oracle, best-constant, per-query random, shuffled-memory, and aligned-memory arms.
- `exp5544-gated-cross-model-sota-csl-transfer` runs the mandated local SOTA GGUF models only behind the
  clean five-arm gate and tests whether memory learned from one model family transfers to another without
  weight mutation.

### Phase 3 - Sparse Repair, Hardware Hygiene, ARC Live Path, Capstone

**Goal:** Improve operational reach without claiming unearned speedups or offline solves.

- `exp5545-gated-sparse-repair-fsm-descriptor-scale` uses the finite-state exact fixture to create a
  richer sparse-repair descriptor family and records exact fallback plus matched-iteration evidence.
- `exp5546-hardware-receipt-substrate-corrigendum` repairs the Exp5532 artifact class by removing
  accidental GGUF/CUDA live-model markers unless a model actually runs, adding seed/checksum fields, and
  keeping `hardware_speedup_claim=false` without matched authenticated timing.
- `exp5547-arc-no-llm-substrate-precheck` repairs the ARC precheck artifact class: no model specs unless
  an LLM proposer is invoked, explicit seed/checksum, registry duplicate check, and live-agent provenance.
- `exp5548-gated-arc-clean-live-levelup` performs one gated live ARC attempt with
  `solve_provenance=live_agent_self_discovery` and registry banking requirements.
- `exp5549-v502-capstone-reconciliation` reconciles artifacts, gates, status, changelog, and claim
  boundaries without modifying `research-roadmap.yaml` or `scripts/research_conductor.py`.

## Dependency Graph

```text
exp5536 transition
  |
  v
exp5537 source delta
  |
  +--> exp5538 SOTA duration/substrate corrigendum
  |       |
  |       v
  |     exp5539 grammar-table preflight
  |       |
  |       v
  |     exp5540 SOTA hard/soft live panel v3
  |
  +--> exp5541 LLM-FSM exact fixture
  |       |
  |       v
  |     exp5545 sparse repair FSM descriptor scale
  |
  +--> exp5542 CSL residue metric corrigendum
  |       |
  |       v
  |     exp5543 retrieval-warmed CSL five-arm ablation
  |       |
  |       v
  |     exp5544 cross-model SOTA CSL transfer
  |
  +--> exp5546 hardware receipt substrate corrigendum
  |
  +--> exp5547 ARC no-LLM substrate precheck
          |
          v
        exp5548 ARC clean live level-up

all terminal evidence --> exp5549 capstone
```

Structured conductor gates:

- `exp5540` requires `exp5538.sota_panel_duration_corrigendum_ready == true` and
  `exp5539.grammar_table_preflight_ready == true`.
- `exp5543` requires `exp5542.csl_residue_tautology_resolved == true`.
- `exp5544` requires `exp5543.csl_five_arm_ready == true`.
- `exp5545` requires `exp5541.exact_fsm_fixture_ready == true`.
- `exp5548` requires `exp5547.arc_clean_precheck_ready == true`.
- `exp5549` reads all prior artifacts and reports any skipped gates explicitly.

## Hardware Requirements

- **Local SOTA GGUF lane:** Use only the mandated local SOTA GGUF headline models when an experiment needs
  LLM inference:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
  Legacy small models are allowed only as CPU smoke-tests and cannot be headline results.
- **GPU evidence:** SOTA experiments must record model specs, runtime backend, GPU/offload evidence where
  applicable, command lines or helper paths used, measured duration, seeds, and exact-validator handoff.
  Missing rows are not abstentions.
- **Hardware lane:** KV260, GateMate, PolarFire, CUDA, and CPU receipts are receipt-only unless the
  artifact includes matched authenticated workload timing and workload hashes. KV260 access must stay on
  the safe SSH/xmutil/UIO path; no host SD-card probing.
- **ARC lane:** ARC live-path tasks must use `solve_provenance=live_agent_self_discovery`, run the registry
  precheck before targeting a level, and avoid offline source reading, exhaustive ground-truth BFS, or
  per-game hand adapters as headline evidence.

## Claim Boundaries

- A SOTA hard/soft reasoning claim is allowed only if `exp5540` is schema-valid, exact-validated,
  adversarial-clean, and live-substrate evidence from `exp5538` remains clean.
- A continuous self-learning claim is allowed only if `exp5542` resolves the residue tautology, `exp5543`
  shows aligned memory beating shuffled/random controls, and `exp5544` records no weight mutation.
- A sparse repair claim is bounded to exact-checked descriptor evidence unless matched timing appears.
- A hardware speedup claim is false by default and can become true only with matched authenticated timing.
- ARC solve credit is allowed only for live-agent self-discovery that is banked in the registry with
  `offline_reproduced=true` and `reproduced_levels>=1`.

## Expected Exit Criteria

By the end of `.502`, Carnot should have:

1. A clean answer on whether the `.501` SOTA hard/soft panel can become live evidence or must remain a
   repaired-structured-output fixture.
2. A grammar-table availability receipt that prevents another blind schema-validity rerun.
3. A deterministic finite-state exact fixture ready for SOTA, sparse repair, and future hardware tasks.
4. CSL evidence that separates event-only, topic-only, shuffled, random, and aligned memory effects.
5. Hardware and ARC artifacts that no longer trip avoidable methodology flags for wrong substrate,
   missing seeds, or missing checksums.
6. One gated ARC live-path level-up attempt with honest registry outcome, whether positive or null.
