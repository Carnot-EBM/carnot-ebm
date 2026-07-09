# Research Roadmap vNEXT - Milestone 2026.07.499

**Milestone title:** Pretest Recovery, Hard/Soft Verification Core, Trajectory-Aware CSL, Hardware Receipts, and ARC Live-Path Generation

**Planner date:** 2026-07-09
**Previous milestone:** 2026.07.498
**Task range:** Exp 5496-5509
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

The planner performed a 2025-2026 refresh across arXiv, OpenReview public
pages where reachable, Extropic writing, Semantic Scholar-style citation
routes for EBT `2507.02092` and ARM-EBM `2512.15605`, Hugging Face Papers,
GitHub repository discovery, and Logical Intelligence public pages. Actionable
non-duplicates were appended to `research-references.md` under:

`## V499 Planner Refresh - 2026-07-09`

New planning consequences:

- **Trajel** (`arXiv:2605.24219`) argues for trajectory-level hallucination
  audits over multi-step agent traces. V499 applies this to the pretest
  cascade, SOTA panels, CSL memory, and capstone failure taxonomy.
- **RT4CHART** (`arXiv:2603.27752`) motivates local-to-global claim
  verification with span/evidence mappings. V499 uses it in helper contracts
  and hard/soft claim fixtures.
- **ExpGraph** (`arXiv:2605.30712`) and **Evo-Memory** (`arXiv:2511.20857`)
  motivate executor-frozen, graph/stream self-learning with explicit
  negative-transfer checks. V499 makes this the CSL path before any headline
  GGUF panel.
- **MILP-Evolve** provides a varied hard/soft constraint fixture source.
  V499 uses it only for executable exact-solver descriptors, not unverified
  heuristic solving.
- **Hamon** is a useful GPU EBM sampler API reference, but it is not a basis
  for a Carnot speedup claim without authenticated matched timing.
- **Extropic TSU/XTR-0** and **Logical Intelligence Kona** remain strategic
  architecture references only. V499 hardware evidence must come from local
  CUDA, KV260, GateMate, PolarFire, or CPU receipts.

## What 2026.07.498 Proved

The operator reports `.498` complete. Its capstone and conductor trail make the
main lesson unusually concrete: many planned science tasks did not fail
scientifically; they never executed because the pretest/self-heal path skipped
them before artifacts existed. The correct next milestone begins by fixing that
execution reliability gap.

| Lane | Experiments | Finding |
|------|-------------|---------|
| Transition and planning | 5482, 5495 | `.498` carried `.497` facts forward and closed with a capstone that preserved guided-decoding quarantine, bounded CSL, no hardware speedup claim, and no ARC registry delta. |
| Source delta and core fixtures | 5483, 5485, 5486, 5487 | Execution was blocked or skipped by pretest failures, so source delta, Preference-MaxSAT, concept telemetry, and helper-contract repair did not produce usable artifacts. |
| Continuous self-learning | 5484, 5488, 5489, 5490 | The required CSL corrigendum and independent-metric lanes did not land; Exp5474's tautology finding still blocks CSL headline claims. Exp5490 gate-blocked because its prerequisite artifact was missing. |
| Active constraints | 5491 | Active-constraint subproblem descriptors completed and remain a usable seed for hardware and exact-fallback descriptors. |
| Hardware | 5492 | PolarFire remained reachable with matched hashes. KV260 was blocked by SSH identity, GateMate by JTAG identity, and `hardware_speedup_claim=false` stayed correct. |
| ARC | 5493, 5494 | The ARC precheck selected `dc22 L3`, but the live attempt was an honest null and was later methodologically flagged for too-short duration/missing methodology. Registry delta stayed zero. |

## Three Biggest Gaps

1. **Execution reliability is now the first scientific blocker.** The PRD
   cannot progress if source-delta, CSL, helper, and MaxSAT lanes are skipped
   before artifacts are emitted. V499 front-loads a pretest cascade diagnostic
   and repair receipt without modifying `scripts/research_conductor.py`.

2. **The verifiable reasoning core lacks the hard/soft fixture layer.** Carnot
   has exact guards and active-constraint descriptors, but `.498` failed to
   ship the minimal Preference-MaxSAT and helper-contract artifacts needed to
   connect hard admissibility, soft preference ranking, claim evidence, and
   local SOTA concept telemetry.

3. **FR-11/ARC/hardware grounding remains bounded.** Continuous self-learning
   is still blocked by the Exp5474 tautology issue, ARC is stuck at a 69-level
   plateau with repeated no-bank attempts, and hardware receipts are real but
   not a speedup result. V499 uses graph/stream memory, live ARC perception
   generation, and multi-board receipt continuity to move these bounded lanes
   without overstating them.

## Target Architecture

```text
                          +---------------------------------------+
                          |  Milestone execution substrate         |
                          |  pretests, gates, artifacts, capstone  |
                          +-------------------+-------------------+
                                              |
                     pretest taxonomy, source delta, gate receipts
                                              |
        +-------------------------------------v----------------------------------+
        | Hard/soft verification core                                            |
        | exact hard constraints, Preference-MaxSAT ranking, hierarchical claims, |
        | helper contracts compiled to predicates, exact fallback references      |
        +----------------------+-------------------------------+-----------------+
                               |                               |
                 SOTA GGUF evidence panel                      | descriptor stream
                               |                               |
        +----------------------v-------------------+   +-------v----------------+
        | Local SOTA inference with receipts        |   | Active constraints     |
        | Qwen3.6-35B-A3B, Gemma-4-31B-it,          |   | MILP/MaxSAT/CSP rows,  |
        | Gemma-4-26B-A4B-it via llama.cpp/CUDA     |   | exact fallback checks  |
        +----------------------+--------------------+   +-------+----------------+
                               |                            board descriptors
                               |                                  |
        +----------------------v--------------------+   +---------v--------------+
        | Continuous self-learning                  |   | Hardware receipt path  |
        | metric-independence corrigendum,          |   | CUDA, KV260, GateMate, |
        | graph/stream memory, frozen executor,     |   | PolarFire, no speedup  |
        | no-memory baseline, negative transfer     |   | claim without timing   |
        +----------------------+--------------------+   +------------------------+
                               |
                               | learned lessons and trajectory taxonomy
                               |
        +----------------------v-----------------------------------------------+
        | ARC live path                                                         |
        | registry precheck, null-coordinate audit, classical perception,        |
        | salience-tiered action generation, live_agent_self_discovery solve     |
        | provenance if a level is actually reproduced                           |
        +----------------------------------------------------------------------+
```

## Phase Plan

### Phase 0 - Transition, Pretest Recovery, and Source Delta

- **Exp5496:** transition `.498` terminal facts into `.499` execution context.
- **Exp5497:** diagnose and repair/receipt the `.498` pretest skip cascade
  without touching `scripts/research_conductor.py`.
- **Exp5498:** run execution-time 2025-2026 source delta only after the pretest
  cascade receipt indicates the lane can execute.

### Phase 1 - Hard/Soft Verification Core

- **Exp5499:** build the minimal Preference-MaxSAT typed claim-state fixture
  with hard exact constraints, soft preferences, executable references, and
  false-accept accounting.
- **Exp5500:** if Exp5499 is ready, run a local SOTA GGUF concept/claim panel
  with mandatory flagship GGUF model receipts and exact validators as final
  authority.
- **Exp5501:** recover helper-contract repair using hierarchical
  claim/evidence mappings and executable predicates.
- **Exp5505:** extend active-constraint descriptors with MILP/MaxSAT/CSP rows
  and exact fallback semantics.

### Phase 2 - Continuous Self-Learning With Independent Metrics

- **Exp5502:** produce the CSL tautology corrigendum required before any
  Exp5474-style CSL headline can be used.
- **Exp5503:** build an executor-frozen ExpGraph/Evo-Memory style streaming
  replay fixture with no-memory baseline and negative-transfer checks.
- **Exp5504:** if Exp5502 and Exp5503 are clean, run a local SOTA GGUF CSL
  memory panel with independent metrics and frozen model weights.

### Phase 3 - Hardware, ARC, and Synthesis

- **Exp5506:** collect multi-board receipts across PolarFire, KV260, GateMate,
  CUDA, and CPU where reachable, using active-constraint descriptors or the
  last clean fallback descriptor. No speedup claim is allowed without matched
  timing.
- **Exp5507:** precheck ARC target eligibility, null-coordinate validity, and
  perception grounding before any level-up attempt.
- **Exp5508:** if Exp5507 is ready, run one live ARC perception-generation
  level-up attempt with changed mechanism, `solve_provenance` discipline, and
  an explicit `offline_reproduced=true` plus `reproduced_levels>=1` banking
  gate for any new level claim.
- **Exp5509:** synthesize `.499` actual artifacts into a capstone with PRD gap
  table, gate truth table, failure taxonomy, hardware/ARC truth claims, and
  next recommendations.

## Natural Next-Experiment Chain

```text
.498 pretest skip cascade
  -> Exp5497 pretest diagnostic/repair receipt
      -> Exp5498 source delta
      -> Exp5499 Preference-MaxSAT fixture
      -> Exp5502 CSL corrigendum

Exp5499 hard/soft fixture
  -> Exp5500 SOTA concept/claim panel
  -> Exp5501 helper-contract hierarchical claim fixture
  -> Exp5505 active-constraint MILP/MaxSAT descriptors

Exp5474 TAUTOLOGY + missing .498 CSL artifacts
  -> Exp5502 CSL metric-independence audit
  -> Exp5503 graph/stream memory replay
      -> Exp5504 SOTA GGUF CSL memory panel

Exp5491 descriptors + Exp5492 hardware receipts
  -> Exp5505 descriptor extension
  -> Exp5506 multi-board hardware receipts

Exp5494 dc22 L3 no-bank and methodology flag
  -> Exp5507 ARC null-coordinate/perception precheck
      -> Exp5508 ARC live perception-generation attempt

All lanes
  -> Exp5509 capstone
```

## Dependency Graph

```text
5496 transition
  |
  v
5497 pretest cascade diagnostic
  +--> 5498 source delta
  +--> 5499 Preference-MaxSAT fixture
  |       +--> 5500 SOTA concept/claim panel
  |       +--> 5501 helper-contract claim fixture
  |       +--> 5505 active-constraint descriptors
  |               +--> 5506 hardware receipts
  |
  +--> 5502 CSL tautology corrigendum
          +--> 5503 graph/stream memory replay
                  +--> 5504 SOTA CSL memory panel

5507 ARC precheck
  +--> 5508 ARC live perception-generation attempt

5496..5508
  +--> 5509 capstone
```

## Hardware Requirements

| Experiment | Required substrate | Preconditions | Claim discipline |
|------------|--------------------|---------------|------------------|
| 5500 | Local CUDA plus llama.cpp/GGUF for at least one mandated SOTA model | cached GGUF file exists, llama.cpp reports CUDA/GPU offload, GPU memory delta captured | Concept evidence only; exact validators decide. |
| 5504 | Local CUDA plus llama.cpp/GGUF for at least one mandated SOTA model | same as Exp5500 plus clean CSL independence gate | Frozen weights only; no CSL headline if metrics are not independent. |
| 5506 | CPU, CUDA, `ssh polarfire`, `ssh kria`, `openFPGALoader -c dirtyJtag --detect` | no host `/dev/mmcblk*` KV260 probing; record blocked boards honestly | Receipt-only unless authenticated matched timing exists. |
| 5508 | Local ARC live agent runtime | registry precheck, no duplicate solve, no offline BFS/per-game hand adapter | Any solve must be `live_agent_self_discovery`. |

Mandated SOTA local GGUF models for every LLM-bearing experiment:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small GGUF models may appear only as CPU smoke tests and must not be
headline models.

## Risk Register and Guardrails

- **Pretest cascade repeats:** Exp5497 must emit a concrete taxonomy and
  receipt before gated science tasks run. If the same skip class remains, do
  not pretend downstream artifacts exist.
- **Failed-experiment reruns:** All same-scope reruns include `prior_failures`
  with `retire_if_same_verdict: true`.
- **Guided decoding quarantine:** V499 does not perform token-level guided
  decoding. SOTA panels generate or score complete candidates and then apply
  exact validators.
- **CSL tautology:** Exp5504 is gated on independent metrics. Exp5503 still
  runs as a new graph/stream fixture so the milestone satisfies FR-11 without
  depending on Exp5474.
- **ARC duplicate/off-path solves:** Exp5507 must read
  `ops/arc_solve_registry.yaml` and avoid already-reached levels. Exp5508 may
  not use offline ground-truth BFS or per-game hand adapters as credited solve
  paths.
- **Hardware overclaim:** PolarFire/KV260/GateMate/CUDA receipts may be useful
  even when blocked, but speedup claims require authenticated matched timing.
- **Protected files:** `research-roadmap.yaml` and
  `scripts/research_conductor.py` are not modified by this planning turn.

## Expected Outcomes

1. A clean answer to whether `.498` skipped because of a fixable pretest
   failure, missing tests, or source-level regressions.
2. A minimal hard/soft verification artifact that can support future local
   SOTA concept telemetry.
3. A CSL lane that is independent, graph/stream based, and honest about
   negative transfer.
4. Continued hardware and ARC standing-floor evidence without speedup or solve
   overclaiming.
5. A `.499` capstone that decides which lanes are headline-ready, bounded,
   blocked, retired, or ready for the next milestone.
