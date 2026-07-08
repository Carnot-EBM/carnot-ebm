# Research Roadmap vNEXT - Milestone 2026.07.494

**Milestone title:** Structured Corrigenda, Verified Workflow Memory, ARC Live Reinduction, and Hardware Timing Boundaries

**Planner date:** 2026-07-08
**Previous milestone:** 2026.07.493
**Task range:** Exp 5428-5440
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
- `ops/arc_solve_registry.yaml`
- `scripts/experiment_template.py`
- `scripts/roadmap_schema.py`
- `scripts/audit_roadmap_gates.py`
- `scripts/arc_levelup_guarantee_lint.py`
- `scripts/exclusion_manifest_lint.py`
- `.493` artifacts under `results/`

## Literature Refresh Incorporated

The planner performed a 2025-2026 refresh across arXiv, OpenReview,
Extropic writing, Semantic Scholar citation routes for EBT `2507.02092` and
ARM-EBM `2512.15605`, HuggingFace Papers, GitHub, and Logical Intelligence
public pages before designing experiments. Actionable findings were appended
to `research-references.md` under `### V494 Planner Refresh - 2026-07-08`.

Promoted sources and planning consequences:

- **A-MEM: Agentic Memory for LLM Agents** (`arXiv:2502.12110`): motivates
  memory graphs with structured attributes and links, but `.494` keeps memory
  updates deterministic, provenance-backed, rollbackable, and no-weight.
- **Memory-Augmented Reinforcement Learning Agent for CAD Generation**
  (`arXiv:2605.19748`): motivates verified workflow memory over a
  kernel/toolchain fixture with case and skill memory. `.494` uses this as a
  deterministic CSL fixture, not as a CAD or RL claim.
- **Automatic Ontology Construction Using LLMs as an External Layer of Memory,
  Verification, and Planning** (`arXiv:2604.20795`): motivates an
  ontology-backed constraint-memory fixture with SHACL-style validation and
  deterministic planner authority.
- **Symbolic-Neural Soft-Logic Reasoning** (`arXiv:2605.25618`): motivates
  advisory soft-logic residuals while exact finite-domain solvers remain final
  authority.
- **Logical Intelligence Aleph formal-verification posts**: reinforce the
  candidate-generator plus deterministic-checker architecture. Carnot has no
  local Aleph/Kona baseline, so these remain architecture context only.

Secondary-source status:

- OpenReview again surfaced Spilled Energy, EBT, neural-CSP correctness, and
  constrained-diffusion work. These are duplicate, retired/noisy, or already
  covered; no generated-text or token-internal lane is reopened.
- HuggingFace Papers reinforced constrained-decoding and verification themes,
  but did not add a stronger local baseline than the arXiv deltas.
- Semantic Scholar public routes for EBT and ARM-EBM resolved mainly to source
  papers and mirrors; no citation-derived experiment is promoted.
- GitHub repositories such as `AgenticMemory`, `agiresearch/A-mem`, KAN lists,
  constrained-decoding lists, and ARC exploration repos remain watch
  references. None replace Carnot's deterministic solver or ARC provenance
  rules.
- Extropic X0/XTR-0 and TSU writing remains hardware-watch context only; no
  authenticated local TSU execution path exists.

## What 2026.07.493 Proved

The `.493` capstone is the planning source of truth:

- **Continuous self-learning is headline-ready at controller level.**
  Exp5421 detected hidden forgetting under stable accuracy while preserving
  raw episodes, rollback, and no weight mutation. Exp5422 promoted only
  threshold-passing fragments and kept rejected/abstained fragments inactive.
- **Structured verification still has a corrigendum gap.** Exp5417 and Exp5418
  had valid local SOTA GGUF/offload receipts and useful row artifacts, but both
  were blocked from headline use by adversarial tautology flags: Exp5417
  collapsed abstention and semantic-error metrics, and Exp5418 collapsed
  final-only action-unreachability and its delta. Reproducibility checksum
  gaps were also reported.
- **Active-constraint guidance is bounded and useful.** Exp5419 scaled LNS
  hints to five fixtures, reduced solver work by 234, and preserved solver
  authority.
- **P-bit and hardware lanes have receipt discipline but no acceleration
  claim.** Exp5420 and Exp5424 produced hash-matched CPU/PolarFire preflight
  and timing receipts. The board was slower in the observed receipts, and
  `hardware_speedup_claim=false` remained correct.
- **KAN certificates remain bounded.** Exp5425 cleanly rejected observable
  false active-constraint claims and unsupported board/token/internal claims,
  while preserving true measured properties. It did not make a broad KAN
  verification claim.
- **ARC did not bank a new level.** Exp5423 was an honest null on `lf52` L3:
  46 attempts, 22 frontier expansions, 45 landmarks, no offline BFS, no
  per-game adapter, registry total unchanged at 69.
- **Token/internal feature lanes remain closed.** No artifact supplied
  authenticated logits, hidden states, attention, token-logprob, or
  intermediate-exit receipts.

## Three Biggest Gaps

1. **Structured verifier evidence is not adversarial-clean.** The PRD requires
   verifiable reasoning, not plausible aggregate metrics. `.494` first repairs
   row-level provenance, metric independence, and checksums before any scale-up.

2. **CSL has safe promotion but not verified workflow memory.** `.493` proved
   no-weight controller learning and reliance drift checks. The PRD vision
   needs durable, reusable learning units with explicit kernel/ontology
   verification, retrieval-trap deflection, and transfer stress.

3. **Live evidence still lacks north-star movement.** ARC has repeated
   no-bank attempts, hardware has no speedup claim, and token/internal lanes
   are closed. `.494` keeps one concrete live ARC level-up attempt, upgrades
   hardware timing to variance/receipt boundaries, and avoids unsupported
   internal-signal claims.

## Target Architecture

```text
                         +--------------------------------------+
                         | Local SOTA GGUF inference substrate  |
                         | Qwen3.6-35B-A3B, Gemma-4-31B-it,     |
                         | Gemma-4-26B-A4B-it via llama.cpp     |
                         +-------------------+------------------+
                                             |
                      structured candidates + row-level provenance
                                             |
        +------------------------------------v----------------------------------+
        | Structured corrigendum and replication layer                          |
        | independent metrics, checksums, risk calibration, prefix/action safety |
        | deterministic verifiers are final authority                           |
        +-------------------+--------------------------+------------------------+
                            |                          |
           ontology/soft-logic constraint memory       | active constraints
                            |                          |
        +-------------------v----------------+   +-----v------------------------+
        | Verified workflow CSL controller   |   | Solver, p-bit, hardware      |
        | raw episodes, case/skill memory,   |   | LNS hints, exact solvers,    |
        | SHACL/kernel gates, rollback       |   | CPU/PolarFire timing receipts |
        +-------------------+----------------+   +-----+------------------------+
                            |                          |
                            +------------+-------------+
                                         |
        +--------------------------------v--------------------------------------+
        | Live evidence and bounded certificates                                |
        | ARC self-discovery level-up attempt, KAN measurement-access certs,     |
        | PRD gap table; token/internal lanes stay closed without receipts       |
        +-----------------------------------------------------------------------+
```

## Phase Plan

### Phase 0 - Transition, Source Delta, and Structured Corrigenda

- **Exp5428:** archive `.493` terminal evidence and stage `.494` task range.
- **Exp5429:** run execution-time source delta against the V494 planner
  refresh and append only non-duplicate actionable references.
- **Exp5430:** repair Exp5417/Exp5418 structured tautology and checksum
  gaps with row-level provenance and mandatory local SOTA GGUF receipts.
- **Exp5431:** if Exp5430 is clean, replicate structured constraint-taxonomy
  results at bounded scale with independent semantic, abstention, risk, and
  action-reachability metrics.

### Phase 1 - Constraint Memory, Active Constraints, and Hardware Boundaries

- **Exp5432:** convert ontology and soft-logic literature into a deterministic
  constraint-memory fixture with SHACL-style validation and exact solver
  authority.
- **Exp5433:** scale active-constraint LNS with subproblem diversity,
  conflict-front hints, and solver accept/reject/overwrite accounting.
- **Exp5434:** if Exp5433 is ready, collect CPU/PolarFire p-bit timing
  variance receipts with matched hashes and no speedup claim.

### Phase 2 - Verified Workflow Continuous Self-Learning

- **Exp5435:** required CSL experiment: build verified workflow memory with
  case/skill memory, ontology/kernel validation, retrieval-trap deflection,
  raw episodes, rollback, and no weight mutation.
- **Exp5436:** if Exp5435 is ready, stress CSL transfer under distribution
  shifts and negative-transfer controls.

### Phase 3 - ARC, Certificates, and Synthesis

- **Exp5437:** run one concrete ARC live-path level-up attempt via
  registry-checked reinduction/runtime self-discovery. Count only
  reproduction-gated `offline_reproduced=true` and `reproduced_levels>=1`.
- **Exp5438:** if Exp5432 is ready, extend bounded KAN measurement-access
  certificates to ontology and workflow-memory claims.
- **Exp5439:** aggregate `.494` evidence into a PRD gap and failure-taxonomy
  table.
- **Exp5440:** emit the `.494` capstone truth table and recommendations.

## Dependency Graph

```text
exp5428 transition
  -> exp5429 source delta
  -> exp5430 structured tautology corrigendum
      -> exp5431 structured taxonomy replication

V494 literature + Exp5421/Exp5422 CSL evidence
  -> exp5432 ontology/soft-logic constraint memory
      -> exp5438 KAN ontology measurement certificate

Exp5419 active-constraint evidence
  -> exp5433 active-constraint diversity LNS
      -> exp5434 p-bit PolarFire timing variance

Exp5421/Exp5422 CSL evidence + Exp5432 constraint memory
  -> exp5435 verified workflow memory CSL
      -> exp5436 CSL transfer stress

Exp5423 honest ARC null + arc_solve_registry
  -> exp5437 live ARC reinduction level-up attempt

exp5430, exp5431, exp5432, exp5433, exp5434, exp5435, exp5436, exp5437, exp5438
  -> exp5439 PRD gap table
      -> exp5440 capstone
```

## Hardware Requirements

| Experiment | Compute | Hardware notes | Claim boundary |
|-----------|---------|----------------|----------------|
| 5428, 5429, 5432, 5433, 5435, 5436, 5438-5440 | CPU | Repository artifacts and deterministic fixtures | No live model or hardware speedup claim |
| 5430, 5431 | Dual RTX 3090 preferred | Must use CUDA-enabled `llama-cpp-python`/GGUF runtime; MODEL_SPECS include all mandated SOTA GGUFs | Block if offload/runtime receipts are absent |
| 5434 | CPU plus PolarFire if reachable | Same workload hash, result hash, repeat counts, timing distribution; KV260 SSH-only if checked; GateMate diagnostic-only unless physical/JTAG evidence returns | `hardware_speedup_claim=false` unless matched repeated board timing justifies a future claim |
| 5437 | CPU ARC offline/live runtime | Registry precheck, live-agent self-discovery only, no offline source/BFS/per-game adapter credited path | Registry count changes only after reproduction-gated new level |

## Risk Controls

- **Structured flags:** Exp5431 is gated on Exp5430
  `structured_corrigendum_clean=true`. The roadmap must not scale blocked
  `.493` structured evidence if the same tautology verdict recurs.
- **GGUF runtime:** Any LLM task must include the three mandated local SOTA GGUF
  model specs and fail fast if CUDA/offload receipts are missing.
- **CSL safety:** Verified workflow memory cannot mutate model weights, and
  promoted memory must retain raw episodes, rollback pointers, reliance
  metrics, and verifier provenance.
- **Hardware honesty:** PolarFire timing can report slower board execution.
  That is still useful evidence. The milestone does not need, and must not
  fabricate, a speedup claim.
- **ARC discipline:** The live agent's own runtime attempts and runtime reverse
  engineering are the credited path. Offline BFS, hidden source reading, and
  per-game adapters are not headline solves.
- **Token/internal closure:** Token, hidden-state, attention, logprob, and
  intermediate-exit lanes remain closed unless a future backend artifact
  authenticates those receipts.

## Expected Outcomes

- A clean go/no-go decision on whether `.493` structured local SOTA evidence
  can be rehabilitated.
- A bounded ontology/soft-logic constraint-memory fixture that supports CSL
  workflow memory without trusting open-ended memory evolution.
- Stronger active-constraint and p-bit timing receipts with no unsupported
  acceleration claims.
- One live ARC level-up attempt that either banks a reproduction-gated new
  level or emits an honest null with reusable frontier evidence.
- A capstone that clearly separates headline-ready, bounded, honest-null, and
  blocked lanes for the next planner.
