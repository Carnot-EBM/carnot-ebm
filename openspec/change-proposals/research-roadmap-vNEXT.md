# Research Roadmap vNEXT - Milestone 2026.07.492

**Milestone title:** Formal Corrigendum, Resource-Aware Self-Learning, Active-Constraint Guidance, and Live Evidence Recovery

**Planner date:** 2026-07-08
**Previous milestone:** 2026.07.491
**Task range:** Exp 5402-5414
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
- `.491` artifacts under `results/`

## Literature Refresh Incorporated

The planner performed a 2025-2026 source refresh before designing the
experiments and appended the actionable findings to `research-references.md`
under `### V492 Planner Refresh - 2026-07-08`.

Promoted sources and planning consequences:

- **SWEnergy** (`arXiv:2512.09543`): motivates extending FR-11 self-learning
  receipts from correctness and verifier-cost deltas to wall-time, token,
  memory, and unproductive-loop accounting.
- **Warm-starting active-set solvers with GNNs** (`arXiv:2511.13174`):
  motivates active-constraint and conflict-front hints where the deterministic
  solver remains authoritative, instead of neural full-solution proposals.
- **UPSi uncertainty-aware predictive safety filters** (`arXiv:2604.26836`):
  motivates uncertainty-gated promotion of learned memory/world-model
  fragments before they can affect routing or live ARC attempts.
- **HaluNet** (`arXiv:2512.24562`): remains watch-only because Carnot still
  lacks authenticated token/internal backend receipts. It does not reopen the
  retired external generated-text scorer lane.
- **BitsMoE** (`arXiv:2606.00079`): informs runtime awareness for MoE GGUF
  variants, but is not a headline experiment without local quant variants and
  repeatable offload receipts.

Secondary-source status:

- OpenReview and HuggingFace Papers reinforced constrained generation,
  interactive verification, and verifier-first evaluation, but did not add a
  new local baseline beyond the arXiv items above.
- Semantic Scholar routes for EBT `2507.02092` and ARM-EBM `2512.15605` did
  not produce a fresh local delta beyond already-recorded NRGPT, fixed-point,
  and distributional EBM references.
- Extropic TSU/X0/XTR and Logical Intelligence Kona/Aleph pages remain
  architecture context only. Carnot has no executable local TSU, Kona, or Aleph
  baseline for `.492`.

## What 2026.07.491 Proved

The `.491` capstone (`results/experiment_5401_capstone_v491.json`) left a
clear split between bounded wins and lanes that still need repair:

- **Structured constraints scaled:** Exp5391 produced a 24-fixture deterministic
  panel where constrained final-state/tool-action rows reached semantic
  validity 1.0, unconstrained rows reached 0.0, and unsafe false accepts stayed
  at 0. The evidence supports scaling structured local SOTA receipts.
- **Formal-encoding safety is not headline-ready:** Exp5392 emitted a
  formal-encoding artifact, but the conductor adversarial check flagged it
  `CRITICAL TAUTOLOGY`. `.492` must repair this before using the safety fixture
  as evidence.
- **Solver guidance improved under deterministic authority:** Exp5393 cleaned
  the overwrite-guidance tautology issue, and Exp5394 showed CPU-only p-bit
  boundary hints improved aggregate solver conflict delta while preserving
  validity and fallback. Hardware transfer remains unproved.
- **Continuous self-learning reached a real controller baseline:** Exp5395 and
  Exp5396 produced influence-share routing, verifier-cost reduction, stale and
  poison deflection, rollback, raw episode retention, and no model-weight
  mutation. `.492` should add resource accounting and uncertainty gates.
- **ARC still did not bank a live level:** Exp5397 honestly attempted re86 L3
  through live-agent self-discovery and returned `honest_null:
  bounded_budget_no_levelup`. Another salience-only rerun is not justified.
- **Hardware repeatability is absent:** Exp5398 found KV260 unreachable,
  PolarFire repeat count 0, GateMate workload path false, and no speedup claim.
  `.492` should restore reachability and repeated same-workload receipts only.
- **KAN certificates are useful but bounded:** Exp5399 emitted a dynamic
  counterexample certificate with false-property rejection 1.0 and true-property
  preservation 1.0, while explicitly avoiding a broad KAN verification claim.
- **Token/internal feature lanes remain closed:** No `.491` artifact reopened
  local logits, hidden-state, attention, or early-exit evidence.

## Three Biggest Gaps

1. **Formal safety/verifier evidence is still fragile.** The PRD needs
   verifiable, local constraint reasoning. Exp5391 is promising, but Exp5392's
   tautology flag means formal-encoding safety cannot be used until row-level
   evidence, checksums, and independent false/true-property controls are clean.

2. **Self-learning is not yet resource- and uncertainty-aware.** FR-11 now has
   a good no-weight-mutation controller baseline, but the long-term vision needs
   durable learning that accounts for resource waste and refuses uncertain
   memory/world-model promotion before it alters routing.

3. **Live external evidence is still the frontier.** ARC did not bank a level,
   hardware lacks repeatability, and token/internal receipt paths remain closed.
   `.492` must improve live self-discovery and board evidence without making
   unsupported speedup or hidden-feature claims.

## Target Architecture

```text
                         +--------------------------------------+
                         | Local SOTA GGUF inference substrate  |
                         | Qwen3.6-35B-A3B, Gemma-4-31B-it,     |
                         | Gemma-4-26B-A4B-it via GGUF runtime  |
                         +-------------------+------------------+
                                             |
                               structured candidates + receipts
                                             |
        +------------------------------------v-----------------------------------+
        | Formal and structured constraint verifier layer                         |
        | schema checks, trace checks, policy/safety encodings, checksums, rows   |
        | deterministic verifier is final authority                              |
        +---------------+------------------------------+-------------------------+
                        |                              |
        active constraints/conflict fronts             | resource-aware routing
                        |                              |
        +---------------v--------------+   +-----------v------------------------+
        | Solver and p-bit diagnostics |   | Continuous self-learning controller |
        | solver accepts/rejects/       |   | influence shares, raw episodes,     |
        | overwrites hints; CPU first   |   | rollback, uncertainty gates         |
        +---------------+--------------+   +-----------+------------------------+
                        |                              |
                        +--------------+---------------+
                                       |
        +------------------------------v----------------------------------------+
        | Live evidence surfaces                                                 |
        | ARC trajectory/frontier generation, hardware repeat receipts, KAN      |
        | counterexample certificates; token/internal lanes stay closed          |
        +-----------------------------------------------------------------------+
```

## Phase Plan

### Phase 0 - Transition, Source Delta, and Formal Repair

- **Exp5402:** archive `.491`, stage `.492`, and record the real closed/open
  evidence set.
- **Exp5403:** run the execution-time source delta and keep
  `research-references.md` current without reopening retired scopes.
- **Exp5404:** repair the Exp5392 formal-encoding tautology flag using
  checksum-backed row evidence, independent false/true controls, and mandated
  local SOTA GGUF receipts.
- **Exp5405:** if Exp5404 is clean, scale the structured safety/action panel
  across the mandated local SOTA GGUF models.

### Phase 1 - Active-Constraint Solver Guidance

- **Exp5406:** implement active-constraint/conflict-front warm-start guidance
  where the solver remains authoritative and can overwrite every hint.
- **Exp5407:** if Exp5406 is clean, stress the p-bit/action-sequence QUBO lane
  with active-constraint hints and a sorting-network micro-baseline, CPU only.

### Phase 2 - Resource-Aware Continuous Self-Learning

- **Exp5408:** extend the Exp5395/5396 controller into a resource-accounted
  self-learning router with wall-time, token, memory, unproductive-loop, quality,
  rollback, and no-weight-mutation receipts.
- **Exp5409:** if Exp5408 is clean, add UPSi-style uncertainty gates for memory
  and world-model promotion before learned fragments can affect routing.

### Phase 3 - Live Evidence Recovery and Capstone

- **Exp5410:** attempt an ARC live-path level-up using trajectory/frontier
  generation plus blob salience and uncertainty gates. The deliverable is live
  agent self-discovery, not offline BFS or per-game reverse engineering.
- **Exp5411:** restore hardware repeatability evidence across available boards.
  KV260 checks are SSH-only; PolarFire requires repeated same-workload receipts;
  GateMate remains diagnostic-only unless the physical/JTAG path is restored.
- **Exp5412:** extend the bounded KAN/KANDy counterexample certificate to a new
  false-property family tied to routing or active constraints.
- **Exp5413:** aggregate `.492` receipts into a PRD gap table.
- **Exp5414:** emit the `.492` capstone truth table and recommendations for
  the next milestone.

## Dependency Graph

```text
exp5402 transition
  -> exp5403 source delta
  -> exp5404 formal-encoding corrigendum
      -> exp5405 gated structured safety/action scale-up

Exp5393/Exp5394 prior solver evidence
  -> exp5406 active-constraint warm-start guidance
      -> exp5407 gated p-bit/QUBO stress

Exp5395/Exp5396 prior self-learning evidence
  -> exp5408 resource-accounted CSL controller
      -> exp5409 uncertainty-gated memory/world-model promotion

Exp5397 no-bank ARC attempt + arc_solve_registry
  -> exp5410 live ARC trajectory/frontier level-up attempt

Exp5398 hardware repeatability absence
  -> exp5411 hardware repeatability restoration

Exp5399 KAN certificate
  -> exp5412 expanded KAN certificate

exp5405, exp5407, exp5409, exp5410, exp5411, exp5412
  -> exp5413 evidence table
  -> exp5414 capstone
```

## Hardware Requirements

- **Local SOTA GGUF inference:** Exp5404 and Exp5405 require authenticated
  local GGUF runtime evidence for at least one mandated SOTA model in headline
  rows, and should include all three model specs in the artifact:
  `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`. Legacy small models are smoke-test only.
- **CUDA/offload:** Any SOTA GGUF headline row must record runtime, command,
  model path, n_gpu_layers or equivalent, memory/offload evidence, duration,
  and a no-CPU-headline assertion. `llama.cpp` commands must use single-turn
  mode where applicable.
- **KV260:** SSH reachability only: `ssh -o ConnectTimeout=5 -o BatchMode=yes
  kria 'true'`. Do not use host SD-card or `/dev/mmcblk*` probes.
- **PolarFire:** repeated same-workload receipts are required before claiming
  repeatability. No speedup claim is allowed unless authenticated repeated
  board timing exists.
- **GateMate:** diagnostic-only unless DirtyJTAG/physical/JTAG reachability is
  restored. No destructive probes.
- **Hardware speedup:** default `hardware_speedup_claim=false` for `.492`.

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not reopen retired external generated-text scorer, retired ARC
  salience-only first-contact, CPU-only SOTA headline, or hardware speedup lanes.
- Do not propose ARC offline BFS, per-game adapters, duplicate solves, or
  outer-loop reverse engineering as a headline solve.
- Do not treat token likelihood, hidden-state, attention, or early-exit ideas as
  evidence unless a local backend emits authenticated receipts first.
