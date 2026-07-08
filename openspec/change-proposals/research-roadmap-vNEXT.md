# Research Roadmap vNEXT - Milestone 2026.07.495

**Milestone title:** Verifier-Potential Generation, Governed Online Memory, and Solver-Authoritative Hardware Boundaries

**Planner date:** 2026-07-08
**Previous milestone:** 2026.07.494
**Task range:** Exp 5441-5453
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
- `scripts/roadmap_schema.py`
- `scripts/audit_roadmap_gates.py`
- `scripts/arc_levelup_guarantee_lint.py`
- `scripts/exclusion_manifest_lint.py`
- `ops/e2e-test-plan.md`

## Literature Refresh Incorporated

The planner performed a 2025-2026 refresh across arXiv, OpenReview,
Extropic writing, Semantic Scholar citation routes for EBT `2507.02092` and
ARM-EBM `2512.15605`, HuggingFace Papers, GitHub repositories, and Logical
Intelligence public pages before designing experiments. Actionable findings
were appended to `research-references.md` under
`### V495 Planner Refresh - 2026-07-08`.

Promoted sources and planning consequences:

- **Score x Decoder** (`arXiv:2606.00739`): motivates prefix/particle
  decoding as a sampler-guidance problem, but `.495` uses only deterministic
  verifier potentials with reward-budget accounting.
- **Stratified Scaling Search** (`arXiv:2604.06260`) and **DTV meta-step
  decoding** (`arXiv:2605.17626`): motivate verifier calls during generation,
  not just final-output filtering. `.495` tests structural checkpoints and
  rollback on bounded local SOTA GGUF rows.
- **p-bit guided CDCL** (`arXiv:2605.04033`) and **Potts mean-field
  constraints** (`arXiv:2602.04200`): motivate stochastic hints as temporary
  assumptions while exact solvers keep correctness authority.
- **SSGM memory governance** (`arXiv:2603.11768`), **execution provenance**
  (`arXiv:2606.04990`), **Execute-Distill-Verify**
  (`arXiv:2606.24428`), **MemFail** (`arXiv:2605.26667`), and the
  **Experience Compression Spectrum** (`arXiv:2604.15877`): motivate a
  continuous self-learning loop with raw traces, case memory, skill memory,
  rules, provenance, replay, rollback, and trap-memory stress.
- **Deterministic AST hallucination correction** (`arXiv:2601.19106`):
  motivates adding static AST/KB witness constraints to structured verifier
  rows before crediting any code/API claim.

Secondary-source status:

- OpenReview reinforced verifier-guided sampling, certified neural constraint
  solving, and KAN variants, but no page displaced Carnot's exact-solver
  authority.
- HuggingFace Papers reinforced EBT-style energy minimization and constrained
  decoding, but no source changed local GGUF runtime requirements.
- Semantic Scholar public routes for EBT and ARM-EBM did not reveal a stronger
  citation-derived task than the source-paper hooks already tracked.
- GitHub constrained-decoding, type-constrained generation, KAN, and
  p-bit/Ising repositories remain implementation watch references.
- Extropic TSU/XTR-0 and Logical Intelligence Aleph/Kona posts remain
  architecture context only; Carnot has no local authenticated TSU, Kona, or
  Aleph execution path.

## What 2026.07.494 Proved

The `.494` capstone is the immediate planning source of truth:

| Lane | Experiments | Finding |
|------|-------------|---------|
| Structured verification | 5430, 5431 | The structured tautology corrigendum and taxonomy replication are clean, row-derived, and headline-ready under local SOTA GGUF receipts. |
| Ontology constraint memory | 5432 | Ontology and soft-logic memory can be made deterministic with solver authority and trap rejection. |
| Active constraints | 5433 | Diverse LNS hints covered four families, reduced solver work by 138, and preserved solver authority. |
| Hardware timing | 5434 | CPU/PolarFire receipts were hash-matched and useful, but no speedup claim is justified. |
| Continuous self-learning | 5435, 5436 | Verified workflow memory promoted only validated case/skill sidecars, transferred in-domain, rejected shifted negative transfer, retained raw episodes, preserved rollback, and mutated no weights. |
| ARC | 5437 | The live ARC attempt was an honest null on `cn04` L4; registry count remains 69. |
| KAN certificates | 5438 | Bounded certificates rejected false ontology/retrieval/unsupported claims without making broad KAN claims. |
| Synthesis | 5439, 5440 | Closed/partial/blocked lanes were reconciled; token/internal lanes remain closed without backend receipts. |

## Three Biggest Gaps

1. **Verified reasoning is still mostly post-hoc.** `.494` made structured
   verification clean, but the PRD vision needs constraints to shape
   generation before the model commits to invalid prefixes. `.495` introduces
   verifier-potential fixtures and a gated local SOTA GGUF decoding pilot.

2. **Continuous self-learning has safe memory units but not a governed online
   lifecycle.** `.494` proved verified workflow memory and transfer stress.
   The next gap is multi-session promotion across raw traces, case memory,
   skills, and rules with provenance, replay, decay, rollback, and
   memory-failure attribution.

3. **Hardware and ARC remain north-star weak.** Active constraints and
   p-bit receipts are bounded but not yet integrated with restored-sparsity
   ledgers or repeatable board timing. ARC still has repeated no-bank attempts.
   `.495` keeps hardware honest and includes one live-path, registry-checked
   ARC level-up attempt using a mechanism distinct from retired first-contact
   exploration reruns.

## Target Architecture

```text
                         +--------------------------------------+
                         | Local SOTA GGUF inference substrate  |
                         | Qwen3.6-35B-A3B, Gemma-4-31B-it,     |
                         | Gemma-4-26B-A4B-it via llama.cpp     |
                         +-------------------+------------------+
                                             |
                     partial candidates + structural checkpoints
                                             |
        +------------------------------------v----------------------------------+
        | Verifier-potential generation layer                                  |
        | prefix rewards, SMC/rollback accounting, AST/KB witnesses,           |
        | exact schema/semantic/tool-state solvers as final authority           |
        +-------------------+--------------------------+------------------------+
                            |                          |
            governed experience stream                 | solver hint stream
                            |                          |
        +-------------------v----------------+   +-----v------------------------+
        | Continuous self-learning memory    |   | Active constraints and p-bit |
        | raw traces -> cases -> skills ->   |   | assumptions, restored        |
        | rules, provenance, replay, decay,  |   | sparsity, exact fallback,    |
        | rollback, trap-memory stress       |   | CPU/board timing receipts    |
        +-------------------+----------------+   +-----+------------------------+
                            |                          |
                            +------------+-------------+
                                         |
        +--------------------------------v--------------------------------------+
        | Live evidence and bounded certificates                                |
        | ARC live self-discovery level-up attempt, KAN measurement certificates,|
        | PRD gap table, capstone; token/internal lanes stay closed             |
        +-----------------------------------------------------------------------+
```

## Phase Plan

### Phase 0 - Transition and Source Delta

- **Exp5441:** archive `.494` terminal evidence and stage `.495` task range.
- **Exp5442:** run execution-time source delta against the V495 planner
  refresh and append only non-duplicate actionable references.

### Phase 1 - Verifier-Potential Generation and Deterministic Witnesses

- **Exp5443:** build a deterministic verifier-potential fixture with prefix
  reward accounting, exact final checks, and row checksums.
- **Exp5444:** if Exp5443 is ready, run a small local SOTA GGUF energy-guided
  constrained-decoding pilot against unconstrained and grammar-only baselines.
- **Exp5445:** add deterministic AST/KB witness constraints for code/API-like
  hallucination rows and wire them into structured verifier evidence.

### Phase 2 - Governed Continuous Self-Learning and Solver Hints

- **Exp5446:** required CSL task: run governed multi-session workflow memory
  across trace, case, skill, and rule promotion levels with replay and rollback.
- **Exp5447:** if Exp5446 is ready, stress memory failure modes, stale
  summaries, retrieval collisions, and negative transfer.
- **Exp5448:** extend active-constraint hints into p-bit/CDCL-style temporary
  assumptions with restored-sparsity and solver-authority ledgers.
- **Exp5449:** if Exp5448 is ready, collect CPU/PolarFire/KV260 timing and
  variance receipts for the exact same workload hashes, with no speedup claim.

### Phase 3 - ARC, Certificates, and Synthesis

- **Exp5450:** run one ARC live-path level-up attempt with registry precheck,
  measurement-access predicate induction, and live-agent self-discovery only.
- **Exp5451:** issue bounded KAN measurement-access certificates for
  verifier-potential and governed-memory claims.
- **Exp5452:** aggregate `.495` evidence into a PRD gap and failure-taxonomy
  table.
- **Exp5453:** emit the `.495` capstone truth table and recommendations.

## Natural Next-Experiment Chain

```text
Exp5430/5431 clean structured verification
  -> Exp5443 verifier-potential fixture
      -> Exp5444 local SOTA energy-guided decoding pilot
      -> Exp5451 bounded KAN verifier-potential certificates

V495 source refresh + deterministic code-hallucination literature
  -> Exp5445 AST/KB witness constraints
      -> Exp5452 PRD gap table

Exp5435/5436 verified workflow CSL
  -> Exp5446 governed online workflow memory
      -> Exp5447 memory-failure stress
      -> Exp5451 bounded KAN governed-memory certificates

Exp5433 active constraints + V495 p-bit/CDCL literature
  -> Exp5448 p-bit assumption bridge
      -> Exp5449 hardware timing receipts

Exp5437 ARC no-bank + arc_solve_registry + known-issues ARC floor
  -> Exp5450 live ARC level-up attempt

Exp5443-Exp5451
  -> Exp5452 PRD gap table
      -> Exp5453 capstone
```

## Hardware Requirements

| Experiment | Compute | Hardware notes | Claim boundary |
|-----------|---------|----------------|----------------|
| 5441, 5442, 5443, 5445, 5446, 5447, 5448, 5451-5453 | CPU | Deterministic fixtures, repository artifacts, exact solvers | No live model or hardware speedup claim |
| 5444 | Dual RTX 3090 preferred | Must use CUDA-enabled `llama-cpp-python`/GGUF runtime; `MODEL_SPECS` include the three mandated SOTA GGUFs | Block headline result if offload/runtime receipts are absent |
| 5449 | CPU plus PolarFire/KV260 if reachable | Same workload hash, result hash, repeat counts, timing distribution; KV260 SSH-only; GateMate diagnostic-only unless physical/JTAG evidence returns | `hardware_speedup_claim=false` unless future matched board-local timing justifies otherwise |
| 5450 | CPU ARC offline/live runtime | Registry precheck, live-agent self-discovery only, no source reading, no offline BFS, no per-game adapter credited path | Registry count changes only after reproduction-gated new level |

## Risk Controls

- **GGUF runtime:** Exp5444 must use the three mandated local SOTA GGUFs:
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`. It must load via GGUF/llama.cpp paths,
  not `AutoTokenizer.from_pretrained` on GGUF repositories.
- **Verifier authority:** Verifier potentials may guide generation, but exact
  deterministic verifiers remain final authority. No learned or LLM self-score
  is accepted as a certificate.
- **CSL safety:** Memory promotion cannot mutate model weights. Promoted
  memory must retain raw traces, provenance, replay evidence, rollback pointers,
  temporal decay policy, and negative-transfer controls.
- **Hardware honesty:** Hardware tasks may report slower board execution.
  Speedup remains unclaimed without repeated matched board-local timing.
- **ARC discipline:** Only the live agent's own attempts and runtime reverse
  engineering count. Offline BFS, hidden source reading, and hand per-game
  adapters are not headline solve paths.
- **Closed lanes stay closed:** Token/internal hidden-state/logprob claims,
  external generated-text scorer lanes, non-local TSU/Kona/Aleph execution, and
  retired first-contact exploration reruns remain closed without authenticated
  new receipts or operator override.
