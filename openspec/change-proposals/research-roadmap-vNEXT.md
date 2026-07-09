# Research Roadmap vNEXT - Milestone 2026.07.496

**Milestone title:** Tautology-Clean Guided Decoding, Distortion Guards, Governed CSL Policy, and ARC Perception Integrity

**Planner date:** 2026-07-09
**Previous milestone:** 2026.07.495
**Task range:** Exp 5454-5467
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
Intelligence public pages before designing experiments. Actionable deltas were
appended to `research-references.md` under
`### V496 Planner Refresh - 20260709`.

New planning consequences:

- **Chance-Constrained Inference** (`arXiv:2602.01637`) turns hallucination
  mitigation into deployment-time risk control. V496 guided decoding must
  report accepted-generation risk, abstention, finite-sample confidence, and
  infeasible-input detection.
- **CoCoA conflict-aware adaptive decoding** (`arXiv:2508.17670`) and
  **strict-constraint distortion** (`arXiv:2601.01490`) make factual
  distortion a first-class failure mode. V496 should not credit constraint
  satisfaction when the model silently rewrites known facts to satisfy the
  constraint.
- **DAVinCI dual attribution and verification** (`arXiv:2604.21193`) motivates
  row-level claim attribution, evidence-span IDs, and calibrated entailment or
  contradiction receipts for SOTA GGUF factuality rows.
- **OLIVIA** (`arXiv:2605.11169`) and **CL-Bench** (`arXiv:2606.05661`) push
  CSL from static memory presence to frozen-model online decision policies
  tested against no-memory and naive in-context baselines.
- **LCAD** (OpenReview `rbl8fHjLuF`) motivates monotone verifier-potential
  accounting, but V496 keeps exact final verifiers as authority and rejects
  metrics that prove success using the same scalar they are evaluating.

Already-indexed sources remain active constraints on the design: tractable
locally constrained decoding (`arXiv:2606.01926` / OpenReview `LYBs6f3jlK`),
KAN PWA/MILP verification (`arXiv:2602.06737`), STATIC trie decoding
(`arXiv:2602.22647`), million-p-bit FPGA sampling (`arXiv:2606.25313`),
governed evolving memory (`arXiv:2603.11768`), and minimal-core-guided repair.

Secondary-source status:

- OpenReview reinforced local-constrained-decoding bias, Lyapunov-style
  constraint control, verifier-guided sampling, and certified neural
  constraint solving.
- HuggingFace Papers surfaced DAVinCI, verification-guided reasoning, and
  constrained-decoding pages, but none displaced local GGUF runtime
  requirements.
- Semantic Scholar public routes for EBT and ARM-EBM did not reveal a stronger
  citation-derived task than the source-paper hooks already tracked.
- GitHub constrained-decoding, CL-Bench, KAN, p-bit, and Ising repositories
  remain implementation watch references.
- Extropic TSU/XTR-0 and Logical Intelligence Aleph/Kona posts remain
  architecture context only; Carnot has no local authenticated TSU, Kona, or
  Aleph execution path.

## What 2026.07.495 Proved

The `.495` capstone and changelog are the immediate planning source of truth:

| Lane | Experiments | Finding |
|------|-------------|---------|
| Transition and source freshness | 5441, 5442 | `.495` activated cleanly and source deltas were reconciled. |
| Verifier-potential fixtures | 5443 | Deterministic prefix-potential fixtures with exact final authority are usable. |
| Local SOTA guided decoding | 5444 | The live SOTA verifier-potential pilot ran but was adversarially flagged for `TAUTOLOGY`; it is not a headline result. |
| Deterministic witnesses | 5445 | AST/KB witness constraints for code/API-like claims are complete and usable as exact evidence. |
| Continuous self-learning | 5446, 5447 | Governed memory and memory-failure stress are complete, with no weight mutation and negative-transfer controls. |
| Active constraints and hardware | 5448, 5449 | P-bit sparsity bridge and hardware timing receipts are complete; no hardware speedup is claimed. |
| ARC | 5450 | Measurement-access predicate induction produced an honest null/no-bank on `ka59` L2; registry count remains 69. |
| Certificates and synthesis | 5451-5453 | Bounded KAN certificates, PRD gap synthesis, and capstone truth table are complete; token/internal lanes remain closed without backend receipts. |

## Three Biggest Gaps

1. **Generation-time verification is blocked by metric validity.** `.495`
   proved deterministic verifier-potential fixtures, but the first local SOTA
   pilot was flagged as tautological. V496 must repair the accounting before
   any new guided-decoding headline.

2. **Constraint satisfaction can hide factual distortion and biased sampling.**
   The PRD wants trustworthy reasoning, not just schema-conformant output.
   V496 adds conflict-aware decoding controls, chance-constrained acceptance,
   attribution receipts, and locally constrained decoding bias checks.

3. **Learning and physical grounding are still bounded.** CSL has safe memory
   units but not an online policy evaluated against realistic baselines and
   SOTA rows. ARC remains at no-bank, and hardware remains receipt-only without
   speedup. V496 advances policy, perception integrity, and p-bit timing-ratio
   evidence while keeping exact authorities in charge.

## Target Architecture

```text
                         +--------------------------------------+
                         | Local SOTA GGUF inference substrate  |
                         | Qwen3.6-35B-A3B, Gemma-4-31B-it,     |
                         | Gemma-4-26B-A4B-it via llama.cpp     |
                         +-------------------+------------------+
                                             |
                              prompts, candidates, row receipts
                                             |
        +------------------------------------v----------------------------------+
        | Tautology-clean verifier-potential layer                             |
        | independent row metrics, exact final verifiers, attribution spans,    |
        | chance-constrained acceptance, distortion and LCD-bias guards         |
        +-------------------+--------------------------+------------------------+
                            |                          |
           governed experience stream                  | solver hint stream
                            |                          |
        +-------------------v----------------+   +-----v------------------------+
        | Continuous self-learning policy    |   | Active constraints and p-bit |
        | frozen-model action/memory routing |   | assumptions with minimal     |
        | UCB/confidence, replay, rollback,  |   | cores, restored sparsity,    |
        | no-memory and ICL baselines        |   | exact fallback, timing ratios|
        +-------------------+----------------+   +-----+------------------------+
                            |                          |
                            +------------+-------------+
                                         |
        +--------------------------------v--------------------------------------+
        | Live evidence and bounded certificates                                |
        | ARC perception/metric integrity, live self-discovery level-up attempt, |
        | KAN measurement certificates, PRD gap table, capstone                 |
        +-----------------------------------------------------------------------+
```

## Phase Plan

### Phase 0 - Transition and Source Delta

- **Exp5454:** archive `.495` terminal evidence and stage `.496` task range.
- **Exp5455:** run execution-time source delta against the V496 planner
  refresh and append only non-duplicate actionable references.

### Phase 1 - Tautology-Clean Guided Decoding and Distortion Guards

- **Exp5456:** repair the Exp5444 tautology finding with row-independent
  metric accounting and an adversarially checkable dependency graph.
- **Exp5457:** if Exp5456 is clean, rerun a bounded local SOTA GGUF guided
  decoding panel with conflict, distortion, attribution, chance-risk, and
  locally constrained decoding bias controls.
- **Exp5458:** apply minimal-core-guided repair to deterministic
  verifier-potential and AST/KB witness rows.
- **Exp5459:** build a constraint-distortion guard that distinguishes honest
  constraint violation from false factual rewrites.

### Phase 2 - Governed Continuous Self-Learning Policy

- **Exp5460:** required CSL task: convert governed memory into a frozen-model
  online action/memory policy with no-memory and naive in-context baselines.
- **Exp5461:** if Exp5460 is ready, run a bounded local SOTA GGUF CSL memory
  routing panel and measure negative transfer, context cost, and verifier cost.

### Phase 3 - Solver, Hardware, ARC, and Synthesis

- **Exp5462:** extend the active-constraint p-bit bridge with minimal-core and
  p-dit assignment controls while exact solvers keep final authority.
- **Exp5463:** if Exp5462 is ready, collect CPU and reachable-board timing
  receipts for the same workload hashes; no speedup claim.
- **Exp5464:** run ARC metric-integrity and perception precheck for
  null-coordinate and salience failure modes.
- **Exp5465:** if Exp5464 is clean, run one live ARC connected-component
  salience level-up attempt with live-agent self-discovery provenance.
- **Exp5466:** aggregate `.496` evidence into a PRD gap and failure-taxonomy
  table.
- **Exp5467:** emit the `.496` capstone truth table and recommendations.

## Natural Next-Experiment Chain

```text
Exp5444 TAUTOLOGY finding
  -> Exp5456 guided-decoding metric corrigendum
      -> Exp5457 gated local SOTA distortion/bias guarded decoding rerun
          -> Exp5466 PRD gap table

Exp5443 verifier-potential fixtures + Exp5445 AST/KB witnesses
  -> Exp5458 minimal-core repair formalization
      -> Exp5459 constraint-distortion guard

Exp5446 governed CSL + Exp5447 memory-failure stress
  -> Exp5460 governed CSL policy bandit
      -> Exp5461 gated local SOTA memory-routing panel

Exp5448 p-bit sparsity bridge + V496 p-bit/p-dit literature
  -> Exp5462 minimal-core p-bit/p-dit bridge
      -> Exp5463 hardware timing-ratio receipts

Exp5450 ARC no-bank + known-issues perception lane
  -> Exp5464 ARC metric-integrity precheck
      -> Exp5465 ARC live connected-component salience level-up attempt

Exp5456-Exp5465
  -> Exp5466 PRD gap table
      -> Exp5467 capstone
```

## Hardware Requirements

| Experiment | Compute | Hardware notes | Claim boundary |
|-----------|---------|----------------|----------------|
| 5454, 5455, 5456, 5458, 5459, 5460, 5462, 5464, 5466, 5467 | CPU | Deterministic fixtures, repository artifacts, exact solvers | No live model or hardware speedup claim |
| 5457, 5461 | Dual RTX 3090 preferred | Must use CUDA-enabled `llama-cpp-python` or native llama.cpp GGUF runtime; `MODEL_SPECS` include the three mandated SOTA GGUFs | Block headline result if offload/runtime receipts are absent |
| 5463 | CPU plus PolarFire/KV260 if reachable | Same workload hash, result hash, repeat counts, timing distribution; KV260 SSH-only; GateMate diagnostic-only unless physical/JTAG evidence returns | `hardware_speedup_claim=false` unless future matched board-local timing justifies otherwise |
| 5464, 5465 | CPU ARC runtime | Registry precheck, live-agent self-discovery only, no source reading, no offline BFS, no per-game adapter credited path | Registry count changes only after reproduction-gated new level |

## Risk Controls

- **GGUF runtime:** Exp5457 and Exp5461 must use at least one of the three
  mandated local SOTA GGUFs and list all three in `MODEL_SPECS`:
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`. They must load via GGUF/llama.cpp paths,
  not `AutoTokenizer.from_pretrained` on GGUF repositories.
- **Metric independence:** Exp5456 must prove guided-decoding success metrics
  are computed from row outcomes and exact verifier labels, not from the
  guided reward scalar itself.
- **Verifier authority:** Verifier potentials and policy scores may guide
  search, but exact deterministic verifiers, AST/KB witnesses, and solvers
  remain final authority.
- **CSL safety:** Continuous self-learning cannot mutate model or adapter
  weights. Policy updates must carry raw traces, provenance, UCB/confidence
  receipts, replay evidence, rollback pointers, and negative-transfer controls.
- **Hardware honesty:** Hardware tasks may report slower board execution.
  Speedup remains unclaimed without repeated matched board-local timing.
- **ARC discipline:** Only the live agent's own attempts and runtime reverse
  engineering count. Offline BFS, hidden source reading, and hand per-game
  adapters are not headline solve paths.
- **Closed lanes stay closed:** Token/internal hidden-state/logprob claims,
  external generated-text scorer lanes, non-local TSU/Kona/Aleph execution,
  and retired first-contact exploration reruns remain closed without
  authenticated new receipts or operator override.
