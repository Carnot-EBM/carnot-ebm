# Research Roadmap vNEXT - Milestone 2026.07.497

**Milestone title:** Rewrite-State Verification, KAN-Assured CSL Scale-Up, Helper Lemmas, Hardware Receipts, and ARC Salience Rotation

**Planner date:** 2026-07-09
**Previous milestone:** 2026.07.496
**Task range:** Exp 5468-5481
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
`### V497 Planner Refresh - 20260709`.

New planning consequences:

- **inRAN** (`arXiv:2601.03219`) turns online adaptation into an interpretable
  KAN-surrogate plus safe-solver and chance-assurance loop. V497 uses this to
  push continuous self-learning from a governed bandit into a KAN-audited
  action/memory policy with rollback and threshold-offset receipts.
- **LemmaNet** (`arXiv:2603.22114`) and **Compile to Compress**
  (`arXiv:2604.18587`) motivate helper-lemma or helper-contract discovery from
  source/spec/context plus verifier failure signatures. V497 applies this only
  to deterministic AST/KB/doc-code witness rows and exact rechecks.
- **Ultrafast on-chip KAN online learning** (`arXiv:2602.02056`) suggests that
  sparse local coefficient updates are a plausible long-term hardware form for
  CSL policies. V497 keeps this as simulated fixed-point compatibility, not a
  board speedup claim.
- **SEM-CTRL** (`arXiv:2503.01804`) and **Theoria** (`arXiv:2607.01223`) are
  promoted into deterministic fixture design for semantic constraints and
  rewrite-state change accounting. This is the precondition before any future
  guided-decoding headline.
- **HALT/HUB** (`arXiv:2602.02888`) remains process telemetry when GGUF logprobs
  are available; it is not final authority and does not reopen retired external
  text-scorer lanes.
- **FPGA MPPI** (`arXiv:2601.17231`) and **FPGA Ising decomposition**
  (`arXiv:2602.15985`) sharpen receipt discipline for hardware work: matched
  workload hashes, repeated timing, board identity, and no speedup language
  without authenticated local data.

Secondary-source status:

- OpenReview reinforced semantic constrained decoding and neural-CSP framing,
  but no item superseded exact validators.
- HuggingFace Papers surfaced VeriFY, HALT/HUB, and self-verification pages;
  these inform telemetry and CSL baselines, not post-training mandates.
- Semantic Scholar public routes for EBT and ARM-EBM surfaced adjacent papers
  but no stronger local experiment than the already-tracked EBT/ARM-EBM hooks.
- GitHub EBM/KAN/ML4CO/hallucination repositories remain watch references.
- Extropic TSU/XTR-0 and Logical Intelligence Aleph/Kona pages remain
  architecture context only; Carnot has no local authenticated TSU, Kona, or
  Aleph execution path.

## What 2026.07.496 Proved

The `.496` capstone, changelog, and conductor log are the immediate planning
source of truth:

| Lane | Experiments | Finding |
|------|-------------|---------|
| Transition and source freshness | 5454, 5455 | `.496` activated cleanly and execution-time source deltas were reconciled. |
| Guided decoding | 5456, 5457 | The tautology corrigendum was clean, but the live local SOTA distortion-guarded panel was adversarially flagged: `lcd_bias_check_failed`, semantic false accepts, and factual distortion blocked readiness. |
| Deterministic repair and guards | 5458, 5459 | Minimal-core repair and deterministic constraint-distortion guards are clean and usable as exact-authority fixtures. |
| Continuous self-learning | 5460, 5461 | Governed frozen-model CSL policy and SOTA GGUF memory routing are clean: negative transfer was deflected without weight mutation. |
| Active constraints and hardware | 5462, 5463 | p-bit/p-dit boundary exchange and matched receipt collection are bounded; PolarFire returned matching hashes but slower timing, KV260 was unreachable, and no speedup is claimed. |
| ARC | 5464, 5465 | ARC metric-integrity precheck was clean; the live bp35 L3 salience attempt was an honest null/no-bank with live-agent self-discovery provenance. |
| Synthesis | 5466, 5467 | `.496` closed four headline lanes, kept three bounded, quarantined guided decoding, and recorded ARC/hardware speedup as honest nulls. |

## Three Biggest Gaps

1. **Generation-time verification still lacks a clean preflight.** `.496`
   proved deterministic guards but showed that live guided decoding can still
   pass local-looking constraints while failing LCD-bias and factual-distortion
   checks. V497 must build a small rewrite-state and semantic-constraint fixture
   that is independent of the guided reward scalar before another guided panel.

2. **Continuous self-learning is safe but not yet interpretable or scaled.**
   `.496` showed governed CSL can preserve quality on SOTA GGUF rows. The PRD
   asks for autonomous directed self-learning, so V497 adds a KAN-style
   surrogate, chance-style assurance, replay evidence, and larger SOTA memory
   routing while keeping all model weights frozen.

3. **Solver, hardware, and ARC grounding remain bounded.** Minimal cores and
   p-bit/p-dit bridges exist, but helper-lemma repair, neural-LNS style boundary
   accounting, and board timing are not yet composed. ARC remains at 69
   reproducible levels after a no-bank on bp35 L3; V497 rotates to a non-duplicate
   live target and keeps solve provenance strict.

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
        | Rewrite-state and semantic-constraint preflight                       |
        | typed state deltas, licensed changes, hidden-premise checks,          |
        | answer-set-like semantic guards, LCD-bias/distortion fixtures         |
        +-------------------+--------------------------+------------------------+
                            |                          |
           governed experience stream                  | exact witness stream
                            |                          |
        +-------------------v----------------+   +-----v------------------------+
        | Continuous self-learning policy    |   | Helper-lemma/core repair     |
        | frozen SOTA GGUF action/memory     |   | verifier failure signatures, |
        | routing, KAN surrogate, chance     |   | AST/KB/doc-code witnesses,   |
        | assurance, replay, rollback        |   | exact solver/test rechecks   |
        +-------------------+----------------+   +-----+------------------------+
                            |                          |
                            |                 solver hint / boundary stream
                            |                          |
                            |           +--------------v-----------------------+
                            |           | Active constraints and hardware      |
                            |           | p-bit/p-dit variables, LNS destroy/  |
                            |           | repair telemetry, exact fallback,    |
                            |           | matched CPU/board timing receipts    |
                            |           +--------------+-----------------------+
                            |                          |
        +-------------------v--------------------------v------------------------+
        | Live evidence, ARC, and milestone synthesis                           |
        | live-agent ARC salience rotation, registry precheck, no duplicate      |
        | solve, PRD gap table, capstone truth table, ops-doc alignment          |
        +-----------------------------------------------------------------------+
```

## Phase Plan

### Phase 0 - Transition and Source Delta

- **Exp5468:** archive `.496` terminal evidence and stage `.497` task range.
- **Exp5469:** run execution-time source delta against the V497 planner refresh
  and append only non-duplicate actionable references.

### Phase 1 - Rewrite-State Verification and SOTA Telemetry

- **Exp5470:** build deterministic Theoria/SEM-CTRL-style rewrite-state and
  semantic-constraint fixtures that specifically cover the Exp5457 LCD-bias and
  factual-distortion failure modes.
- **Exp5471:** if Exp5470 is clean, scale deterministic guard composition:
  minimal-core repair, semantic graph receipts, distortion guards, and exact
  final authority.
- **Exp5472:** if Exp5471 is clean, run a bounded local SOTA GGUF telemetry
  micro-panel. This is not a guided-decoding rerun; it measures model receipts,
  exact outcomes, optional logprob telemetry, and abstention behavior under the
  clean fixture.

### Phase 2 - KAN-Assured Continuous Self-Learning

- **Exp5473:** required CSL task: add an interpretable KAN-style surrogate and
  chance-style assurance receipts to the frozen action/memory policy.
- **Exp5474:** if Exp5473 is ready, run a larger local SOTA GGUF CSL
  memory/action routing panel using the mandated GGUF models and no weight
  mutation.
- **Exp5475:** build the behavioral memory evidence ladder for CSL replay:
  support removal, paraphrase robustness, locality, conflict handling,
  downstream action use, and matched no-memory/ICL baselines.

### Phase 3 - Helper Lemmas, Boundary Exchange, Hardware, ARC, and Synthesis

- **Exp5476:** extend minimal-core repair into LemmaNet-style helper lemmas or
  helper contracts for deterministic AST/KB/doc-code witness rows.
- **Exp5477:** apply ConsFormer-LNS-style accounting to p-bit/p-dit boundary
  exchange: destroy strategy, repair mode, exact fallback, and solver-authority
  outcomes.
- **Exp5478:** if Exp5477 is ready, collect CPU and reachable-board receipts
  for identical workload hashes; no speedup claim.
- **Exp5479:** run an ARC target-rotation and perception/salience precheck that
  avoids duplicate solved levels and recent no-bank targets.
- **Exp5480:** if Exp5479 is clean, run one live ARC salience level-up attempt
  with `solve_provenance=live_agent_self_discovery`.
- **Exp5481:** emit the `.497` capstone with PRD gap table, failure taxonomy,
  headline/bounded/blocked truth table, and ops-doc alignment recommendations.

## Natural Next-Experiment Chain

```text
Exp5457 LCD-bias/factual-distortion failure
  -> Exp5470 rewrite-state + semantic-constraint fixture
      -> Exp5471 deterministic guard-composition scale-up
          -> Exp5472 gated local SOTA evidence telemetry panel

Exp5460 governed CSL policy + Exp5461 clean SOTA CSL routing
  -> Exp5473 KAN-surrogate chance-assured CSL policy
      -> Exp5474 gated local SOTA CSL scale-up
          -> Exp5475 behavioral memory evidence ladder

Exp5458 minimal-core repair + Exp5459 distortion guard
  -> Exp5476 helper-lemma/core witness repair
      -> Exp5481 capstone PRD gap table

Exp5462 p-bit/p-dit bridge + ConsFormer-LNS reference
  -> Exp5477 boundary-exchange LNS accounting
      -> Exp5478 matched hardware receipts

Exp5464 ARC precheck + Exp5465 bp35 L3 honest null
  -> Exp5479 ARC target-rotation precheck
      -> Exp5480 live salience level-up attempt

Exp5468-Exp5480
  -> Exp5481 capstone and next-roadmap recommendations
```

## Hardware Requirements

| Experiment | Compute | Hardware notes | Claim boundary |
|-----------|---------|----------------|----------------|
| 5468, 5469, 5470, 5471, 5473, 5475, 5476, 5477, 5479, 5481 | CPU | Deterministic fixtures, repository artifacts, exact solvers, YAML/docs aggregation | No live model or hardware speedup claim |
| 5472, 5474 | Dual RTX 3090 preferred | Must use CUDA-enabled `llama-cpp-python` or native llama.cpp GGUF runtime; `MODEL_SPECS` include the three mandated SOTA GGUFs | Block headline result if offload/runtime receipts are absent; legacy small models are smoke-tests only |
| 5478 | CPU plus PolarFire/KV260 if reachable | Same workload hash, result hash, repeat counts, timing distribution; KV260 SSH-only; GateMate diagnostic-only unless physical/JTAG evidence returns | `hardware_speedup_claim=false` unless future matched board-local timing justifies otherwise |
| 5479, 5480 | CPU ARC runtime | Registry precheck, live-agent self-discovery only, no source reading, no offline BFS, no per-game adapter credited path | Registry count changes only after reproduction-gated new level |

## Risk Controls

- **GGUF runtime:** Exp5472 and Exp5474 must list all three mandated SOTA GGUFs
  in `MODEL_SPECS`: `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`. They must load via GGUF/llama.cpp paths,
  not `AutoTokenizer.from_pretrained` on GGUF repositories. Legacy small models
  are smoke-tests only and cannot be headline results.
- **Guided-decoding quarantine:** V497 does not rerun live guided decoding as a
  headline. Exp5472 is a telemetry panel over exact fixtures, not token-level
  verifier-potential steering.
- **Verifier authority:** Semantic constraints, HALT-style telemetry, KAN policy
  scores, and p-bit/p-dit hints may guide or diagnose. Exact deterministic
  verifiers, AST/KB witnesses, solvers, and reproduction gates remain final
  authority.
- **CSL safety:** Continuous self-learning cannot mutate model or adapter
  weights. Policy updates must carry raw traces, provenance, confidence/assurance
  receipts, replay evidence, rollback pointers, and negative-transfer controls.
- **Hardware honesty:** Hardware tasks may report slower board execution.
  Speedup remains unclaimed without repeated matched board-local timing.
- **ARC discipline:** Only the live agent's own attempts and runtime reverse
  engineering count. Offline BFS, hidden source reading, outer-loop RE, and
  hand per-game adapters are not headline solve paths.
- **Closed lanes stay closed:** External generated-text scorer lanes, broad
  fine-tuning/RL reruns, non-local TSU/Kona/Aleph execution, retired ARC
  first-contact exploration-signal reruns, and CPU-only SOTA offload reruns
  remain closed without authenticated new receipts or operator override.
