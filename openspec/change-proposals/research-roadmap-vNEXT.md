# Research Roadmap vNEXT - Milestone 2026.07.491

**Milestone title:** Verifier-Budget Routing, Solver-Guidance Corrigendum, and Live Salience Grounding

**Planner date:** 2026-07-08
**Previous milestone:** 2026.07.490
**Task range:** Exp 5389-5401
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
- `results/experiment_5388_capstone_v490.json`

## Literature Refresh Incorporated

The planner performed a 2025-2026 source refresh and appended the actionable findings to
`research-references.md` under `### V491 Planner Refresh - 2026-07-08` before designing
experiments.

Promoted sources and planning consequences:

- **Sortify / Influence Exchange** (`arXiv:2603.27765`): motivates moving Exp5382's successful
  real-workflow self-learning from fixed memory governance to an influence-accounted verifier
  router with explicit cost, quality, rollback, and no-weight-mutation evidence.
- **Energy-Aware Routing to Large Reasoning Models** (`arXiv:2601.00823`): motivates treating
  local SOTA GGUF calls, cheap deterministic checks, and grammar preflights as a routed resource
  pool. `.491` should report verifier-cost reductions, not unsupported model-energy claims.
- **Mathematical Encoding Safety Gaps** (`arXiv:2605.03441`): motivates a formal-encoding fixture
  where local SOTA models classify structured safety/intent constraints and deterministic checks
  retain final authority.
- **Agentic Model Checking** (`arXiv:2605.21434`) and **VAGEN** (`arXiv:2602.00575`): reinforce the
  Carnot division of labor: agents may propose specs or probing actions, while deterministic
  verifiers and live environment evidence discharge soundness-relevant claims.
- **Ising MPPI** (`arXiv:2512.15533`): motivates p-bit/Ising diagnostics over action-sequence
  boundary exchange and overwrite/fallback behavior, still simulation-only until repeatable board
  timing exists.

Sources already present in `research-references.md` but used as planning inputs:

- **KANDy** (`arXiv:2602.20413`): supports a compact KAN dynamics/counterexample certificate.
- **Graph-Based Exploration for ARC-AGI-3** (`arXiv:2512.24156`) and
  `dolphin-in-a-coma/arc-agi-3-just-explore`: support the required ARC connected-component /
  color-blob salience task already called out in `ops/known-issues.md`.

Secondary-source status:

- Semantic Scholar routes for EBT `2507.02092` and ARM-EBM `2512.15605` did not produce a fresh
  reproducible local change; no citation-count claim is made.
- HuggingFace Papers reinforced constraint-satisfaction and interactive-verification fixtures, but
  older KITAB entries are benchmark context only.
- Extropic TSU/X0/XTR and Logical Intelligence Kona/Aleph pages remain non-local architecture
  context. They are not executable baselines for `.491`.

## What 2026.07.490 Proved

The `.490` capstone closed with stronger evidence than `.489` but left three major blockers:

- `structured_methodology_receipt_ready=true` and `structured_protocol_clean=true`: the mandated
  local SOTA GGUF structured-output protocol produced a valid live receipt after the `.489`
  duration failure.
- `constraint_tax_panel_ready=true`: the small constraint-tax panel showed that deterministic
  structured constraints can expose wrong-valid and unreachable tool/action outputs that
  unconstrained generation misses.
- `budget_memory_corrigendum_clean=true`: the earlier memory-tautology issue was corrected from
  row-level evidence.
- `continuous_self_learning_real_workflow_ready=true` and
  `continuous_self_learning_requirement_satisfied=true`: Exp5382 ran a real dependency/drift
  verifier workflow over 12 sessions and 36 checks with better context efficiency, lower verifier
  cost, preserved quality, rollback, and no model-weight mutation.
- `overwrite_guidance_scale_ready=true` in the artifact, but Exp5383 was adversarially flagged
  `CRITICAL TAUTOLOGY` by the conductor. Solver-guidance scale-up must be re-emitted or repaired
  before it can become headline evidence.
- `pbit_boundary_overwrite_ready=true`: the p-bit boundary/overwrite joint diagnostic exists, but
  it is still CPU simulation and downstream of the flagged solver-guidance artifact.
- `arc_new_level_banked=false`: Exp5385 honestly attempted re86 L3 through live-agent
  self-discovery and did not bank a new level.
- `hardware_hash_chained_receipt_ready=true` and `hardware_speedup_claim=false`: PolarFire produced
  a reachable workload receipt, KV260 was unreachable, GateMate remained physically/JTAG blocked,
  and no repeatability evidence supported speedup.
- `future_token_signal_allowed=false`: token/internal-feature energy remains closed until a backend
  exposes logits, hidden states, attention, or depth exits with clean provenance.

## Three Biggest Gaps

1. **Verifier value is still too small-scale.** The PRD needs convincing local, verifiable reasoning
   evidence. `.490` proved the shape on a small panel; `.491` must expand fixture count and add
   adversarial formal-encoding constraints without relying on external text scorers.

2. **Continuous self-learning is positive but not yet a controller.** FR-11 requires adaptive,
   durable self-learning. `.490` showed a real workflow can save context and verifier cost; `.491`
   should make routing decisions explicit with influence shares, quality-preserving gates, raw
   evidence retention, poisoning controls, and rollback.

3. **Live-reachable evidence remains the frontier.** Solver guidance is flagged, ARC did not level
   up, hardware lacks repeatable board timing, and token/internal energy is closed. `.491` should
   repair flagged solver evidence, improve live ARC salience, and keep hardware claims strictly
   hash-linked and no-speedup until repeatability exists.

## Target Architecture

```text
             +------------------------------------------------+
             | Local SOTA GGUF Inference Substrate             |
             | Qwen3.6-35B-A3B / Gemma 31B / Gemma 26B-A4B    |
             | llama.cpp or llama-cpp-python with CUDA offload |
             +------------------------+-----------------------+
                                      |
                         structured outputs + grammar budget
                                      |
             +------------------------v-----------------------+
             | Expanded Constraint Verifier Panel              |
             | schema, semantic, tool-state, safety encoding   |
             | deterministic verifier remains final authority  |
             +------------+---------------------+-------------+
                          |                     |
              repaired row-level proof          | routed verifier budget
                          |                     |
       +------------------v---------+   +-------v----------------------+
       | Solver / p-bit Guidance     |   | Continuous Self-Learning      |
       | hints are optional, solver  |   | influence shares, raw traces, |
       | can overwrite/fallback      |   | rollback, no weight mutation  |
       +------------------+---------+   +-------+----------------------+
                          |                     |
                          |                     |
       +------------------v---------------------v----------------------+
       | Live Evidence Surfaces                                      |
       | ARC blob salience in live agent path; hardware hash graph; |
       | token/internal features remain closed without backend proof |
       +------------------------------------------------------------+
```

## Phase Plan

### Phase 0 - Source Delta and Expanded Local Fixtures

- **Exp5389:** archive `.490` outcomes and stage `.491` execution context.
- **Exp5390:** run the execution-time SOTA delta and add new references without reopening retired
  lanes.
- **Exp5391:** scale the clean structured/constraint-tax panel with the mandated local SOTA GGUF
  models.
- **Exp5392:** add a formal-encoding safety/intent constraint fixture, again with local SOTA GGUF
  evidence and deterministic final checks.

### Phase 1 - Solver Corrigendum and Gated P-bit Ablation

- **Exp5393:** repair Exp5383 by recomputing overwrite-guidance claims from row-level evidence and
  making the tautology risk explicit.
- **Exp5394:** only if Exp5393 is clean, run the joint overwrite/p-bit ablation over action-sequence
  boundary exchange. No hardware speedup claim is allowed.

### Phase 2 - Continuous Self-Learning and Live ARC Salience

- **Exp5395:** implement an influence-share verifier-budget router for the continuous self-learning
  workflow, preserving quality and rollback.
- **Exp5396:** harden the self-learning memory guard with raw episode retention, forged-reasoning
  controls, stale-memory controls, and no weight mutation.
- **Exp5397:** implement connected-component/color-blob salience and salience-tiered action
  prioritization inside the live ARC agent path, then attempt the next reachable live level after a
  registry precheck.

### Phase 3 - Evidence Surfaces, Certificates, and Capstone

- **Exp5398:** extend hardware receipts into hash-linked repeatability evidence across available
  boards, while keeping `hardware_speedup_claim=false` unless repeated board timing exists.
- **Exp5399:** produce a compact KAN/KANDy-style dynamics counterexample certificate for constraint
  drift or verifier routing.
- **Exp5400:** aggregate the clean local SOTA, safety, self-learning, solver, ARC, KAN, and hardware
  evidence into a PRD-aligned evidence table.
- **Exp5401:** close `.491` with a capstone truth table and next-roadmap recommendations.

## Dependency Graph

```text
exp5389 transition
  -> exp5390 source delta
  -> exp5391 scaled constraint-tax fixtures
  -> exp5392 formal-encoding fixture

exp5393 solver-guidance tautology corrigendum
  -> exp5394 gated overwrite/p-bit ablation

exp5382 prior clean workflow
  -> exp5395 influence-share verifier-budget router
  -> exp5396 raw-episode memory guard

ops/known-issues.md + arc_solve_registry
  -> exp5397 live ARC blob salience level-up attempt

exp5386 prior hardware receipts
  -> exp5398 hardware repeatability evidence graph

KANDy reference + verifier artifacts
  -> exp5399 KAN dynamics certificate

exp5391/5392/5393/5394/5395/5396/5397/5398/5399
  -> exp5400 evidence table
  -> exp5401 capstone
```

Structured conductor gate:

- `exp5394-v491-gated-overwrite-pbit-ablation` is gated on
  `exp5393-v491-overwrite-guidance-tautology-corrigendum.overwrite_guidance_corrigendum_clean == true`.

## Model and Inference Requirements

Every `.491` experiment that calls an LLM must use at least one mandated local SOTA GGUF model in
`MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Implementation prompts should prefer `scripts/experiment_template.py` and the `cached_sota_pair()`
pattern. Legacy small models such as Qwen3.5-0.8B or gemma-4-E4B-it are allowed only as CPU smoke
tests and must not be headline-result models.

GGUF/CUDA preconditions for LLM tasks:

- Confirm model cache entries before running.
- Confirm `llama_cpp` / `llama-cpp-python` GPU support or `llama-cli -st` CUDA offload support.
- Record offload layers, command line, wall time, prompt count, parse/schema/semantic rates, and
  unsafe false accepts.
- If CUDA/offload is unavailable, emit an honest blocked artifact instead of falling back to a
  CPU-only headline result.

Per `CLAUDE.md`, staged experiment tasks default to `agent_type: codex` and `model: gpt-5.5` unless
a future operator instruction explicitly re-enables another backend.

## Hardware Requirements

Available hardware and constraints from `research-hardware-wishlist.md`:

- Dual RTX 3090 CUDA system: primary local SOTA GGUF substrate.
- PolarFire: reachable in `.490`; safe workload receipts are allowed.
- KV260: must use SSH/board-local commands only. Never inspect or write host `/dev/mmcblk*` as KV260
  evidence.
- GateMate: DirtyJTAG USB visibility alone is not a workload receipt; physical/JTAG availability
  must be reported honestly.
- No TSU/XTR/Kona/Aleph hardware path exists locally.
- `hardware_speedup_claim` must remain false unless repeated board-local timing and repeatability
  evidence are captured.

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not reopen retired external generated-text scorer lanes.
- Do not claim token/internal-feature energy from final text or incomplete token-probability rows.
- Do not use CPU-only legacy GGUF smoke tests as headline SOTA evidence.
- Do not headline Exp5383-style solver guidance unless the `.491` corrigendum is clean.
- Do not count ARC solves from offline BFS, per-game adapters, outer-loop reverse engineering, or a
  level already listed in `ops/arc_solve_registry.yaml`.
- Do not make a hardware speedup claim without authenticated repeated board timing.
