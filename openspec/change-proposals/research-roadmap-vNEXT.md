# Research Roadmap vNEXT: 2026.07.479

Status: proposed for activation after milestone `2026.07.478`

Milestone title: Artifact Credibility, Controlled Self-Learning, and Verifier Decision Repair

## Inputs Read

This plan is grounded in:

- `research-program.md`
- `_bmad/prd.md`
- `_bmad/architecture.md`
- `ops/status.md`
- `ops/changelog.md`
- `research-complete.yaml`
- `research-roadmap.yaml` for completed milestone `2026.07.478`
- `openspec/change-proposals/`
- `ops/conductor-log.md`
- `research-references.md`
- `research-hardware-wishlist.md`
- `ops/exclusion_manifest.yaml`
- `ops/arc_solve_registry.yaml`
- `CLAUDE.md`
- `CODEX.md`

The plan also incorporates the V479 research update appended to `research-references.md` on 2026-07-04.

## What 2026.07.478 Proved

Milestone `.478` closed with useful progress but no new headline verifier claim:

- GAP-1 is positive but not promotable. `exp5222` parsed the `.477` positive gate correctly, but promotion
  stayed `blocked_instability` because the selected subset could not be frozen without held-out tuning.
- GAP-4 is scientifically unresolved because the clean-looking canonical-pool and validation artifacts were
  adversarially flagged. `exp5224` built a canonical n=120 pool, and `exp5225` produced a null validation
  with wins=0, losses=0, ties=120, but both were headline-ineligible under current TAUTOLOGY rules.
- VerIbmc-style local solver feedback is blocked by artifact hygiene, not by a clean scientific null.
  `exp5226` reported zero uplift but was flagged for DURATION_TOO_SHORT and missing methodology fields.
- Continuous self-learning made the strongest forward move. `exp5227` produced consumer-ready typed memory
  with four heads: constraints, provenance, failures, and skills/rubrics.
- ARC diagnosis improved but did not change behavior. `exp5228` produced a usable skill rubric, but no
  recommended live patch; `exp5229` was gate-blocked.
- KAN verification has a tiny working certificate path. `exp5230` produced a bounded KAEM PWA/MILP
  certificate, but it remains small-scale.
- Hardware continuity remains reachability-only. KV260 and PolarFire are reachable; GateMate remains blocked
  at physical/JTAG; no speedup claim exists.
- The capstone correctly excluded flagged and gated artifacts, leaving GAP-1, GAP-4, VerIbmc, and ARC as
  blocked or null while preserving typed memory and the tiny KAN certificate.

## Three Biggest Gaps to PRD Vision

1. **Artifact credibility is now a blocking dependency.**
   The PRD requires verifiable reasoning, but `.478` showed that valid-looking nulls and pool builders can be
   excluded by generic adversarial checks. `.479` must calibrate the QA layer around structured nulls and
   methodology receipts before spending more GPU time on verifier claims.

2. **Verifier evidence is not yet becoming deployable decisions.**
   GAP-1 is unstable, GAP-4 is blocked by QA flags, VerIbmc is methodology-blocked, and ARC has diagnosis
   without a live patch. `.479` should convert each into one of three honest states: promoted, clean null, or
   explicitly retired/blocked with a concrete next criterion.

3. **Self-learning memory exists, but needs controlled evidence of useful reuse.**
   Typed memory is consumer-ready, but the PRD's continuous self-learning goal needs controlled streams,
   aligned-vs-shuffled controls, retention checks, and degradation checks. `.479` should test memory as a
   decision-changing substrate, not just as a ledger.

## SOTA Findings Incorporated

The V479 scan added these experiment-driving references:

- Free-Energy Signatures for hallucination detection (`arXiv:2606.19404`): attention-derived thermodynamic
  and spectral descriptors can be tested later as frozen-model diagnostics without an external text scorer.
- FLaG latent evidence grouping (`arXiv:2606.00301`): supports group-conditioned reliability for typed
  verifier memory instead of a single global confidence number.
- JANUS structured bidirectional generation (`arXiv:2603.03748`): motivates dependency/backfill candidate
  protocols with receipts for future GAP-4 generation.
- Retrieval-Warmed Energy-Based Reasoning (`arXiv:2606.26476`): provides a five-arm ablation pattern for
  separating genuine aligned reuse from bias shift and random warm starts.
- AgentCL (`arXiv:2606.02461`): motivates controlled reusable task streams and degradation tests for
  continuous learning agents.
- Hard-CSP GNN benchmarks (`arXiv:2602.18419`): warns against neural CSP superiority claims without strong
  deterministic baselines.
- Analog KAN co-optimization (`arXiv:2606.27892`): informs long-run KAEM/KAN hardware mapping, but not a
  local speedup claim.
- Extropic TSU/XTR-0 public writing: continues to support the EBM sampler hardware thesis, but Carnot lacks
  local TSU access.
- Logical Intelligence Aleph/Kona posts: reinforce verifier-first architecture and formal proof substrates,
  without enough reproducible Kona detail to use as a baseline.
- Semantic Scholar API is reachable now for EBT (`arXiv:2507.02092`, 26 citations observed on 2026-07-04)
  and ARM-EBM (`arXiv:2512.15605`, 8 citations observed on 2026-07-04), so `.479` should record raw metadata
  instead of inheriting `.478`'s 429-only status.

## Architecture Target

```text
                  Local SOTA GGUF Models
   Qwen3.6-35B-A3B / Gemma-4-31B / Gemma-4-26B-A4B
                            |
                            v
                 +----------------------+
                 | Candidate Proposers  |
                 | - GAP-4 protocol     |
                 | - VerIbmc invariant  |
                 | - ARC patch proposal |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Calibrated Evidence  |
                 | - provenance fields  |
                 | - null/tautology QA  |
                 | - runtime receipts   |
                 | - model/seed records |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Deterministic Gates  |
                 | - Z3/SMT/ESBMC      |
                 | - ARC live registry  |
                 | - MILP/PWA certs    |
                 | - exclusion manifest |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Typed Self-Learning  |
                 | - aligned controls   |
                 | - shuffled controls  |
                 | - retention checks   |
                 | - rollback policy    |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | PRD-facing Decisions |
                 | - promoted verifier  |
                 | - clean null         |
                 | - explicit block     |
                 | - hardware boundary  |
                 +----------------------+
```

The milestone keeps the same architectural discipline: local models may propose, but deterministic gates and
calibrated artifact receipts decide. Memory updates must survive controlled controls before being described
as self-learning. Hardware remains a continuity/boundary track.

## Phase Plan

### Phase 0 - Archive and Refresh

Tasks:

- `exp5233-archive-478-activate-479`
- `exp5234-sota-ingestion-v479`

Goal: close `.478` truthfully, activate `.479`, and refresh the external-source map after the V479 planning
scan. The SOTA task must record Semantic Scholar raw responses for EBT and ARM-EBM.

### Phase 1 - Artifact QA and Verifier Decisions

Tasks:

- `exp5235-adversarial-qa-null-tautology-calibration-v479`
- `exp5236-gap4-clean-status-after-qa-calibration-v479`
- `exp5237-gap1-stability-freeze-or-retire-v479`
- `exp5238-veribmc-methodology-correct-rerun-or-retire-v479`

Goal: repair or document the QA false-positive boundary around structured nulls, then make clean decisions
on GAP-4, GAP-1, and VerIbmc without reopening retired external-scorer or broad candidate-search patterns.

### Phase 2 - Controlled Self-Learning and ARC Patch Path

Tasks:

- `exp5239-continuous-self-learning-controlled-memory-ablation-v479`
- `exp5240-arc-rubric-to-patch-synthesis-v479`
- `exp5241-arc-gated-live-patch-attempt-v479`

Goal: use `.478` typed memory as a real consumer under AgentCL/RW-EBR-style controls, then synthesize a
specific ARC live patch only if the rubric and memory expose a testable intervention. The live ARC attempt is
gated and must use live-agent provenance if it claims any level movement.

### Phase 3 - KAN, Hardware, and Capstone

Tasks:

- `exp5242-kan-certificate-abstraction-scale-v479`
- `exp5243-hardware-continuity-kan-pbit-boundary-v479`
- `exp5244-capstone-v479`

Goal: expand the tiny KAN certificate into a slightly broader abstraction stress test, keep hardware
continuity bounded to reachability/hash/sampler-boundary evidence, and reconcile all docs/specs without
claiming flagged or gated artifacts.

## Dependency Graph

```text
exp5233 archive/activate
  |
  +--> exp5234 SOTA ingestion
  |
  +--> exp5235 QA null/tautology calibration
          |
          v
      exp5236 GAP-4 clean status after calibration

exp5233 archive/activate
  |
  +--> exp5237 GAP-1 stability freeze/retire
  |
  +--> exp5238 VerIbmc methodology-correct rerun

exp5234 SOTA ingestion
  |
  +--> exp5239 controlled typed-memory ablation
          |
          v
      exp5240 ARC rubric-to-patch synthesis
          |
          v
      exp5241 ARC gated live patch attempt

exp5234 SOTA ingestion
  |
  +--> exp5242 KAN certificate abstraction scale
  |
  +--> exp5243 hardware continuity

all completed, blocked, or gated artifacts
  |
  v
exp5244 capstone
```

Structured gates:

- `exp5236` runs only when `exp5235.qa_calibration_passed == true`.
- `exp5241` runs only when `exp5240.recommended_live_patch_available == true` and
  `exp5240.patch_test_ready == true`.

## Model and Inference Requirements

Every experiment that uses an LLM must include at least one mandated local SOTA GGUF model in its model
specification:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

LLM-bearing prompts must direct implementers to use `scripts/experiment_template.py` and the
`cached_sota_pair()` pattern when local inference is needed. Legacy small models may be used only as CPU
smoke tests and cannot support headline claims.

LLM-bearing tasks in this milestone:

- `exp5238-veribmc-methodology-correct-rerun-or-retire-v479`
- `exp5240-arc-rubric-to-patch-synthesis-v479` if it uses a local proposer
- `exp5241-arc-gated-live-patch-attempt-v479` if the live patch invokes a model proposer

## Hardware Requirements

- Dual RTX 3090 CUDA box: required for any SOTA GGUF inference in VerIbmc or ARC patch tasks.
- CPU-only path: enough for archive, SOTA ingestion, QA calibration, GAP-4 reclassification, GAP-1 stability
  analysis, controlled memory ablation, and KAN/MILP stress tests if solvers are local.
- KV260: SSH-only reachability and hash smoke. Do not require host-visible `/dev/mmcblk` devices.
- PolarFire: SSH/hash smoke only unless an existing safe workload is already present.
- GateMate: no rerun unless the operator has changed physical/JTAG setup; otherwise record the existing
  blocked physical state.
- Extropic TSU/XTR-0: watch only; there is no local hardware access.

## No-Go Rules

- Do not modify `scripts/research_conductor.py`.
- Do not modify `research-roadmap.yaml` during planning.
- Do not push.
- Do not reopen the retired external text-scorer Phase D scope.
- Do not claim ARC solves from offline source reading, exhaustive ground-truth BFS, or a hand per-game
  development adapter.
- Do not claim GAP-4, VerIbmc, or KAN headline evidence from artifacts still adversarially flagged.
- Do not claim hardware speedups without a real workload, baseline, measurements, and reproducible hashes.
- Do not use legacy tiny GGUF models for headline LLM results.

## Success Criteria

`.479` succeeds if it produces:

- A clean archive/activation artifact and current SOTA ingestion artifact.
- A documented QA calibration decision that either fixes structured-null false positives with tests or
  explains why the current flags are scientifically correct.
- A GAP-4 status that is headline-eligible if clean, or explicitly blocked with no ambiguous null claim.
- A GAP-1 stability decision that either freezes a non-leaky subset or retires/blocks the current promotion
  path.
- A methodology-correct VerIbmc result or an explicit retirement decision if zero uplift repeats under clean
  receipts.
- A controlled self-learning result with aligned, shuffled, random, constant, and no-memory controls.
- An ARC patch artifact that either gates a live attempt or records why no patch is justified.
- A KAN certificate stress artifact that expands beyond the `.478` tiny proof without overclaiming.
- A hardware continuity artifact with reachability/hash facts and no speedup claim.
- A capstone that reconciles `openspec/`, `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  `research-references.md`, and exclusion/provenance decisions.
