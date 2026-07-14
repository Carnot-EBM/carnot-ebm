# Research Roadmap vNEXT — Milestone 2026.07.506

**Title:** Evidence-Preserving Local Verification, Longitudinal KAN Self-Learning, Live ARC Filter Adjudication, and Hybrid Sampling
**Status:** Proposed
**Task range:** Exp5603–Exp5612
**Execution manifest:** `research-roadmap-next.yaml`

## Milestone thesis

Milestone `.505` closed two tempting but invalid continuations. Hash-only cached records made
the 648 local-SOTA parser failures unrecoverable, so no solve-versus-verify claim survived.
The causal-memory corrigendum reconstructed independent metrics, but forward transfer stayed
zero and the policy lane failed promotion; the gated PACE continuation therefore did not run.
The exact verifier extension was also correctly skipped because it had no clean residual set.
The ordinary ARC floor reproduced known levels but added none. Subsequent outer-loop evidence
retired PTRM as a generator, found no full-stack scoring advantage, kept the 9B frozen ARC
generator after the larger candidate lost on the actual reinduction path, and showed that the
new inert-click pruner was a no-op at the measured `m0r0` budget.

`.506` does not retry those chains unchanged. It repairs evidence preservation before making
new local-SOTA calls, replaces the retired memory-policy chain with a KAN-only longitudinal
FR-11 test, measures the already-wired ARC filters at downstream intermediates with a positive
control, and keeps the mandatory live-agent level-up attempt unconditional. A new cDLS-inspired
sampler lane asks whether hybrid proposals improve quality-adjusted mixing or CPU/CUDA crossover
without inheriting a workshop paper's claims.

## What milestone `.505` and the immediate outer loop proved

| Evidence | Terminal fact | Consequence for `.506` |
|---|---|---|
| Exp5578 transition | `.504` boundaries were preserved: 648 parser failures, flagged memory evidence, clean KAN, zero ARC registry delta. | Start from terminal artifacts, not the superseded 14-task draft. |
| Exp5579 source refresh | Property templates and entity-attribution failures were actionable exact-verifier ideas. | Apply them only after a clean, preserved residual panel exists. |
| Exp5580 parser forensics | 648 hash-only records split into 468 truncations and 180 other failures; raw responses were unavailable. Synthetic controls passed, but `parser_repair_ready=false`. | A response envelope and lossless replay positive control must precede new model calls. |
| Exp5581/5582 gates | Remeasurement was gate-blocked; exact extension was preemptively skipped. | Run a new panel only after the evidence envelope passes. No cached-row resurrection. |
| Exp5583 metric corrigendum | Independent metrics were reconstructed, but forward transfer was `0`, forgetting was positive, and `policy_ready=false`. | Retire the causal-memory/two-timescale policy chain. |
| Exp5584 gate | PACE-style controller was gate-blocked three times. | Continuous self-learning must use the clean KAN substrate without the failed policy prerequisite. |
| Exp5585 ARC floor | `lf52` L2 and prior known L6 were reproduced through a development proxy; no new level was banked. | Rotate the target and improve the reachable live path; do not re-solve `lf52`. |
| Exp5600 PTRM LOO | Only one of five held-out games passed; PTRM-as-generator was retired. | No PTRM generator continuation. |
| Exp5592/5599 scoring and generator A/Bs | The full scoring stack had no level/efficiency advantage; the 27B candidate lost to the frozen 9B on actual reinduction and cost about seven times more. | Keep the current frozen live generator; this milestone targets classical reachable filters. |
| Exp5601/5602 ARC filters | Object history saw two real change-positive hashes, but no same-base divergence; inert-click pruning observed no signatures and changed no expansions at the measured budget. | Require a reachable positive control and downstream intermediate ledger; repeat null retires the filters. |

## The three largest gaps to the PRD vision

### Gap 1 — local-SOTA verification still lacks an admissible evidence chain

Carnot has exact ASP/FSM labels but no valid local-SOTA solve-versus-verify result. The last
panel discarded the raw material needed to distinguish malformed output, truncation, and parser
bugs. FR-12 needs replayable evidence from prompt through exact verdict. `.506` therefore ships
an append-only response envelope with raw or losslessly compressed bytes, content hashes,
runtime/model receipts, parser version, and exact-validator outcome before spending a full panel.

**Implication:** a clean null is publishable evidence; another hash-only aggregate is not.

### Gap 2 — FR-11 has a component, not a longitudinal self-learning system

The active-spline KAN updater passed exact-energy tests, but the causal-memory policy did not
show forward transfer and its dependent controller never ran. FR-11 requires persistent online
adaptation, held-out improvement, non-forgetting, rollback, and immutable decisions. `.506`
tests the clean KAN alone across ordered sessions with delayed replay, shuffled-order and frozen
controls, drift/poison injection, checkpoints, and rollback.

**Implication:** success means bounded, exact-gated online weights on a non-LLM component; it
does not imply autonomous LLM fine-tuning or general continual intelligence.

### Gap 3 — the live ARC agent still does not discover new levels reliably

The registry remains at 69 reproducible levels across 24 games. PTRM, richer candidate scoring,
and a larger generator did not promote. Two cheaper live-path mechanisms now exist, but neither
has a downstream benefit measurement with a reachable mechanism positive control. `.506` first
adjudicates those filters under matched budgets, then runs a separate live-agent self-discovery
attempt even if the filters fail.

**Implication:** candidate or state reduction is not solve credit; only the live agent's own
attempt plus offline reproduction and registry banking can increase the north-star metric.

## 2025–2026 research incorporated

The `V506 Planner Refresh - 20260714` block was appended to `research-references.md` before this
roadmap was designed.

| Source | Executable use in `.506` |
|---|---|
| ScientistOne / Chain-of-Evidence, arXiv:2605.26340 | Lossless local response envelope and claim-to-row provenance before remeasurement. |
| Aggregate invariants for continuous subgraph matching, arXiv:2606.24421 | Intermediate-invariance ledger for ARC pruning: candidates, actions, states, expansions, level gains, and wall time. |
| cDLS, OpenReview ProbML 2026 | Bounded hybrid continuous/discrete proposal benchmark against the existing exact-target discrete sampler. |
| Agentic property templates, arXiv:2607.09072 | Human-auditable exact predicate templates for clean residuals. |
| Deceptive Grounding, arXiv:2607.09349 | Wrong-variable/entity attribution stress class when real residuals support it. |
| Temporal-difference visual representations, arXiv:2606.15956 | Watch-only; no reopened replay or representation-pretraining lane in `.506`. |

Direct Semantic Scholar citation queries for EBT (`2507.02092`) and ARM-EBM
(`2512.15605`) added no stronger executable dependency. Extropic still exposes no authenticated
local TSU route; Logical Intelligence's Kona remains proprietary architecture context; GitHub
and Hugging Face discovery produced no replacement for Carnot's exact validators or local
llama.cpp path.

## Target architecture

```text
          EXACT ASP/FSM CORPUS + DETERMINISTIC VALIDATORS
                              |
              +---------------v----------------+
              | append-only response envelope  |
              | raw bytes + hashes + receipts  |
              | prompt + parser + exact result |
              +---------------+----------------+
                              |
         +--------------------v---------------------+
         | local SOTA solve-vs-verify panel         |
         | Qwen3.6-35B-A3B + Gemma-4-31B/26B-A4B   |
         | llama.cpp CUDA, paired exact labels      |
         +--------------------+---------------------+
                              |
                   clean held-out residuals
                              |
              +---------------v----------------+
              | property-template exact        |
              | verifier extension             |
              | entity/variable attribution    |
              +--------------------------------+

     +--------------------------+       +---------------------------+
     | FR-11 KAN-only learner   |       | live ARC reachable path   |
     | ordered sessions         |       | inert + object-history A/B|
     | held-out + delayed replay|       | intermediate ledger       |
     | poison -> rollback       |       +-------------+-------------+
     +--------------------------+                     |
                                           unconditional live-agent
                                           self-discovery +1 attempt
                                                       |
                                           reproduce -> registry bank

     +--------------------------------------------------------------+
     | exact Ising descriptors -> discrete DLS vs cDLS              |
     | identical targets/seeds/samples -> CPU/CUDA quality + timing |
     +--------------------------------------------------------------+
```

## Phase 0 — continuity, freshness, and evidence infrastructure (Exp5603–Exp5605)

**Exp5603 — `.505` to `.506` transition.** Archive the eight conductor tasks plus the relevant
post-milestone outer-loop artifacts, allocate the non-colliding range Exp5603–Exp5612, and emit
the dependency/gate map. This is infrastructure slot one.

**Exp5604 — execution-time source delta.** Search all mandated current sources after the V506
planner marker, deduplicate against the full references ledger and exclusion manifest, and map
only executable deltas to remaining tasks. An honest no-op is valid. This is the SOTA-ingestion
slot.

**Exp5605 — response evidence envelope.** Implement the missing provenance contract and prove
byte-for-byte replay, truncation visibility, parser-version replay, and semantic fail-closed
behavior on small cached local-SOTA calls. This is infrastructure slot two and the only task
allowed to create the gate for expensive inference.

## Phase 1 — evidence-preserving verifier co-evolution (Exp5606–Exp5607)

**Exp5606 — clean local-SOTA solve-versus-verify panel.** Gate on the envelope's lossless replay
and semantic positive control. Use all three mandated headline GGUF families where cached and
runtime-feasible: `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Exact labels remain authoritative, raw outputs remain
replayable, and per-family parser ceilings prevent averaging away failure. This is one final
remeasurement shape; another evidence collapse retires it.

**Exp5607 — property-template exact residual extension.** Gate on a clean panel, split residuals
by constraint family, and propose only deterministic human-auditable predicates. Include
wrong-entity/wrong-variable attribution only when the clean residual ledger contains it. Require
zero unsafe false accepts, no valid-row rejection regression, held-out residual reduction, and
bounded overhead. No LLM judge or generated-text energy scorer is introduced.

## Phase 2 — longitudinal self-learning and live ARC (Exp5608–Exp5610)

**Exp5608 — KAN-only longitudinal continuous self-learning.** Reuse Exp5570's active-spline
updater without the retired memory policy. Compare frozen, shuffled-order, always-update, and
exact-gated arms over at least four ordered sessions and held-out families. Require independent
forward transfer, delayed backward retention, zero unsafe false accepts, immutable checkpoints,
poison rejection, and successful rollback. This is the milestone's required continuous
self-learning experiment.

**Exp5609 — ARC filter intermediate-invariance A/B.** Measure the already-wired inert-click and
object-history mechanisms independently and together under matched seeds, rosters, and action
budgets. First prove each mechanism is reachable on a non-source-aware trace; then report
candidates, actions, states, expansions, level gains, wall time, and reproduction. If no
downstream intermediate changes or the same null repeats, retire the affected filter.

**Exp5610 — unconditional live-agent `+1` level attempt.** Registry-precheck the 25-game roster,
exclude already-reproduced target levels, and choose a rotated target with reachable headroom.
Use promoted filters only if Exp5609 passed; otherwise use the current live baseline. Credit only
`solve_provenance=live_agent_self_discovery`, then offline reproduce and bank a genuinely new
level. No source reads, exhaustive BFS, per-game adapter, or development-proxy headline is
admissible.

## Phase 3 — sampling systems evidence and reconciliation (Exp5611–Exp5612)

**Exp5611 — cDLS matched CPU/CUDA benchmark.** Implement a bounded continuous-intermediate
proposal beside the existing discrete Langevin sampler. Run identical exact Ising targets,
seeds, warmup, retained samples, and schedules at `n=128,256,512,1024` where resources permit.
Report ESS, autocorrelation, energy/constraint quality, acceptance, wall time, and authenticated
device receipts. Claim crossover only on successful quality-matched pairs.

**Exp5612 — capstone reconciliation.** Aggregate all nine upstream artifacts, run adversarial,
spec-coverage, roadmap, and ARC integrity checks, update the completion ledger and operational
docs, and issue narrow promotion/retirement decisions. This planning/infrastructure slot cannot
upgrade blocked or skipped evidence.

## Dependency graph

```text
Exp5603 transition ------------------------------+
Exp5604 source delta ----------------------------+----> Exp5612 capstone

Exp5605 response envelope
    └──[lossless_replay && semantic_controls]──> Exp5606 clean SOTA panel
                                                    └──[panel clean]──> Exp5607 exact extension

Exp5570 clean KAN evidence ---------------------> Exp5608 KAN longitudinal CSL

Exp5601 signal + Exp5602 null ------------------> Exp5609 filter A/B
                                                    └── advisory only ─┐
current live baseline + registry precheck -----------------------------> Exp5610 +1 attempt

Exp5573 matched sampler evidence + cDLS source --> Exp5611 hybrid crossover

Exp5603–Exp5611 --------------------------------> Exp5612 reconciliation
```

Exp5610 is intentionally not conductor-gated on Exp5609: the standing ARC floor requires a real
attempt even when the candidate mechanism fails. The task branches internally to the unchanged
live baseline when filter promotion is false.

## Hardware and model requirements

| Resource | Tasks | Requirement and claim boundary |
|---|---|---|
| Local GGUF cache | Exp5605–5606 | Headline inference uses at least one mandated model per task; Exp5606 targets all three mandated families. Legacy 0.8B/E4B models are smoke-only and excluded from headline metrics. |
| NVIDIA CUDA GPUs | Exp5605–5606, Exp5611 | Authenticate GPU identity, PID/process, offloaded layers or kernel device, free memory, and runtime. CPU fallback blocks a CUDA headline but may produce a labeled diagnostic. |
| System RAM / NVMe | Exp5605–5606 | Preserve raw or losslessly compressed response envelopes locally; record size and content hash. Never retain only hashes. |
| CPU | Exp5608–5611 | Exact validators, KAN sessions, ARC environment, and matched sampler baseline. Record processor/runtime and wall time. |
| KV260 / PolarFire / GateMate | none required | Prior board lanes are terminal or status-only. `.506` makes no board speedup claim and does not reopen physical bring-up. |
| Network | Exp5604 only | Research discovery. All model inference and verification remain local-first. |

## Promotion and retirement rules

- **Verification:** promote only with lossless replay, exact-oracle agreement, per-model parser
  failure at or below the preregistered ceiling, and uncertainty-aware paired effects. Repeated
  evidence collapse retires this panel shape.
- **Exact extension:** promote only with zero unsafe false accepts, held-out residual reduction,
  no valid-row regression, and bounded runtime overhead.
- **Self-learning:** promote only with positive held-out longitudinal utility, independent
  forward/backward metrics, poison rejection, rollback success, and no LLM-weight mutation.
- **ARC filters:** a candidate-count reduction alone is insufficient. A repeat no-op with a
  reachable positive control retires the corresponding mechanism.
- **ARC solve:** only live-agent self-discovery followed by offline reproduction and a new registry
  entry counts. Development proxies and outer-loop RE remain diagnostics.
- **Sampling:** speedup or crossover requires identical targets/schedules, successful matched
  pairs, and non-inferior quality/mixing. Workshop claims and unmatched timing support no result.

## Expected milestone outputs

1. A reusable evidence-preserving response contract for local GGUF research.
2. The first admissible clean result—or bounded retirement—on local-SOTA solving versus verifying.
3. A safe exact predicate extension decision grounded in real residuals.
4. A longitudinal, rollback-tested verdict on spline-local KAN continuous self-learning.
5. A decision-grade retirement/promotion verdict for the two live ARC filters.
6. At least one real rotated live-agent attempt to bank `+1` reproducible ARC level.
7. A quality-matched CPU/CUDA verdict on continuous-intermediate discrete sampling.
8. Reconciled OpenSpec, traceability, status, changelog, completion, and exclusion records.
