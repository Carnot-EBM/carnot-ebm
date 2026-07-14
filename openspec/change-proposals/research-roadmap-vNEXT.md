# Research Roadmap vNEXT — Milestone 2026.07.507

**Title:** Native Local-SOTA Runtime Certification, Drift-Aware KAN Self-Learning, Forward-Inverse ARC Verification, and Exact cDLS
**Status:** Proposed
**Task range:** Exp5613–Exp5624
**Execution manifest:** `research-roadmap-next.yaml`

## Milestone thesis

Milestone `.506` produced two promotable assets and three bounded negative results. The
lossless response envelope preserved eight model calls with perfect replay and zero semantic
false accepts. The exact-gated active-spline KAN learner improved held-out behavior, retained
earlier constraints, rejected poison, and passed rollback and delayed-regression controls. In
contrast, the three-family local-SOTA panel never established CUDA offload, every family
collapsed at parsing, and the panel shape was retired. The ARC inert-click and object-history
filters were reachable but repeated their no-op and were retired; the unconditional `sk48` L8
attempt added no reproducible level. The cDLS run exercised CPU and CUDA code, but used only
one seed and produced zero quality-matched pairs, so neither exact-target equivalence nor a
crossover claim survived.

`.507` follows those boundaries. It certifies a changed native llama.cpp CUDA substrate without
reopening the retired solve-versus-verify panel. It advances the clean KAN component from one
bounded sequence to controlled nonstationary streams, using Average Lifelong Error and Critical
Task Duration to decide when retention becomes harmful. It adapts World Action Verifier's
forward/inverse asymmetry to the ARC live path using only the agent's own actions and
observations, then runs the standing unconditional level attempt. Finally, it repairs cDLS at
the Markov-kernel level before spending another multi-seed CPU/CUDA benchmark.

## What milestone `.506` proved

| Evidence | Terminal fact | Consequence for `.507` |
|---|---|---|
| Exp5603 transition | `.505` and the post-milestone outer loop were archived with parser, causal-memory, PTRM, and ARC boundaries intact. | Allocate a non-colliding Exp5613–Exp5624 range and do not revive retired chains. |
| Exp5604 source refresh | One KAN lazy-identity diagnostic was actionable; the artifact was later flagged because its source-only substrate looked compute-bound. | Keep a source-ingestion slot, use an aggregation provenance category, and treat no-op discovery as valid. |
| Exp5605 response envelope | Eight rows replayed losslessly; truncation and corruption controls passed; semantic false accepts were zero. | Reuse the envelope as runtime evidence infrastructure. Do not rerun its schema. |
| Exp5606 local-SOTA panel | All three mandated GGUFs were cached, but `gpu_offload_authenticated=false`; parser failure was total and Gemma-26B truncation was high. `panel_complete=false`. | Retire this panel shape. Build a native-runtime certificate only; make no verifier-quality claim in `.507`. |
| Exp5607 gate | Exact residual extension did not run because the panel supplied no clean residuals. | Do not propose another residual extension until a future milestone has a clean, independently justified corpus. |
| Exp5608 KAN longitudinal learner | Exact-gated active splines passed the benefit and safety gates, with positive held-out and backward-retention deltas, poison rollback, and no LLM-weight mutation. Forward transfer remained zero. | Preserve this as the sole FR-11 learning substrate and test it under drift, duration, and family shift. |
| Exp5609 ARC filter A/B | Both filters were reachable but produced the same downstream no-op; both were retired. | No more candidate-pruning or salience-filter variants. Improve transition-model trust instead. |
| Exp5610 ARC live attempt | The live agent attempted `sk48` L8 from its own observations but banked no new level. | Rotate target/game, change the reachable model-update branch, and retain an unconditional baseline fallback. |
| Exp5611 cDLS benchmark | CPU/CUDA execution rows existed, but one seed and zero quality-matched pairs made the timing result inadmissible. | Prove detailed balance and exact small-state target parity first; then use at least three seeds and preregistered quality gates. |
| Exp5612 capstone | Response preservation and KAN promoted; local-SOTA asymmetry, exact extension, ARC filters/level-up, and cDLS crossover did not. | The next milestone centers on runtime readiness, nonstationary FR-11, ARC transition verification, and sampler correctness. |

## The three largest gaps to the PRD vision

### Gap 1 — local-SOTA execution is preserved but not operationally trustworthy

FR-12 ultimately needs local proposals whose exact verification path is replayable. Carnot now
has the replay contract, but its Python llama.cpp route could not authenticate GPU offload and
the retired panel produced no usable structured responses. Repeating that panel would violate
the failure ledger. `.507` instead asks a narrower infrastructure question: can the native
llama.cpp binary or server load, offload, generate, terminate, and replay a minimal structured
response for each mandated GGUF with device/process evidence? A certificate unlocks later
science; it is not itself evidence that an LLM verifies better than it solves.

### Gap 2 — continuous self-learning has no policy for nonstationary constraints

Exp5608 showed that a spline-local KAN component can update safely in one ordered sequence, but
forward transfer was zero and the task did not distinguish transient shifts from persistent rule
changes. FR-11 needs a controller that decides whether to retain, smooth, reset, or adapt as the
constraint distribution changes. `.507` builds an exact stream with domain-space and temporal
drift axes, estimates the empirical Critical Task Duration, and gates a predictive-window KAN
controller on a nondegenerate switch signal. Exact validation, poison rollback, delayed
regression, and immutable decisions remain authoritative.

### Gap 3 — the ARC live agent lacks a trustworthy self-updating transition model

The north-star registry did not move in `.506`, and the filter, PTRM, scoring, larger-generator,
and exploration-signal families are closed. A different weakness remains: an online planner can
poison itself when a learned forward effect attributes the wrong action or successor state.
World Action Verifier suggests a distinct mechanism—verify state plausibility and inverse action
reachability separately, then demand forward/inverse cycle consistency. `.507` implements that
contract from the live agent's own action/observation history, measures whether verified updates
improve model fidelity on already reproduced levels, and only then exposes the promoted branch
to a registry-prechecked live level attempt.

## 2025–2026 research incorporated

The `V507 Planner Refresh - 20260714` block was appended to `research-references.md` before this
roadmap was designed.

| Source | Executable use in `.507` |
|---|---|
| To Retain or to Adapt?, arXiv:2607.05609 | Define ALE, instability, transient error, and an empirical Critical Task Duration over exact constraint streams. |
| When Does Continual Learning Require Learning, arXiv:2607.07847 | Cross domain-space shifts with temporal drift and compare external/frozen control with spline-local learning under one protocol. |
| Loss Smoothing for Continual Adaptation, ICLR 2026 CAO Workshop | Add a bounded loss-smoothed adaptation arm; exact validators decide whether smoothing helps or merely delays necessary change. |
| World Action Verifier, arXiv:2604.01985 | Build a generic forward/inverse action-effect cycle verifier from ARC live observations, with no external video or game-specific adapter. |
| cDLS, OpenReview ProbML 2026 | Retain only the continuous-intermediate proposal hypothesis; require local exactness and mixing evidence before timing claims. |

The direct Semantic Scholar EBT (`2507.02092`) and ARM-EBM (`2512.15605`) citation trails added
no stronger executable dependency. Hugging Face Papers and GitHub searches repeated indexed
verifier and constrained-decoding work. Extropic still exposes no authenticated local TSU route,
and Logical Intelligence publishes no local Kona artifact. Those systems remain context, not
evidence.

## Target architecture

```text
  CACHED LOCAL GGUFs + LOSSLESS RESPONSE ENVELOPE
                 |
                 v
  +----------------------------------------------+
  | native llama.cpp CUDA runtime certificate    |
  | Qwen3.6-35B-A3B + Gemma-4-31B + 26B-A4B     |
  | model hash + offload + PID/GPU + replay      |
  +----------------------------------------------+
                 (readiness only; no verifier claim)

  EXACT CONSTRAINT-DRIFT STREAM
  space shift x temporal drift x task duration
                 |
                 v
  +--------------------------+     +----------------------------+
  | KAN duration map         |---->| predictive-window KAN      |
  | ALE / instability /      |gate | retain / smooth / reset /  |
  | transient error / switch |     | adapt + exact rollback     |
  +--------------------------+     +----------------------------+

  ARC LIVE ACTIONS + OBSERVATIONS ONLY
                 |
                 v
  +----------------------------------------------+
  | forward/inverse transition-cycle verifier    |
  | plausible successor + reachable action       |
  | + cycle consistency + corruption controls    |
  +----------------------+-----------------------+
                         |
                         v
  matched known-level integration A/B --promotion advisory--> live +1 attempt
                                                               |
                                                       reproduce -> registry

  EXISTING cDLS KERNEL
          |
          v
  exact small-state stationary/detailed-balance audit
          |
          +--quality gate--> multi-seed matched CPU/CUDA crossover
```

## Phase 0 — continuity, freshness, and native runtime readiness (Exp5613–Exp5615)

**Exp5613 — `.506` to `.507` transition.** Lock every `.506` terminal artifact, record the two
promotions and the retired/blocked branches, allocate Exp5613–Exp5624, and emit the new gate map.
This is infrastructure slot one.

**Exp5614 — execution-time source delta.** Search all mandated sources after the V507 planner
marker, deduplicate against the complete reference and failure ledgers, and map only genuinely
new executable deltas. A clean no-op is terminal success. This is the SOTA-ingestion slot.

**Exp5615 — native llama.cpp CUDA runtime certificate.** Use all three mandated GGUF families,
the Exp5605 envelope, and a changed native CLI/server substrate. Authenticate build capability,
offloaded layers, process/GPU memory, stop behavior, and response replay. Run only small
structured positive/truncation controls. Do not calculate solve-versus-verify accuracy or reopen
Exp5606. This is infrastructure slot two and is routed to Opus because it is hardware/runtime
integration with multiple failure modes.

## Phase 1 — drift-aware continuous self-learning (Exp5616–Exp5618)

**Exp5616 — exact nonstationary constraint-stream fixture.** Generate a deterministic,
machine-checkable stream crossing spatial constraint-family shifts with temporal predicate drift,
shared versus conflicting rules, and controlled durations. Include no-drift, reversible-drift,
and persistent-drift controls; publish exact labels and checksums before learner evaluation.

**Exp5617 — KAN Critical Task Duration map.** Reuse the clean active-spline KAN substrate and
compare retain/replay, reset/adapt, loss-smoothed, and frozen arms over preregistered duration
cells. Measure ALE, instability, transient error, backward retention, unsafe accepts, and the
empirical switch point with uncertainty. This experiment identifies the policy boundary; it does
not choose adaptively.

**Exp5618 — predictive-window KAN continuous self-learning.** Gate on a nondegenerate and safe
Exp5617 duration map. Let a causal controller use only past exact-energy/residual history to
choose retain, smooth, reset, or adapt. Compare against the best fixed arm and an oracle selector
that is reported but cannot headline. Promotion requires lower held-out ALE, bounded regret,
positive transfer or faster valid adaptation, backward safety, poison rejection, rollback, and a
lazy-identity guard. This is the milestone's required continuous self-learning experiment.

## Phase 2 — forward/inverse ARC transition verification and level attempt (Exp5619–Exp5621)

**Exp5619 — ARC transition-cycle verifier prototype.** On agent-owned live traces from already
reproduced levels, learn generic action-effect signatures and verify each transition by inverse
action recovery plus forward replay. Hold out episodes within each game; do not claim cross-game
transfer. Permuted actions and successors provide corruption controls. The task receives
`solve_provenance=development_proxy` and no new-level credit.

**Exp5620 — gated live-path transition-update A/B.** Gate on the prototype's clean positive and
negative controls. Wire verified transition updates into the generic live planner, then compare
baseline versus cycle-guarded updating on at least three already reproduced games under matched
seeds and budgets. Promotion requires actual verifier reachability, lower held-out forward error
or invalid-plan rate, no known-level regression, and a downstream action/search improvement.
This changes transition-model updates, not proposal pruning, salience filtering, or an intrinsic
reward.

**Exp5621 — unconditional live-agent `+1` attempt.** Registry-precheck every candidate level,
exclude `sk48` L8 and all already reproduced targets, and rotate toward authenticated headroom.
Use Exp5620 only if it promoted cleanly; otherwise run the unchanged no-new-LLM live baseline.
The attempt is not conductor-gated. Only the live agent's own discovery followed by independent
generic reproduction and registry banking counts.

## Phase 3 — corrected cDLS and reconciliation (Exp5622–Exp5624)

**Exp5622 — exact cDLS kernel audit.** Enumerate small Ising state spaces, construct or estimate
the transition kernel, and test normalization, support, detailed balance, stationary-distribution
total variation, and deterministic replay. Add a Metropolis-Hastings or otherwise proven
correction if the continuous projection is biased. Predeclare the large-n quality-equivalence
gate before timing.

**Exp5623 — gated multi-seed CPU/CUDA crossover.** Run only if Exp5622 passes exactness. Compare
corrected cDLS with the unchanged discrete Langevin baseline on identical targets, schedules,
sample counts, and at least three seeds across `n=128,256,512,1024` where memory permits. Report
acceptance, ESS, autocorrelation, energy distribution, constraint satisfaction, raw timing, and
authenticated devices. Only successful quality-matched pairs enter a speedup or crossover claim.

**Exp5624 — `.507` capstone reconciliation.** Aggregate all eleven upstream tasks, apply every
retirement rule, run adversarial/spec/roadmap/ARC checks and applicable end-to-end tests, and
reconcile OpenSpec, traceability, completion, status, changelog, exclusions, references, and the
ARC registry. It cannot upgrade a blocked, skipped, development-proxy, or unmatched result.

## Dependency graph

```text
Exp5613 transition ----------------------------------------------+
Exp5614 source delta --------------------------------------------+----> Exp5624 capstone

Exp5605 envelope + changed native CUDA substrate ---------------> Exp5615 runtime certificate
                                                                    (no downstream panel in .507)

Exp5616 exact drift fixture
    └──[fixture/oracle clean]──> Exp5617 KAN duration map
                                  └──[switch fit + safe]──> Exp5618 predictive KAN CSL

Exp5619 forward/inverse cycle verifier
    └──[positive + corruption controls]──> Exp5620 live update A/B
                                             └── advisory only ─┐
current live baseline + registry precheck -----------------------> Exp5621 +1 attempt

Exp5622 exact cDLS kernel audit
    └──[detailed balance + target parity]──> Exp5623 matched CPU/CUDA benchmark

Exp5613–Exp5623 -----------------------------------------------> Exp5624 reconciliation
```

Exp5621 is intentionally not structured-gated on Exp5620. The standing ARC floor requires a real
attempt even when the new mechanism fails. Exp5615 likewise has no scientific panel downstream
in this milestone: its purpose is to close the runtime precondition without laundering a minimal
smoke test into verifier evidence.

## Hardware and model requirements

| Resource | Tasks | Requirement and claim boundary |
|---|---|---|
| Local GGUF cache | Exp5615 | `MODEL_SPECS` contains `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF`. Legacy Qwen3.5-0.8B and Gemma-E4B are smoke-only and cannot satisfy the certificate. |
| Native llama.cpp CUDA build | Exp5615 | Native CLI/server capability, version, build flags, `--single-turn` behavior, offloaded layers, PID, and GPU memory must be authenticated. CPU fallback is diagnostic only. |
| NVIDIA GPUs | Exp5615, Exp5623 | Record device/driver/runtime/free memory and process or kernel evidence. Exp5623 may use CUDA only after the exact CPU kernel audit passes. |
| System RAM / NVMe | Exp5615–Exp5618 | Cached 26–35B GGUFs, lossless response rows, exact drift fixtures, checkpoints, and immutable decision ledgers. |
| CPU | Exp5616–Exp5623 | Exact validators, KAN stream runs, ARC environment, small-state enumeration, and the matched sampler baseline. Record identity and wall time. |
| KV260 / PolarFire / GateMate / TSU | none required | No board or proprietary-hardware speedup claim is in scope. Existing wishlist entries remain unchanged. |
| Network | Exp5614 only | Literature discovery. All model inference, learning, ARC execution, and sampler work remain local-first. |

## Promotion and retirement rules

- **Native runtime:** certify only if every mandated model has a real cached file/hash, native
  CUDA capability, nonzero offload evidence, lossless envelope replay, correct termination, and
  zero semantic false accepts. The same no-offload verdict retires this native-certificate attempt.
- **Duration map:** accept a switch boundary only when multiple duration/family cells contain
  both retention-favorable and adaptation-favorable regimes, the fit and intervals are reported,
  and unsafe false accepts remain zero.
- **Continuous self-learning:** promote only when the causal predictive controller beats the best
  fixed non-oracle baseline on held-out ALE without backward safety loss, passes poison/rollback
  and delayed-regression controls, and demonstrably changes the active spline state.
- **ARC transition verifier:** corruption rejection and inverse/forward cycle controls must pass
  before live wiring. Promotion requires reachable transition decisions and downstream model or
  planning improvement; another reachable no-op retires the mechanism.
- **ARC solve:** only `solve_provenance=live_agent_self_discovery`, independent generic replay,
  and a new registry entry count. Development proxies, source reads, per-game adapters, and
  outer-loop reverse engineering receive no solve credit.
- **cDLS:** detailed balance and exact small-state target parity precede timing. Speedup/crossover
  requires at least three seeds, identical targets/schedules, successful quality-matched pairs,
  and intervals excluding parity in the favorable direction.

## Expected milestone outputs

1. A reusable three-model native llama.cpp CUDA runtime certificate or terminal retirement.
2. An exact, checksummed nonstationary constraint-stream benchmark.
3. A measured KAN retention/adaptation duration boundary.
4. A safe predictive-window continuous self-learning verdict under spatial and temporal drift.
5. A generic ARC forward/inverse transition-cycle verifier and live-path promotion decision.
6. At least one real registry-prechecked live-agent attempt to bank `+1` reproducible ARC level.
7. An exactness-first verdict on cDLS and, only if justified, a multi-seed CPU/CUDA crossover.
8. Reconciled OpenSpec, traceability, status, changelog, completion, exclusion, reference, and ARC records.
