# Carnot Research Roadmap vNEXT: Verification Co-evolution, Reset-free Self-learning, and Live Generation

**Created:** 2026-07-11
**Milestone:** 2026.07.504
**Status:** Planned; activates only after milestone 2026.07.503 is reconciled
**Task range:** exp5564-exp5577
**Execution file:** `research-roadmap-next.yaml`
**Supersedes:** the completed 2026.07.503 roadmap (exp5550-exp5563)

## Executive decision

Milestone `.503` closed three attractive but non-working continuations and exposed a more useful path.
Grammar-row completion is not ready, the local-SOTA hard/soft panel therefore remains blocked, and the
cross-family CSL v2 result was a tautological null. Those lanes must not receive another version-number
rerun. In contrast, `.503` established two clean substrates worth extending: an exact ASP/FSM fixture with
a bounded sparse-repair signal, and causal write-manage-read memory with a real action-impact delta.

Milestone `.504` turns those substrates into a longitudinal program:

1. measure whether local SOTA models can verify controlled near-misses better than they can solve them,
   while keeping the exact ASP/FSM validator authoritative;
2. let the verifier and the energy/memory policy co-evolve under exact feedback, including a real
   spline-local parameter update and delayed-regression gate; and
3. test whether better candidate diversity reaches the live ARC path, while separately honoring the
   operator-reserved PTRM generator build and producing matched sampler-quality hardware evidence.

This is a 14-experiment, four-phase milestone. It includes two infrastructure/evidence slots, one
execution-time SOTA-ingestion slot, a dedicated continuous self-learning phase, one ordinary ARC live
level-up slot, and one separate `arc-trm-generator` slot.

## What milestone 2026.07.503 proved

| Evidence | Result | `.504` consequence |
|---|---|---|
| Exp5552/5553/5554 grammar row path | Automaton row completion was blocked at 0.333 support; the GBNF smoke and panel v4 were gate-skipped. | Retire the row-completion continuation. Use independent exact ASP/FSM instances and ordinary structured responses; do not depend on the missing grammar backend. |
| Exp5555 exact ASP/FSM fixture | Defaults, contradictions, and soft-preference fixtures were exact and clean. | Promote the fixture into a controlled valid/near-miss solve-versus-verify corpus. |
| Exp5556 sparse ASP/FSM repair | Bounded positive repair evidence landed; no speedup claim. | Reuse its descriptor path for verifier stress and matched CPU/CUDA sampling, not another scale-only rerun. |
| Exp5557 CSL corrigendum | The five-arm tautology was repaired; aligned memory exceeded shuffled and no-memory controls. | The causal substrate is usable, but still does not constitute broad continuous learning. |
| Exp5558 causal memory | Write-manage-read action impact was +0.8333 and policy quality exceeded always-full memory by +0.3333. | Search memory policies longitudinally, with held-out transfer and rollback. |
| Exp5559 cross-model CSL v2 | Flagged/blocked: every arm was 0.1667 and cross-family delta was 0.0. | Do not issue a v3 cross-family rerun. Test within-system reset-free adaptation with exact feedback instead. |
| Exp5560 hardware timing hygiene | Receipt was clean, but no authenticated matched timing pairs existed. | Run a same-seed, same-schedule CPU/CUDA comparison; boards remain continuity receipts unless matched execution is available. |
| Exp5561/5562 ARC | The rotated target precheck was clean, but the live attempt produced no target reproduction and registry delta 0. | Continue the mandatory ARC floor with a mechanism that directly addresses observed strategy collapse, not more budget on the same policy. |
| Exp5563 capstone | Structured-SOTA, broad CSL, hardware speedup, and ARC-delta claims were all false; exact fixtures and causal memory were true. | Narrow the milestone around verifier co-evolution, genuine online adaptation, and candidate generation. |

## The three largest gaps to the PRD vision

### Gap 1 — verification is exact only after structure exists

Carnot can validate an ASP/FSM candidate exactly, but `.503` could not reliably create the required
structured rows. The PRD's verifiable-reasoning vision needs an honest separation between solving,
verification, and exact authority. `.504` therefore asks a falsifiable question: on the same controlled
instances, how often can the mandated local SOTA models identify a near-miss they could not themselves
solve? Criteria decomposition and repeated scoring are diagnostics; the exact ASP/FSM validator remains
the oracle.

### Gap 2 — continuous self-learning is still memory-only and short-horizon

The clean `.503` result changed memory actions but did not update the energy landscape, did not persist
through multiple sessions, and did not prove backward retention. FR-11 requires more. `.504` adds a bounded
SelfMem-style policy tournament, a spline-local KAN energy update with replay and rollback, a reset-free
local-SOTA harness, and a delayed-regression promotion gate. No frontier teacher or cloud labeler is used.

### Gap 3 — global candidate generation still does not reach new live states

The live ARC registry did not move. The first real SGE run exposed strategy collapse into repetitive
"wait" proposals, while four deterministic TRM pilots failed to generalize. `.504` addresses those exact
failure modes with (a) an anti-stagnation strategy-diversity controller that the live E3 path can reach and
(b) the separately reserved PTRM stage: stochastic multi-trajectory recursion, history/intent conditioning,
ACT-style halting, and Carnot-verifier selection. Neither task receives solve credit unless the live agent
self-discovers and the registry reproduces the level.

## 2025–2026 research incorporated

The planning refresh was added to `research-references.md` before this roadmap was designed.

| Source | Action in `.504` |
|---|---|
| SelfMem, arXiv:2607.03726 | Bounded memory-policy tournament with exact-energy promotion and rollback. |
| Continual Harness, arXiv:2605.09998 | Reset-free multi-session harness adaptation, replacing its external teacher with local exact feedback. |
| LLM-as-a-Verifier, arXiv:2607.05391 | Discrete, criteria-decomposed, granular, and repeated local-verifier arms on an exact corpus. |
| Verification Horizon, arXiv:2606.26300 | Generator-stratified scalability, faithfulness, and robustness audit plus an explicit co-evolution trigger. |
| Ultrafast KAN online learning, arXiv:2602.02056 | Active-spline-only online energy update with bounded update cost. |
| PTRM, arXiv:2605.19943; Loop, Think, & Generalize, arXiv:2604.07822 | Stochastic trajectories, verifier selection, history conditioning, ACT halting, and overthinking curves in the reserved TRM slot. |
| SGE, arXiv:2603.02045 | Anti-collapse diversity forcing in the already-real local strategy proposer. |
| p-bit FPGA and thermodynamic scaling work | Matched sample-quality and autocorrelation receipts before any hardware speedup claim. |

Semantic Scholar citation trails for EBT (`2507.02092`) and ARM-EBM (`2512.15605`) were checked; the
actionable cited work was already indexed. Extropic's public writing still exposes no authenticated TSU
execution path, and Logical Intelligence's Kona/Aleph material remains architecture context rather than a
local benchmark dependency.

## Target architecture

```text
                      EXACT AUTHORITY / FEEDBACK
                 +--------------------------------+
                 | ASP/FSM validator + energy     |
                 | valid rows + controlled misses |
                 +---------------+----------------+
                                 |
                                 v
+----------------------+   +-----+------------------+   +----------------------+
| Local SOTA GGUF      |-->| Solve-vs-verify panel |-->| Verifier co-evolution|
| Qwen3.6-35B-A3B      |   | criteria/repeat arms  |   | trigger by generator |
| Gemma-4-31B / 26B-A4B|   | LLM is not the oracle|   | family + difficulty  |
+----------------------+   +-----------+------------+   +----------+-----------+
                                    |                           |
                                    v                           v
                         +----------+-----------------------------+
                         | Continuous self-learning controller    |
                         | memory-policy tournament                |
                         | active-spline KAN update + replay       |
                         | reset-free sessions + rollback          |
                         +----------------+-------------------------+
                                          |
                          +---------------+----------------+
                          |                                |
                          v                                v
              +-----------+-------------+      +-----------+-------------+
              | Live ARC candidate path |      | Sampling backends       |
              | SGE anti-stagnation      |      | matched CPU / CUDA      |
              | PTRM reserved generator |      | board continuity receipt|
              | E3 -> reproduce -> bank |      | no unmatched speedup    |
              +-------------------------+      +-------------------------+
```

The exact validator provides labels and promotion feedback. It does not generate a solution, and an LLM
judge is never treated as an oracle. The continuous controller may alter memory operations and KAN spline
parameters, but promotion requires held-out improvement, backward-retention bounds, and successful
rollback. ARC credit remains on the live agent's own attempt stream plus runtime reverse engineering.

## Phase 0 — evidence lock and source delta (Exp5564–5565)

**Exp5564 — transition `.503` into `.504`.** Aggregate the 14 completed artifacts, preserve blocks and
flags, and lock the new task range and gates. This is an evidence receipt, not a re-analysis of failed
lanes.

**Exp5565 — execution-time source delta.** Repeat the mandated source sweep at execution time and append
only non-duplicate actionable deltas. No new source is allowed to silently reopen retired grammar or
cross-family CSL scopes.

## Phase 1 — verifier asymmetry and co-evolution (Exp5566–5568)

**Exp5566 — exact ASP/FSM near-miss corpus.** Build at least 120 rows across defaults, contradictions,
soft preferences, and transition consistency. Pair valid solutions with one- and two-edit corruptions,
label them with the exact validator, and prove mutation-distance and class-balance controls.

**Exp5567 — local-SOTA solve-versus-verify panel.** Behind the corpus gate, run at least 36 exact-labeled
instances on Qwen3.6-35B-A3B and one Gemma-4 flagship GGUF. Compare direct solving with discrete,
criteria-decomposed, granular-score, and repeated-verification arms. Use McNemar/paired bootstrap intervals;
do not claim sub-percent effects. Exact validation decides correctness.

**Exp5568 — verifier co-evolution trigger.** Reuse cached outputs only. Stratify residuals by generator,
constraint family, corruption distance, and verifier arm; assess faithfulness, robustness, and scaling. Emit
a machine-readable trigger rather than silently retuning a threshold.

## Phase 2 — reset-free continuous self-learning (Exp5569–5572)

**Exp5569 — causal memory-policy tournament.** Extend Exp5558 with bounded write/manage/read/forget
policies over a multi-session stream. Compare no memory, shuffled memory, static causal memory, always-full,
and self-optimized causal memory. The policy may change; model weights remain frozen.

**Exp5570 — spline-local KAN energy update.** Use exact ASP/FSM feedback to update only activated spline
coefficients online, with replay and checkpoint rollback. Compare static KAN, dense-update KAN, and
active-spline KAN. This is the milestone's first genuine energy-parameter self-learning test.

**Exp5571 — reset-free local-SOTA harness.** If the memory and KAN gates pass, run consecutive constraint
families without resetting the harness. Compare reset-free adaptation to reset-each-session and shuffled
feedback using at least Qwen3.6-35B-A3B. The local model remains frozen; only the governed harness and
energy calibrator adapt.

**Exp5572 — delayed-regression promotion gate.** Re-evaluate cached checkpoints after an intervening
session, inject stale/contradictory memory, verify rollback, and decide whether a continuous-self-learning
claim is allowed. This task makes no new model calls.

## Phase 3 — hardware and live generation (Exp5573–5577)

**Exp5573 — matched sampler-quality and hardware continuity.** Run identical seeds, schedules, sample
counts, and ASP/FSM-derived Ising instances on CPU and CUDA. Measure energy distribution, best energy,
autocorrelation/effective sample size, wall time, device identity, and warm-up. Board lanes report current
KV260, PolarFire, and GateMate reachability without repeating prohibited destructive media access. A
speedup claim is impossible without matched successful pairs.

**Exp5574 — reserved PTRM generator stage.** Implement and train the first faithful Stage-1 substrate:
Gaussian noise at each recursion step, multiple trajectories, history/intent conditioning, dynamic halting,
overthinking curves, and Carnot-verifier selection. Run a bounded positive control and emit a checkpoint
and held-out protocol. This is `track: arc-trm-generator`; it does not consume the ordinary ARC floor and
does not claim a hidden-game solve.

**Exp5575 — SGE anti-stagnation live-path precheck.** Add a strategy-collapse detector and diversity-forcing
portfolio to the real `LLMStrategyProposer`/`SGECandidateRouter`, prove E3 reachability with fake-completer
tests, and select a registry-valid unsolved target. This directly addresses the observed passive-wait
collapse rather than increasing budget.

**Exp5576 — gated live ARC level-up.** Run the anti-stagnation portfolio with a mandated local SOTA GGUF on
the prechecked target. Only `live_agent_self_discovery` from its own attempts/runtime RE may receive credit;
offline reproduction and registry delta are mandatory. The same null retires this SGE continuation.

**Exp5577 — capstone reconciliation.** Read every artifact, apply every gate, separate clean, bounded,
blocked, flagged, and skipped evidence, and update private specs/traceability/ops docs. Broad claims require
their own explicit evidence fields.

## Dependency graph and gates

```text
Exp5564 transition ------------------------------+
Exp5565 source delta ----------------------------+-------------------------+
                                                                           |
Exp5566 exact corpus --[corpus_ready]--> Exp5567 SOTA panel                |
                                      --[panel_complete]--> Exp5568 trigger|
                                                                           |
Exp5569 memory tournament --[policy_ready]--+                               |
                                            +--> Exp5571 reset-free SOTA ---+
Exp5570 KAN update --------[kan_ready]-------+             |                 |
                                                          +--> Exp5572 -----+
                                                                           |
Exp5573 matched hardware --------------------------------------------------+
Exp5574 PTRM reserved slot ------------------------------------------------+
                                                                           |
Exp5575 SGE precheck --[live_path_ready AND target_unsolved]--> Exp5576 ---+
                                                                           |
All terminal/blocked/skipped artifacts --------------------------> Exp5577
```

Structured gates in `research-roadmap-next.yaml` are conjunctive. A failed prerequisite skips the
downstream agent call; the downstream script must not fabricate a result. Gates test deliverable readiness,
not favorable scientific outcomes.

## Hardware requirements

| Experiments | Required resources | Expected wall time | Failure behavior |
|---|---|---:|---|
| 5564–5566, 5568–5570, 5572, 5575, 5577 | CPU, existing artifacts, local tests | 20–60 min each | Emit a bounded/blocked receipt if required artifacts are absent. |
| 5567 | RTX 3090-class CUDA, cached Qwen3.6-35B-A3B plus one Gemma-4 flagship GGUF, llama.cpp GPU offload | 90–150 min | `blocked_missing_sota_cache` or `blocked_no_cuda_offload`; no CPU fallback headline. |
| 5571 | RTX 3090-class CUDA, cached mandated SOTA GGUF, exact ASP/FSM feedback | 90–180 min | Block if live local inference/offload is not authenticated. |
| 5573 | CPU plus CUDA GPU; optional authenticated SSH to KV260/PolarFire and physical GateMate visibility | 60–120 min | Preserve independent lane receipts. Never access KV260 `mmcblk`; never claim board speedup from unmatched runs. |
| 5574 | RTX 3090-class CUDA, PyTorch training, human-win trajectory corpus | 3–8 h | Write the precondition receipt first; block rather than shrinking into another toy deterministic pilot. |
| 5576 | RTX 3090-class CUDA on a fresh non-default llama-server port, offline ARC kit, live E3 path | 60–180 min | Block on missing target/runtime; null is valid and retires this continuation if unchanged. |

### Current attached-board constraints

- **KV260:** use SSH-only probes; never touch `mmcblk` or perform destructive boot-media work.
- **PolarFire SoC:** workload reachability exists, but no matched end-to-end timing pair exists yet.
- **GateMate:** physical/JTAG setup remains the blocker; do not repeat a software-only probe as evidence of
  progress unless physical visibility changed.
- **Extropic TSU:** no local authenticated device or API is available; it remains watch-only.

## Claim policy

- No grammar-forced structured-SOTA claim; the `.503` row path is closed.
- No LLM-verifier-as-oracle claim; exact ASP/FSM validation is authoritative.
- No broad continuous-self-learning claim without Exp5572 passing forward adaptation, backward retention,
  delayed regression, poisoning resistance, and rollback.
- No hardware speedup claim without matched successful runs and authenticated device receipts.
- No ARC solve claim without `solve_provenance: live_agent_self_discovery`, offline reproduction, and a
  positive registry delta.
- The PTRM slot is a training-stage experiment and cannot count as the ordinary ARC level-up slot.

## Explicitly deferred or retired

- Automaton/GBNF row-completion vN+1 and hard/soft panel v5.
- Cross-family CSL transfer v3 on the Exp5559 substrate.
- Offline ground-truth BFS, per-game calibration adapters, or any ARC solver unreachable from E3.
- Another deterministic fixed-depth TRM action-classification or sequence-refinement pilot.
- Hardware speedup from receipt-only board probes or unmatched timing.
- Kona/Aleph or Extropic comparative claims without local authenticated execution.

## Exit criteria

The milestone is complete when all 14 tasks have terminal artifacts and Exp5577 has reconciled them. A
scientific null or a correctly skipped gate is terminal. The milestone advances Carnot only if it leaves
behind at least one of: a trustworthy solve-versus-verify asymmetry measurement, a promotable online energy
update, a reset-free non-regressing harness, a faithful PTRM Stage-1 substrate, a live ARC registry delta,
or a matched hardware sampler-quality receipt. Otherwise the capstone must narrow or retire the associated
lane rather than renaming it for `.505`.
