# Research Roadmap vNEXT — Milestone 2026.07.505

**Title:** Parser-Grounded Verifier Co-evolution, Two-Timescale Self-Learning, PTRM Adjudication, Epistemic ARC, and Sampling Crossover  
**Status:** Proposed  
**Task range:** Exp5578–Exp5591  
**Execution manifest:** `research-roadmap-next.yaml`

## Milestone thesis

Milestone `.504` established useful substrates, but it did not establish its two central
headline claims. The local-SOTA panel executed on both mandated model families with
authenticated GPU offload, yet all 648 candidate parses failed; solve and verification
accuracy therefore both collapsed to zero. The causal memory tournament completed, but
its two headline deltas were arithmetically identical and was adversarially flagged as a
tautology. By contrast, active-spline KAN learning and matched CPU/CUDA sampler quality
were clean, PTRM Stage 1 created a faithful stochastic substrate, and the capstone correctly
refused solve/verify, broad self-learning, hardware-speedup, and ARC-level-up claims.

`.505` repairs those measurement foundations before extending them. It then asks five
bounded questions:

1. Does a positively controlled parser support a real local-SOTA solve-versus-verify
   asymmetry measurement?
2. Can exact counterexamples drive safe verifier extension rather than another external
   text-scoring rerun?
3. Can corrected memory evidence and active-spline KAN updates form a two-timescale,
   reset-free learner that passes delayed-regression and rollback gates?
4. Does the existing PTRM checkpoint show preregistered leave-one-game-out benefit, and
   can a separately ordinary live ARC track improve candidate discovery with an epistemic
   object-model MCTS mechanism?
5. At what problem size, if any, does matched CUDA sampling cross over CPU while preserving
   sample quality?

## What milestone `.504` proved

| Evidence | Terminal fact | Consequence for `.505` |
|---|---|---|
| Exp5566 exact ASP/FSM corpus | 120 exact-labeled rows were clean. | Keep the corpus and oracle fixed; do not regenerate the benchmark. |
| Exp5567 local-SOTA panel | Qwen3.6 and Gemma-4-26B-A4B ran with authenticated offload, but `parser_failure_count=648`; all accuracy cells were zero. | Treat the result as a parser/instrumentation failure. Repair, positively control, then remeasure once. |
| Exp5568 co-evolution audit | The trigger fired on a worst-family false-accept rate of 1.0, but the upstream parser collapse made the threshold non-transferable. | Recompute only from clean remeasurement and extend exact predicates, not an LLM judge. |
| Exp5569 memory tournament | `policy_ready=true`, but the artifact was flagged because forward and backward deltas were identically `0.3333333334`. | Issue a row-level metric corrigendum before the memory policy can gate learning. |
| Exp5570 spline-local KAN | Active-spline exact-energy learning was clean and `kan_ready=true`. | Reuse the shipped updater; do not retrain an unrelated KAN. |
| Exp5571/5572 reset-free lane | Live harness blocked on no authenticated CUDA offload; promotion was gate-skipped. | Use a fresh native CUDA llama-server receipt and then run the cached delayed gate. |
| Exp5573 sampling | Six quality-matched pairs passed; CUDA was roughly four times slower at n=32/64. | Search for a crossover at larger n; no current speedup claim is allowed. |
| Exp5574 PTRM | Stochastic trajectories, history/intent, and dynamic halting worked, but LOO was not run and depth increased overthinking. | Use the checkpoint for decision-grade LOO evaluation; do not retrain Stage 1. |
| Exp5575/5576 ordinary ARC | The SGE controller was reachable, but global pre-existing gates failed and the live task skipped; capstone retired the continuation. | Do not rerun SGE. Use the known-untried epistemic object-model MCTS path. |
| Exp5577 capstone | 14/14 artifacts reconciled; solve/verify, broad CSL, speedup, and ordinary ARC claims remained false. | Carry these boundaries forward verbatim and require explicit promotion evidence. |

## The three largest gaps to the PRD vision

### Gap 1 — verification evidence is not yet measurement-valid

Carnot has an exact ASP/FSM oracle, but `.504` never obtained parseable local-SOTA
candidates. The PRD requires evidence that verifier feedback is useful on model-produced
work, not merely that an exact checker exists. `.505` first establishes deterministic
parser fixtures and mutation-based positive controls, then remeasures two mandated local
GGUF families. Only clean residuals may drive a counterexample-guided exact verifier
extension.

### Gap 2 — continuous self-learning is not yet longitudinally promotable

The KAN updater is clean, but the memory policy was flagged and live reset-free execution
never began. FR-11 needs persistent adaptation with safety and non-forgetting, not a
one-session component score. `.505` follows PACE with two risk levels: low-risk causal
memory-policy changes first, then higher-risk active-spline energy changes only after the
first timescale saturates. EvoPolicyGym-style fixed budgets and decision ledgers make every
accepted or rejected update attributable. A fresh native CUDA path and delayed adversarial
replay decide whether the controller is promotable.

### Gap 3 — learned candidate generation still has no held-out or live-level gain

PTRM has a usable Stage-1 checkpoint but no leave-one-game-out verdict. The ordinary ARC
registry also stayed flat, and SGE is retired. `.505` separates the obligations: PTRM gets
one reserved held-out adjudication slot, while ordinary ARC gets an expressly untried
epistemic object-model MCTS precheck and gated live-agent self-discovery attempt. Neither
offline adapters nor source-aware solvers can receive solve credit.

## 2025–2026 research incorporated

The `V505 Planner Refresh - 20260711` block was added to `research-references.md` before
this design.

| Source | Action in `.505` |
|---|---|
| PACE, arXiv:2605.23019 | Risk-separated memory-policy and active-spline update timescales, with held-out admission and rollback. |
| EvoPolicyGym, arXiv:2607.02440 | Fixed interaction/adaptation budget, immutable per-change decision ledger, and delayed outcome attribution. |
| LLM-as-a-Verifier and Verification Horizon, already indexed | Parser-grounded remeasurement followed by exact-residual stress and co-evolution; no LLM judge authority. |
| PTRM and Loop, Think, & Generalize, already indexed | Decision-grade LOO comparison and overthinking/halting analysis on the existing stochastic checkpoint. |
| Epistemic object-model planning line recorded in `ops/known-issues.md` | Uncertainty-aware MCTS over object-centric runtime rollouts and causal probes, on the live E3 path. |
| p-bit/Ising accelerator literature, already indexed | Quality-matched CPU/CUDA crossover sweep before any acceleration claim. |

The EBT (`2507.02092`) and ARM-EBM (`2512.15605`) citation routes yielded no stronger
new dependency. Extropic exposes no authenticated local TSU route, and Kona remains
proprietary architecture context rather than local comparative evidence.

## Target architecture

```text
             EXACT ASP/FSM AUTHORITY + CONTROLLED CORPUS
                              |
                    +---------v----------+
                    | parser fixtures +  |
                    | positive controls  |
                    +---------+----------+
                              |
          +-------------------v-------------------+
          | local SOTA solve/verify remeasurement|
          | Qwen3.6 + Gemma-4, llama.cpp CUDA     |
          +-------------------+-------------------+
                              |
                    clean exact residuals
                              |
          +-------------------v-------------------+
          | counterexample-guided exact verifier |
          | predicate extension + held-out replay|
          +-------------------+-------------------+
                              |
     +------------------------v-------------------------+
     | two-timescale continuous self-learning controller|
     |  1. causal memory policy (low risk)               |
     |  2. active-spline KAN energy update (higher risk) |
     | fixed budget -> ledger -> held-out -> rollback     |
     +------------------------+--------------------------+
                              |
                 fresh native CUDA SOTA sessions
                              |
                    +---------v----------+
                    | delayed regression |
                    | poison + rollback  |
                    +--------------------+

  +--------------------------+       +----------------------------+
  | reserved PTRM LOO track  |       | ordinary live ARC track    |
  | existing checkpoint      |       | EOM-MCTS precheck -> live  |
  | no solve credit          |       | reproduce -> registry      |
  +--------------------------+       +----------------------------+

  +---------------------------------------------------------------+
  | matched sampler crossover: CPU <-> CUDA + board continuity    |
  +---------------------------------------------------------------+
```

## Phase 0 — evidence lock and execution freshness (Exp5578–5579)

**Exp5578 — `.504` to `.505` transition.** Aggregate the 14 terminal artifacts and lock
the parser, memory, live-CUDA, PTRM, ARC, and hardware boundaries. This is the first
infrastructure slot and may not reinterpret the parser collapse as a scientific result.

**Exp5579 — execution-time source delta.** Recheck all required source surfaces after the
planner marker and append only non-duplicate actionable evidence. This is the milestone's
bleeding-edge ingestion slot; an honest no-op is acceptable.

## Phase 1 — parser-grounded verification (Exp5580–5582)

**Exp5580 — parser forensics and positive control.** Replay cached Exp5567 responses,
classify every failure mode, introduce a deterministic parser cascade, and prove it on
synthetic valid/malformed fixtures plus hand-checked cached samples. It makes no model
calls and cannot claim improved model quality.

**Exp5581 — clean local-SOTA remeasurement.** Behind parser gates, rerun the fixed 36-row
panel using `unsloth/Qwen3.6-35B-A3B-GGUF` and at least one of
`unsloth/gemma-4-31B-it-GGUF` or `unsloth/gemma-4-26B-A4B-it-GGUF`. Use cached model paths,
native llama.cpp GPU offload, exact validation, paired uncertainty, and an explicit parser
failure ceiling. This is the only remeasurement attempt; a repeated parser collapse retires
the scope.

**Exp5582 — counterexample-guided exact verifier extension.** Reuse only clean cached
residuals. Mine bounded, human-auditable exact predicate candidates, train/select on one
split, and require zero unsafe false accepts on held-out families. This extends exact
constraint coverage; it is not PHASE D, an EBRM/uPRM, or an external generated-text scorer.

## Phase 2 — two-timescale continuous self-learning (Exp5583–5586)

**Exp5583 — causal-memory metric corrigendum.** Recompute independent forward transfer,
backward retention, forgetting, and policy-cost metrics from row-level evidence. Add a
permutation control proving the metrics are not algebraic aliases. The policy is usable
only if the tautology disappears and causal controls still pass.

**Exp5584 — two-timescale exact-gated controller.** Combine the corrected memory policy
with the shipped active-spline KAN updater. Use a fixed adaptation budget and immutable
decision ledger. Admit low-risk memory changes first; unlock spline updates only after
preregistered memory-policy saturation; validate every change on held-out exact fixtures;
and prove checkpoint rollback. This is the milestone's required continuous self-learning
experiment.

**Exp5585 — reset-free live local-SOTA sessions.** Behind controller readiness, start a
fresh native CUDA llama-server on unique ports and authenticate real offload before any
session. Compare reset-free, reset-each-session, and shuffled-feedback arms over consecutive
constraint families using a mandated flagship GGUF. Model weights remain frozen.

**Exp5586 — delayed promotion and poisoning gate.** Reuse cached checkpoints after an
intervening session, inject stale/contradictory memory and exact-label corruption, test
rollback, and reconcile forward adaptation with backward retention. It makes no new model
calls and alone decides the broad continuous-self-learning claim.

## Phase 3 — generator adjudication, live ARC, and hardware (Exp5587–5590)

**Exp5587 — reserved PTRM leave-one-game-out adjudication.** Use the Exp5574 checkpoint
and preregistered protocol across multiple held-out games and seeds. Compare stochastic
PTRM, non-recursive, deterministic-recursive, and majority-selection controls; measure
per-action accuracy, calibration, trajectory diversity, compute, halting, and overthinking.
Reach the held-out verdict and retire the PTRM generator line if it again has no signal.
This `track: arc-trm-generator` slot does not count as ordinary ARC.

**Exp5588 — epistemic object-model MCTS live precheck.** Implement the known-untried
object-centric runtime model, epistemic uncertainty score, causal probe bank, and MCTS
selection behind the real E3 router. Registry-precheck an unsolved level and prove the live
agent can reach the mechanism without game source, exhaustive BFS, or a hand GameAdapter.

**Exp5589 — gated ordinary ARC level-up.** Run the prechecked mechanism on a fresh
llama.cpp port using a mandated local SOTA GGUF. Only the agent's own attempts and runtime
reverse engineering count. Offline reproduction and a positive registry delta are required;
a repeated null retires this EOM-MCTS continuation.

**Exp5590 — matched CPU/CUDA crossover and board continuity.** Sweep n=128, 256, 512,
and 1024 with identical seeds, schedules, sample counts, warm-up rules, and quality metrics.
Estimate the first quality-preserving crossover, if any. Use SSH-only KV260 checks, a real
PolarFire workload receipt, and no GateMate detect rerun unless a new physical/JTAG change
is documented.

## Phase 4 — capstone and reconciliation (Exp5591)

**Exp5591 — `.505` capstone.** Read all 14 artifacts, apply structured gates, separate
clean/bounded/blocked/flagged/skipped evidence, reconcile private specs and ops documents,
and emit explicit claim booleans. This is the second infrastructure slot.

## Dependency graph

```text
Exp5578 transition ------------------------------------------------------+
Exp5579 source delta ---------------------------------------------------+
                                                                         |
Exp5580 parser repair --[ready AND positive_control>=0.95]--> Exp5581    |
Exp5581 remeasure --[complete AND parser_failure_rate<=0.05]--> Exp5582  |
                                                                         |
Exp5583 memory corrigendum --[clean AND policy_ready]--> Exp5584         |
Exp5584 two-timescale --[controller_ready]-------------> Exp5585         |
Exp5585 live sessions --[reset_free_candidate]---------> Exp5586         |
                                                                         |
Exp5587 PTRM LOO --------------------------------------------------------+
                                                                         |
Exp5588 EOM-MCTS precheck --[live_path_ready AND target_unsolved]        |
                                           +-------------> Exp5589 ------+
                                                                         |
Exp5590 sampler crossover -----------------------------------------------+
                                                                         |
All terminal/blocked/skipped artifacts -----------------------> Exp5591
```

All structured gates are conjunctive. A failed gate skips the downstream agent call and
preserves the upstream scientific result. A task must never fabricate a favorable field to
make a downstream task execute.

## Hardware requirements

| Experiments | Required resources | Expected wall time | Failure behavior |
|---|---|---:|---|
| 5578–5580, 5582–5584, 5586, 5591 | CPU, existing artifacts, local tests | 20–90 min | Emit bounded/blocked evidence if prerequisites are absent. |
| 5581 | Dual RTX 3090-class CUDA; cached Qwen3.6 and Gemma-4 GGUFs; llama.cpp | 2–4 h | Block on missing cache or unauthenticated offload; do not use a legacy model for headline cells. |
| 5585 | RTX 3090-class CUDA; fresh native llama-server; exact feedback | 2–4 h | Block before inference if native CUDA/offload receipt fails. |
| 5587 | RTX 3090-class CUDA; Exp5574 checkpoint; PyTorch | 3–8 h | Preserve the preregistered LOO denominator; do not shrink into a smoke test. |
| 5588 | CPU tests and live E3 routing fixtures | 1–3 h | Block if E3 reachability, registry precheck, or no-leak controls fail. |
| 5589 | RTX 3090-class CUDA; mandated local SOTA GGUF; offline ARC kit | 1–4 h | Gate-skip on failed precheck; otherwise honest null is valid and retirement-bearing. |
| 5590 | CPU plus RTX 3090 CUDA; optional authenticated board access | 2–6 h | Keep backend lanes independent; no speedup from failed or unmatched rows. |

### Attached-board constraints

- **KV260:** SSH-only. Never inspect or write `/dev/mmcblk*`; a timeout is a valid reachability receipt.
- **PolarFire SoC:** use an authenticated workload/hash receipt, not reachability alone.
- **GateMate:** do not repeat the unchanged software detect path. Probe only after a documented physical,
  cable, power, or JTAG change; otherwise emit a continuity note.
- **Extropic TSU:** no authenticated local device/API exists, so it remains watch-only.

## Claim and retirement policy

- No solve-versus-verify claim unless parser positive controls pass and production parser failures stay at
  or below 5% on both mandated model families.
- No verifier co-evolution claim from parser failures or an LLM judge; only exact-label residuals and
  held-out unsafe-false-accept evidence count.
- No broad continuous-self-learning claim unless Exp5586 passes forward adaptation, backward retention,
  delayed regression, poisoning resistance, budget accounting, and rollback with frozen LLM weights.
- No PTRM continuation after another preregistered held-out null; the reserved slot must reach a verdict.
- No ARC solve claim without `solve_provenance: live_agent_self_discovery`, offline reproduction, and
  positive registry delta. PTRM does not satisfy the ordinary ARC floor.
- No hardware speedup claim without successful quality-matched pairs and authenticated device timing.
- Same-verdict reruns named in `prior_failures` are retirement-bearing through
  `retire_if_same_verdict: true`.

## Explicitly closed or deferred

- PHASE D external generated-text energy scorers, EBRM/uPRM/ARM-EBM reruns, and broad LoRA/RL reward-model
  training remain retired.
- Grammar/automaton row-completion vN+1 work and cross-family CSL v3 remain retired.
- SGE anti-stagnation and more-budget continuations are retired after `.504`.
- Offline ground-truth BFS, game-source inspection, hand GameAdapters, and duplicate ARC level solves are
  prohibited as live solve evidence.
- Proprietary TSU/Kona comparative claims are deferred until authenticated local artifacts exist.
- No active `research-roadmap.yaml` or conductor source modification is part of this proposal.
