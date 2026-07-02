# Research Roadmap vNEXT: V474 Oracle-Distinct Verifier Scale-Up, Trajectory-Enumeration Wall, and Phase D Retirement

**Milestone:** `2026.07.474`
**Status:** Pre-staged next milestone
**Prepared:** 2026-07-02
**Predecessor:** `2026.07.473`
**Execution manifest:** `research-roadmap-next.yaml`

## Executive Summary

Milestone `2026.07.473` landed one genuine breakthrough (the ARC oracle-distinct Set-Encoder-vs-vote
win survived corrected cross-corpus replication and the DiffusionGemma scale-up gate flipped to
`ungate_now`) inside a batch of otherwise-null/blocked results (two deepen-wall warm-start nulls, a
gate-blocked live level-up attempt, a flagged GAP-4 pilot). V474 makes three decisive moves instead of
re-running the same levers:

1. **Cash in the breakthrough.** Harden the Set-Encoder win past its `n=24` sample-size floor (below the
   CLT `n>=30` minimum CLAUDE.md mandates for any percentage-point delta claim), then activate the
   DiffusionGemma energy-guided-diffusion pilot the gate now permits — the real experiment the whole
   `.379-.473` verifier-moat arc has been building toward.
2. **Attack the diagnosed wall, not another symptom of it.** `ops/verifier_gaps.md` GAP-4891's Stage-2
   probe (`.452`) delivered a decisive negative: goal-energy correctly SEPARATES win from near-win on 3/4
   tested games, but ordering the search frontier by it does not help — the winning trajectory is never
   ENUMERATED. A purpose-built countermeasure (`arc_relational_mask_pruner.py`, branching-factor reduction
   via a learned change-location prior) was built and unit-tested on 2026-06-28 but has **never been run**
   against the stalled games. V474 runs that experiment before proposing anything new.
3. **Retire what seven milestones of evidence already answered.** PHASE D (the off-ARC distributional
   energy verifier moat: LoRA-EBM, uPRM x3, EBRM x3, plus `.473`'s MMLU-Pro continuation) is a
   consistently-replicated null across every construction tried. The `2026-06-30` MANDATORY priority's own
   retirement condition (`retire_if_same_verdict: true`) is met. V474 formally retires the specific
   constructions and redirects the freed majority toward the two live threads above — while keeping the
   door open for a genuinely different mechanism (activation-steering, per fresh literature) rather than a
   fourth external-scorer construction.

V474 also reconciles a stale gap: `GAP-LIVE-INTEGRATION`'s cited evidence (`arc_strategy_router`/
`arc_world_model_dsl` unimported, `target_levels=1`, `value_weight=0.0`) is **no longer true** —
`exp4605`/`exp4652` already fixed the wiring and target-levels, and live-tested the value head to an
honest zero-lift null. Re-proposing that fix would be a doomed no-op; V474 reconciles the gap doc instead
of blindly re-attacking it.

## What V473 Proved

V473 closed with capstone verdict:

`complete: v473 reconciled with DiffusionGemma ungated for future scaling, GAP-4 scale-up not filled,
zero new ARC levels banked, and exp5161 excluded as flagged_adversarial.`

Key results (all file paths under `results/`):

- **`exp5160` (real win, oracle-distinct, cross-corpus replicated).** The exp4245 Set-Encoder-vs-vote ARC
  selection win survives replication on a disjoint ARC-GEN candidate pool after resolving a `game_id`
  terminology mismatch that had wrongly disqualified the original cross-game test. `cross_corpus_delta:
  0.5` (oracle_at_k=0.75, set_encoder_at_1=0.75, vote_at_1=0.25), identical across 5 seeds,
  `second_pool_leak_audit_passed: true`, `verifier_is_oracle: false`, `adversarial_verify` clean (0
  flags). `diffusiongemma_gate_updated_recommendation: ungate_now`. **Caveat carried into V474:**
  `held_out_task_n: 24` is below CLAUDE.md's CLT floor (`N>=30` for any percentage-point delta claim) —
  the delta is real and the mechanical linter is clean, but the sample size itself does not yet clear the
  project's own rigor bar. This is V474 Phase A's first task.
- **`exp5161` (GAP-4 pilot, flagged).** `honest_verdict:
  complete_gap4_pilot_n60_direction_replicated_not_significant_scale_up_recommended`, but
  `flagged_adversarial: true` with a CRITICAL `DURATION_TOO_SHORT` flag (`duration_s=5.59s` against a
  60s compute-bound floor). Diagnosed root cause (see Phase C below): several methodology fields
  (`inference_substrate`, `model_specs`, `random_seed`, `reproducibility_checksum`) were wrapped in
  `{principle, value}` dicts instead of left bare, which breaks `adversarial_verify.py`'s substrate
  recognition and silently falls back to the strictest (`live_llm_inference`, 60s floor) check even when
  a cheaper substrate genuinely applied. Excluded from headline aggregation per the fabrication gate.
- **`exp5157`/`exp5158` (deepen-wall warm-start, both honest nulls).** ReDRAW-style residual warm-start:
  `warmstart_replay_ablation_gate_failed_honest_null_delta_0.0`. DynaMITE-RL-style goal-energy-ranker
  carryover: `goal_energy_ranker_warmstart_gate_failed_improved_1_of_3`. `exp5159` (the gated live
  level-up attempt) consequently reported `blocked_gate_check_failed` — zero levels banked
  (`reproducible_total_levels` stays flat at 69/24 games, unchanged for 2+ consecutive milestones).
- **`exp5165` (retirement hygiene).** Formally retired the generation-axis exploration-signal scope
  (novelty bonus / program-synthesis filter / energy-as-fitness QD — `exp4688`/`exp4689`/`exp5154`, three
  independent nulls) into `ops/exclusion_manifest.yaml` with `blocked_patterns` scoped precisely enough
  to NOT catch the deepen-wall or representation-fix work. This is the template V474 Phase 0 follows for
  the larger PHASE D retirement.
- **`exp5156` (transition task, flagged).** `flagged_adversarial: true` but severity `warn`, not critical
  — `adversarial_verify.py`'s `qd-random-mutation-ablation-omitted` check fired because the transition
  artifact **cites/reports** `exp5154`'s retired QD finding, not because it makes a fresh QD claim. A
  false positive on citation, distinct from `exp5161`'s genuine methodology gap. V474 Phase 0 scopes the
  check to artifacts making their OWN claim.
- **`exp5166` (hardware continuity).** KV260 and PolarFire both `hardware_smoke`-passed
  (`no_speedup_claim: true`); GateMate `blocked_gatemate_dirtyjtag_idcode` (`openFPGALoader --detect`
  could not read IDCODE `0x20000001`). `boards_reachable_count: 2/3`.
- **`exp5164` (retro timing false-zero fix).** Standalone module works and is unit-tested but is
  **explicitly not wired into `scripts/research_conductor.py`** — correctly so, since experiment task
  prompts are barred from touching that file. This is an outer-loop/operator wiring item, not a V474
  roadmap task (see Non-Goals).

## Three Biggest Gaps Against The PRD

1. **FR-12 (verifiable reasoning) has an unproven oracle-distinct generalization story.** The FoVer
   headline (AUROC 0.9131, G1-G4 all met) proves the verifier ensemble discriminates well on ONE corpus
   under CPU verifier-scoring. The `.473` Set-Encoder win is the first ARC-domain, cross-corpus-replicated,
   oracle-distinct positive — but at n=24 it doesn't yet meet the project's own statistical rigor bar, and
   the actual DiffusionGemma guidance experiment the gate exists to unlock has never been run. Closing this
   gap (Phase A) is the most direct move toward "verifiable reasoning that generalizes," not just
   "verifiable reasoning on a frozen benchmark."
2. **FR-11 (autonomous self-learning) has strong online-learning infrastructure that has never been
   pointed at the project's own hardest open problem.** `arc_relational_mask_pruner.py` is, by
   construction, a Tier-1/Tier-2 online constraint learner (research-program.md's framework: pure-CPU
   counter-style updates from the search's own observed transitions, no offline training corpus) — but it
   has sat unit-tested-and-unexercised since `2026-06-28`. Phase B closes this gap by actually running it
   against the diagnosed enumeration wall, which is simultaneously the self-learning experiment CLAUDE.md
   mandates every milestone and the most concrete lever `ops/verifier_gaps.md` currently has open.
3. **Research-process discipline (CLAUDE.md's Failed-Experiment Rerun / Depth-over-breadth ethos) is
   being violated by inertia, not by design.** PHASE D has run for seven milestones (`.461-.467`, `.473`)
   with a saturated null on every construction, and `GAP-LIVE-INTEGRATION` is being carried in
   `ops/verifier_gaps.md` as `status: open` when the code it describes was fixed two milestones' worth of
   work ago (`exp4605`, `exp4652`). Both are scope-reduction / documentation-hygiene gaps, not research
   gaps — closing them (Phase 0) is what frees real capacity for Phases A and B instead of adding a
   thirteenth task that just re-measures what is already known.

## Literature Incorporated For V474

Full scan in the SOTA-ingestion task's citations; headline items surfaced this planning session
(arXiv IDs verified live, 2026-07-02):

- **uPRM** (arXiv:2605.10158, EPFL, May 2026) — the sharpest available replication target: +6.9% over
  majority-vote as a test-time verifier, unsupervised, derived from LLM next-token probabilities across a
  batch of trajectories. Carnot's three uPRM replication attempts (`exp5004`/`exp5018`/`exp5032`) all
  tested on MuSR specifically; whether MuSR is inside uPRM's own validated domain set is an open question
  the retirement task (Phase 0) must check before writing the retirement's scope note, since a null on an
  out-of-validated-domain replication is a narrower finding than "uPRM doesn't replicate."
- **Distributional EBM for structured LLM reasoning** (arXiv:2605.18871, May 2026) — closest published
  analog to Carnot's own architecture (heterogeneous LoRA-adapter ensemble on one frozen encoder). Its own
  honest failure mode — "struggles where pretraining knowledge dominates (code semantics, narrative
  inference)," i.e. wherever a cheap oracle already exists — directly corroborates Carnot's
  circular-moat discipline (`CLAUDE.md` "Circularity / Oracle-Distinctness Discipline"). Cited in the
  Phase 0 retirement writeup as external corroboration, not re-attempted as a fourth construction.
  Zero-shot transfer note (MuSR-trained scorer hits 93.9% on GSM8K unseen) is worth a footnote but does not
  change the retirement call — Carnot's own replications, not the source paper's numbers, are what's null.
- **VerifySteer** (arXiv:2605.20745, May 2026) — a **genuinely different mechanism** from everything PHASE
  D tried: a hidden-state signal near verification-paragraph boundaries encodes accept/reject strictness;
  steering it is "competitive with self-consistency while requiring 4-7x less inference compute." This
  targets the north star's ALREADY-RELAXED win condition (`ops/north-star.md` §5: "equally effective as the
  LM at lower cost/latency... does NOT need an accuracy edge") rather than the harder beat-SC-on-accuracy
  bar every retired PHASE D construction chased. Not a rerun — activation steering vs. external scoring is
  a different construction class. Phase C runs one cheap, exploratory pilot.
- **KAEM — Kolmogorov-Arnold Energy Models** (arXiv:2506.14167) — already integrated
  (`python/carnot/models/kaem_energy.py`, `KAEMEnergy`, Exp 447). No action needed; confirms the project's
  existing KAN-fast-path tier is aligned with the current literature frontier, not behind it.
  No 2025-2026 paper was found using KAN specifically as a reasoning verifier/reward model — an open gap,
  not a search miss (three search angles tried), noted for a future milestone if the KAN tier's role
  expands.
- **"Explore Before You Solve"** (arXiv:2605.25931, ARC-AGI-3-specific) — proposes AERA
  (explore/verify/plan), reports RHAE=0.30 on the private 55-game set with a 0.5B model, and — load-bearing
  for how V474 should read its own numbers — finds "all 25 public ARC-AGI-3 games are solvable via
  non-intelligent strategies," a benchmark-validity point that corroborates CLAUDE.md's existing framing
  (public-game replays are not the scored deliverable). Gives Carnot a fresh external RHAE comparator
  (0.30 private) against Carnot's own ~0.05-0.08.
- **Energy-guided/constrained decoding survey (this session).** No 2025-2026 paper was found using energy
  DESCENT as the decoding mechanism itself (as opposed to as a scorer) — a null result that is itself
  useful corroboration of Carnot's internal finding (`ops/north-star.md` §5) that the field has converged
  on the generator/verifier split rather than reviving energy-descent generation.

Full per-paper detail (all ten requested source categories) appended to `research-references.md` by
Phase A's SOTA-ingestion task (`exp5172`) rather than duplicated here.

## Architecture

```text
                          PHASE 0: Ledger Hygiene (frees capacity)
                          +----------------------------------------+
                          | exp5168 archive .473 / activate .474   |
                          | exp5169 fix false-positive citation    |
                          |         flag in adversarial_verify.py  |
                          | exp5170 retire PHASE D (LoRA-EBM/      |
                          |         uPRM/EBRM external-scorer      |
                          |         constructions) into exclusion  |
                          |         manifest; VerifySteer excluded |
                          |         from the retirement scope      |
                          +--------------------+---------------------+
                                               |
                +------------------------------+------------------------------+
                |                                                              |
   PHASE A: Cash in the ARC oracle-distinct win           PHASE B: Attack the diagnosed enumeration wall
   +---------------------------------------------+        +----------------------------------------------+
   | exp5171 harden exp5160: N 24->30+ (CLT      |        | exp5174 reconcile GAP-LIVE-INTEGRATION        |
   |          floor), confirm delta survives      |        |          (evidence is stale -- exp4605/4652   |
   |          real variance, not seed-identical   |        |          already fixed wiring+target_levels;  |
   |          artifact                            |        |          audit what, if anything, is real)   |
   | exp5172 SOTA ingestion: diffusion guidance +  |        | exp5175 GAP-4891 Stage-3: A/B the relational  |
   |          hierarchical search + VerifySteer    |        |          mask pruner (built+unit-tested,      |
   |          follow-up (feeds exp5173 + Phase B)  |        |          never run) vs. unpruned, on the 3    |
   | exp5173 DiffusionGemma energy-guided          |        |          separating games (cd82/sk48/sp80),   |
   |          discrete-diffusion pilot [GATED on   |        |          reproduction-gated. THIS IS THE      |
   |          exp5171 passing] -- HumanEval/MBPP   |        |          SELF-LEARNING EXPERIMENT (online,    |
   |          executable domain, AR baseline via   |        |          pure-CPU, learns from the search's   |
   |          gemma-4-26B-A4B-it-GGUF              |        |          own observed transitions)            |
   +---------------------------------------------+        | exp5176 live level-up attempt using whichever |
                                                            |          of exp5174/exp5175 validated a lever |
                                                            |          (ARC Level-Up Attempt Guarantee)     |
                                                            +----------------------------------------------+
                                               |                                                              |
                                               +------------------------------+------------------------------+
                                                                              |
                          PHASE C: New verifier construction, hardware, capstone
                          +-------------------------------------------------------+
                          | exp5177 GAP-4 clean methodology rerun (fix bare-value  |
                          |          field root cause, then re-run scale-up)      |
                          | exp5178 VerifySteer-inspired latent-steering pilot     |
                          |          (new mechanism, efficiency-parity framing,    |
                          |          NOT a PHASE D rerun)                          |
                          | exp5179 hardware continuity (KV260 + PolarFire smoke + |
                          |          GateMate dirtyJTAG re-diagnosis)              |
                          | exp5180 capstone -- reconcile the whole milestone      |
                          +-------------------------------------------------------+
```

## SOTA Model Policy

Per CLAUDE.md, every experiment invoking an LLM must declare at least one of the three mandated SOTA
local GGUF models in `model_specs` UNLESS it falls under a more specific frozen-stack rule:

- **`exp5173` (DiffusionGemma pilot)** needs an autoregressive baseline for the guided-vs-unguided /
  best-of-N comparison. Uses `unsloth/gemma-4-26B-A4B-it-GGUF` as that AR baseline — the natural,
  scientifically-required control, not a bolted-on compliance checkbox.
- **`exp5177` (GAP-4 clean rerun)** is an offline/dev-scale calibration task (not the live ARC submission
  path), so it uses `unsloth/Qwen3.6-35B-A3B-GGUF` for the induction-quality generation it needs.
- **`exp5178` (VerifySteer pilot)** needs a model whose hidden states it can probe for the
  verification-boundary steering signal; uses `unsloth/gemma-4-26B-A4B-it-GGUF` (dense architecture,
  more tractable for activation-level hooking than the MoE variant).
- **ARC live-path tasks (`exp5174`/`exp5175`/`exp5176`)** stay on the FROZEN `Qwen3.5-9B-MTP` iGPU stack
  per `[[project_arc_live_generator]]` and the ARC-AGI-3 Submission Sprint Forcing Function's stack-freeze
  clause — this is a more specific rule than the general SOTA-model mandate and supersedes it for
  live-path work specifically. Any OFFLINE dev-scale induction inside those tasks may use GPU 0/1 (per the
  2026-06-27 GPU allocation directive) with a mandated-SOTA model if a large local LLM is genuinely needed.
- `exp5169`/`exp5170`/`exp5172`/`exp5179`/`exp5168`/`exp5180` do not invoke an LLM for their core work (an
  infra fix, a retirement/hygiene pass, a literature synthesis, hardware smoke tests, a transition, and a
  reconciliation capstone respectively) and are exempt.

## Phase Plan

### Phase 0: Ledger Hygiene (frees capacity for A and B)

Reserved infrastructure slots (2, per CLAUDE.md) plus the mandatory milestone transition.

- **`exp5168`** — standard archive-.473/activate-.474 transition.
- **`exp5169`** — fix `scripts/adversarial_verify.py`'s `qd-random-mutation-ablation-omitted` check so it
  scopes to artifacts making their OWN QD/energy-fitness claim, not any artifact that cites/reports one
  (the `exp5156` false-positive). Also audits whether milestone-transition modules set
  `flagged_adversarial` from `exit_code != 0` rather than from `max_severity`, since a WARN-only flag
  should not read the same as a CRITICAL one.
- **`exp5170`** — formally retires the PHASE D external-verifier-scoring constructions (LoRA-EBM
  holistic scorer, uPRM, EBRM) into `ops/exclusion_manifest.yaml`, following the `exp5165` precedent
  exactly. Writes a single publishable-null artifact consolidating all seven milestones of evidence.
  Explicitly scopes `blocked_patterns` to NOT catch `exp5178`'s activation-steering construction.

### Phase A: Cash In The ARC Oracle-Distinct Win

- **`exp5171`** — scale `exp5160`'s held-out cross-corpus set from n=24 to n>=30 (CLT floor) or more,
  using the same disjoint ARC-GEN pool source, and confirm the delta survives with real (not
  seed-identical) variance. Gates `exp5173`.
- **`exp5172`** — SOTA-ingestion task (mandatory per CLAUDE.md's SOTA-Ingestion Cycle Discipline for a
  bleeding-edge headline track): energy-guided/verifier-guided diffusion decoding literature (feeds
  `exp5173`'s design) and hierarchical/subgoal search for interactive agents (feeds Phase B).
- **`exp5173`** — the DiffusionGemma Use-Case-1 pilot itself: energy-guided discrete diffusion on an
  executable domain (HumanEval/MBPP), composing DiffusionGemma's native per-step token distribution with
  Carnot's executable verifier ensemble during the 12-48 denoising steps, measured against an unguided
  baseline AND a best-of-N AR baseline. Gated on `exp5171`.

### Phase B: Attack The Diagnosed Enumeration Wall

- **`exp5174`** — reconciliation audit: `GAP-LIVE-INTEGRATION`'s cited evidence
  (`arc_strategy_router`/`arc_world_model_dsl` unimported, `target_levels=1`, `value_weight=0.0`) is
  stale against current code (`arc_competition_agent.py` already imports both;
  `SUBMITTED_TARGET_LEVELS=3` since `exp4605`; `exp4652` already live-tested a real nonzero
  `value_weight` and found an honest zero-lift null attributed to distribution-shift/calibration, not
  cost). Updates `ops/verifier_gaps.md`'s status and evidence fields to match reality, and separately
  audits how many of the registry's 69 reproducible levels are `solve_provenance:
  live_agent_self_discovery` vs. `development_proxy` (the "mirage vs. real" question the gap actually
  cares about).
- **`exp5175`** — runs `arc_relational_mask_pruner.py` (built + unit-tested 2026-06-28, never
  empirically exercised) combined with the relational goal-energy from GAP-4891 Stage 1, A/B against an
  unpruned control, on the three games where Stage-2 confirmed goal-energy separates win from near-win
  (cd82, sk48, sp80). Reproduction-gated. This module is, by construction, an online/self-learning
  move-pruner (it learns which action classes never touch the relational target region from the search's
  own observed transitions) — satisfies CLAUDE.md's mandatory self-learning-experiment requirement.
- **`exp5176`** — reads both `exp5174` and `exp5175`'s outcomes directly (not a mechanical AND-gate, since
  either lever alone could be the win) and attempts a real live level-up on 2-3 currently-stuck games
  using whichever validated. Satisfies the ARC Level-Up Attempt Guarantee's structural floor regardless of
  outcome, per the `exp5159` precedent (a gated attempt that reports `blocked_upstream_gate_not_passed`
  still counts structurally).

### Phase C: New Verifier Construction, Hardware, Capstone

- **`exp5177`** — GAP-4 clean methodology rerun. Fixes the diagnosed root cause of `exp5161`'s CRITICAL
  flag (bare-value violation: `inference_substrate`/`model_specs`/`random_seed`/
  `reproducibility_checksum` must be bare, not wrapped in `{principle, value}` dicts — principle
  annotations belong in the separate `field_principles` block the artifact already demonstrates
  elsewhere), then re-runs the GAP-4 protocol scale-up `exp5161` itself recommended.
- **`exp5178`** — small, cheap, exploratory pilot of the VerifySteer mechanism (hidden-state
  verification-strictness steering) on a headroom-present corpus, targeting the north star's
  efficiency-parity win condition rather than an accuracy-beats-SC bar. Explicitly NOT a PHASE D rerun
  (different mechanism class: representation steering, not external scoring).
- **`exp5179`** — hardware continuity: KV260 + PolarFire smoke (expected steady-state pass, per 4+
  consecutive clean milestones) + one more GateMate dirtyJTAG IDCODE diagnostic attempt.
- **`exp5180`** — capstone: reconcile every task, exclude flagged artifacts from headline aggregation,
  update `ops/arc_solve_registry.yaml`/`ops/verifier_gaps.md`/`ops/known-issues.md`/`ops/status.md`/
  `ops/changelog.md` per the Documentation Update Rules.

## Dependency Graph

```text
exp5168 (transition)
  |
  +--> exp5169 (infra: adversarial_verify.py fix)         [independent]
  +--> exp5170 (infra: PHASE D retirement)                [independent]
  +--> exp5172 (SOTA ingestion)                            [independent]
  +--> exp5174 (GAP-LIVE-INTEGRATION reconciliation)       [independent]
  +--> exp5179 (hardware continuity)                       [independent]
  +--> exp5177 (GAP-4 clean rerun)                         [independent]
  +--> exp5178 (VerifySteer pilot, reads exp5170's retirement scope note only) [soft dep on exp5170]
  |
  +--> exp5171 (harden exp5160)
  |      |
  |      +--> exp5173 (DiffusionGemma pilot)   [gated_on exp5171.gate_passed==true]
  |
  +--> exp5175 (GAP-4891 Stage-3 pruner A/B, reads exp5172's hierarchical-search citations)
         |
         +--> exp5176 (live level-up attempt)  [reads exp5174 + exp5175 outcomes directly]

exp5180 (capstone)  <-- depends on ALL of the above (reads every artifact)
```

`exp5172` (SOTA ingestion) is a soft input to `exp5173` and `exp5175`'s designs but neither is
mechanically `gated_on` it — ingestion informs design quality, it is not a correctness precondition.

## Hardware Requirements

- **`exp5173` (DiffusionGemma pilot)** needs BOTH RTX 3090s: the only confirmed-successful load path
  (`results/diffusiongemma_energy_prior_probe.json`, `load_mode: 4bit_nf4_devmap_auto_2gpu`) splits the
  26B/4B-active MoE across 2 GPUs at 4-bit NF4. The prior GGUF-path attempt failed
  (`llama_cpp.Llama()` cannot load the converted GGUF — the architecture is not llama.cpp-native, this is
  a likely-permanent blocker, do not retry it) and the naive single-GPU transformers attempt OOM'd. The
  successful probe hit a SEPARATE, fixable bug (`Tensor.item() cannot be called on meta tensors` inside
  the diffusion forward pass under `device_map="auto"`) that `exp5173` must resolve before the actual
  guidance experiment can run. Per the 2026-06-27 GPU allocation directive (conductor owns GPU 0, outer
  loop owns GPU 1), this task should check GPU-1 availability via `nvidia-smi` before launching and report
  `blocked_gpu1_busy` honestly rather than contending silently, since it genuinely needs both devices.
- **`exp5177` (GAP-4 rerun)** and **`exp5178` (VerifySteer pilot)** need a single GPU (GPU 0, conductor's
  dedicated device) for GGUF inference at the mandated SOTA model scale.
- **`exp5174`/`exp5175`/`exp5176`** (ARC live-path work) use the frozen Qwen3.5-9B-MTP iGPU stack for any
  live generation and are otherwise CPU-only (search, pruning, reconciliation).
- **`exp5179`** needs SSH reachability to `kria` (KV260) and `polarfire`, plus USB access to the GateMate
  A1-EVB-2M's onboard DirtyJTAG programmer (`1209:c0ca`). No new hardware acquisition needed this
  milestone.

## Prior-Failure And Exclusion Discipline

- **`exp5170`** formally retires `exp5001`-`exp5086` (D-series, LoRA-EBM/uPRM/EBRM constructions) plus
  `exp5163` (`.473`'s MMLU-Pro continuation) into `ops/exclusion_manifest.yaml`, with
  `retire_if_same_verdict: true` already satisfied by the seven-milestone evidence trail. `blocked_patterns`
  must be scoped to "external verifier scoring beats/matches self-consistency via a LoRA-EBM/uPRM/EBRM-
  style construction" specifically — NOT to "any off-ARC verifier work" — so `exp5178`'s activation-steering
  pilot is not caught. This is the standard-format entry per the `exp5165`/`cross_game_value_transfer_
  retired_exp4342_v401` precedents.
- **`exp5177`** carries a `prior_failures` block citing `exp5161` (`flagged_adversarial: true`,
  CRITICAL `DURATION_TOO_SHORT`), with `addressed_by` naming the diagnosed bare-value-field root cause and
  `retire_if_same_verdict: true` — if the rerun produces the same flag pattern, GAP-4 protocol execution
  itself (not just this one artifact) should be added to the exclusion manifest pending a deeper fix.
- **`exp5171`/`exp5173`/`exp5174`/`exp5175`/`exp5176`/`exp5178`/`exp5179` do not scope-match any retired
  exp_id** in `ops/exclusion_manifest.yaml` as of `2026.07.473`'s close (checked against
  `cross_game_value_transfer_retired_exp4342_v401`, `fover_in_domain_pool_retired_v469`,
  `generation_axis_exploration_signal_retired_exp5154_v473`, and the entry `exp5170` itself is about to
  add) — no `operator_override` is required for them.

## Acceptance Criteria

- `exp5160`'s cross-corpus win either survives at n>=30 with a non-degenerate CI (Phase A proceeds to
  the DiffusionGemma pilot) or narrows honestly (the pilot stays gated and V475 re-evaluates — this is
  an acceptable, non-doomed outcome per the gate's own falsifiability).
- The relational-mask-pruner A/B produces a clean, reproduction-gated answer (pass or honest null) on at
  least the 3 target games — either outcome advances `ops/verifier_gaps.md` GAP-4891 from `building` to a
  resolved status (`filled` or a sharpened `status: building` note naming the NEXT specific lever).
- `ops/exclusion_manifest.yaml` gains the PHASE D retirement entry, verified via
  `scripts/exclusion_manifest_lint.py` to not false-positive against `exp5178`.
- `python3 scripts/publication_gate.py --json` still reports `paper_ready: true` at milestone close (no
  regression to the already-met G1-G4 gate).
- `python3 scripts/arc_levelup_guarantee_lint.py` passes against `research-roadmap-next.yaml` (>=1
  level-up attempt structurally present, via `exp5176`).
- `reproducible_total_levels`/`reproducible_total_games` in `ops/arc_solve_registry.yaml` either grows or
  is honestly reported flat with a named reason — never silently stale.

## Non-Goals

- **Do not re-run PHASE D's retired constructions** (LoRA-EBM external scorer, uPRM, EBRM) without a
  genuinely new corpus or a construction that is not scope-matched to the retirement. `exp5178`'s
  activation-steering pilot is the sanctioned exception (different mechanism class).
- **Do not re-propose GAP-LIVE-INTEGRATION's original "wire the router" framing** — `exp5174` is a
  reconciliation/audit, not a rebuild; the wiring already happened.
- **Do not wire `exp5164`'s retro-timing false-zero fix into `scripts/research_conductor.py`** from any
  V474 task — that file is off-limits to experiment prompts by design. This is flagged here as an
  outer-loop/operator action item for a future interactive session, not a roadmap task.
- **Do not attempt a KV260 latency/speedup claim.** The board is POC-tier and effectively at steady state
  (`ops/known-issues.md` "2026-06-01: KV260 v4-load + latency-number — DO NOT re-investigate"); `exp5179`'s
  KV260 touch is a reachability smoke test only.
- **Do not scope any task as "solve ALL levels of game X."** Per CLAUDE.md's Incremental-Progress Scoping,
  `exp5176` targets +1..+n levels on 2-3 named games, never an all-levels sweep.
