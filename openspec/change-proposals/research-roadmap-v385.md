# Research Roadmap v385 — Reach the faithful baseline via the LIVE CONTIGUOUS run (NOT the broken bounded-pass chain), then DECIDE the verifier moat with an ARBITER-grounded rerank-recovery test that is decision-grade at the INTERMEDIATE baseline

**Milestone:** 2026.06.385
**Planner:** Claude Opus 4.8, 2026-06-13 (operator-curated north star unchanged)
**Predecessor:** [research-roadmap-v383.md](research-roadmap-v383.md) (v384 was operator-pre-staged)
**Headline question:** Did the contiguous run advance the Sudoku-Extreme baseline, and did the
ARBITER-grounded verifier-RERANK recovery test DECIDE the verifier moat at the intermediate baseline —
breaking the 3-milestone pattern of producing no moat verdict because every milestone was 100% blocked
on reaching val ≥ 0.85?

---

## 1. What .384 proved (the honest read, from the artifacts not the prose)

The .384 headline ("did the epoch-fix un-stall the accumulation?") resolved **NEGATIVE — but the
diagnosis itself was wrong.** Reading the artifacts:

- **The "max_epochs epoch-fix" was a MISDIAGNOSIS.** `results/experiment_4146_*.json`:
  `config_max_epochs: 50000`, `checkpoint_epoch: 6399`, `max_epochs_cap_confirmed: false`. The cap was
  never the cause (6399 ≪ 50000). The pass did not fail to train — **it never started.** A pre-launch
  guard (`diagnose_epoch_cap → _noop_reason_for_diagnosis`) read a **stale Lightning Timer state**
  baked into the checkpoint (`timer_train_elapsed_s: 3641.99 ≥ max_time 3600`) and refused to launch,
  emitting `blocked_noop_cap_not_confirmed_timer_elapsed` (`duration_s: 0.121`,
  `post_epoch == seed_epoch == 6399`, `val_exact_accuracy: null`). exp4147/4148/4149 then defensively
  refused to blind-retrain → all `blocked_*_noop_unresolved`.
- **The bounded-pass chain is the broken mechanism.** Resume → 60-min Timer-bounded pass → write back →
  next pass re-triggers (a) the stale-Timer guard, (b) the `global_step == 0` progress-callback dead
  code (manual-optimization loop), and (c) the every-100-epoch write-back race. This is the same
  "powering-run split-mechanism fails across milestones" pattern as the `.373-.375 incident.
- **The real fix is already deployed and working.** An operator-launched **single 12-hour contiguous
  run** (PID 432538) resumes once from the intact 0.278 checkpoint, restores the manual LR step
  (`Restored manual LR step: 19006`), runs continuously, validates every 100 epochs, and persists via
  `save_last`. Measured live this planning session: **val 0.199 floor → 0.278 (.384 seed) → 0.372 →
  0.426 and climbing ~+1.3–1.9pp / 100 epochs.** The contiguous mechanism works where the bounded-pass
  chain failed. (Subtlety: the monitored `best_model_score` path returns `None`; progress persists via
  `save_last`, and val is read from the CSV, not the checkpoint's best-score slot.)
- **The decisive graft correctly DEFERRED** (`exp4150`: `complete: graft_deferred_baseline_below_0.85`)
  — honest, not fabricated. The executable Sudoku verifier + candidate decode + RFT arms are all
  **implemented and real** (`python/carnot/experiment_4109_*.py`, `experiment_4150_*.py`).
- **ARC offline game-solving plateaued** (`exp4151`:
  `fifteenth_game_no_solve_no_unsolved_strict_nonspatial_candidates`; total stuck at 13).
- **Observability is broken** — the operational-retro timing detector false-zeros across ~20 milestones
  (.363–.384: `total_wall_time_minutes: 0`, `experiments_completed: 0`) because its commit-attribution
  scan does not match the conductor's per-experiment commits to the active milestone. Every retro is
  data-blind on execution metrics.

**The structural problem .385 must fix:** `.382 (val 0.278, partial), `.383 (no-op), and `.384 (no-op,
wrong diagnosis) ALL produced **no decisive moat verdict**, because each was 100% gated on first
reaching the faithful (val ≥ 0.85) baseline. The baseline is now advancing via the contiguous run, but
0.85 will likely **not** land inside .385's execution window. A milestone that is again all-or-nothing
on the baseline is churn (north-star §1 / Depth-Over-Breadth).

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The verifier moat is UNPROVEN and keeps getting blocked (the existential gap).** Per north-star
   §5, with the generator commodity/third-party, the verifier is Carnot's entire value-add and "all of
   Carnot's risk now sits in ONE place." The decisive Sudoku graft is the intended proof but has been
   blocked on the baseline for 3 milestones. **Gap-closer:** an ARBITER-grounded
   (arXiv:2605.26172) verifier-RERANK recovery test that is decision-grade at the *intermediate*
   baseline — informative whenever there is measured headroom (oracle@K > vote@1), which a lower
   baseline has *more* of, not less.
2. **ARC-AGI-3 (the north star) — the real harness is unbuilt, and offline solving plateaued.** Per
   north-star §0, ARC-AGI-3 is scored on accuracy AND efficiency; the EFFICIENCY axis (actions-to-solve)
   is where the verifier earns its place (router + action-pruner). .384 exhausted the offline
   non-spatial solve candidates. **Gap-closer:** pivot the ARC slot from "solve another game" to the
   EFFICIENCY axis — an offline verifier-as-action-pruner vs a random/greedy baseline, the prerequisite
   for the offline-beats-baseline gate that precedes any online play (quota-conserve preserved).
3. **The loop cannot see itself (the observability gap).** ~20 milestones of data-blind retros means
   the meta-reflection discipline (CLAUDE.md Operational Principles) has been running on zeros.
   **Gap-closer:** a reserved-infra task to root-cause and fix the timing-detector attribution
   (without touching `research_conductor.py`), with a dated-changelog fallback.

---

## 3. Architecture — decide the moat at the intermediate baseline (not all-or-nothing on 0.87)

```
                   results/trm_runs/sudoku_extreme_baseline/last.ckpt  (the ONE shared baseline)
                                          |
        +---------------------------------+----------------------------------+
        | exp4157 BASELINE HARVEST + DEFENSIVE contiguous continue           |
        |   read val from CSV (no OOM); detect the live run; RECORD-ONLY if  |
        |   alive (no GPU contention); resume ONE contiguous run if dead.    |
        |   STEP-based anti-no-op guard (manual_lr_step), NOT epoch>seed.    |
        +---------------------------------+----------------------------------+
                                          | current_val, baseline_faithful
              +---------------------------+---------------------------+
              |                                                       |
   +----------v-----------+                              +------------v-------------+
   | exp4158 RERANK MOAT  |  decision-grade at ANY        | exp4159 REWARD GRAFT     |
   | (ARBITER 2605.26172) |  baseline with headroom       | (DEFENSIVE on val>=0.85) |
   |  headroom precheck:  |                               |  Phase-0 precision gate  |
   |  oracle@K > vote@1 ? |  --- snapshot ckpt --->        |  A=verifier-cert vs      |
   |  rerank by EXECUTABLE |                              |  B=vote-cert RFT, CI95   |
   |  Sudoku verifier;    |  cost_ratio_vs_llm_judge      |  excl 0; else DEFER       |
   |  pass@1 lift, CI95;  |  (EFFICIENCY axis)            |  -> .386 continues        |
   |  recovers_outvoted   |                               |  resolves DiffusionGemma |
   +----------+-----------+                              +------------+-------------+
              |                                                       |
              +----------------------+--------------------------------+
                                     | the .385 MOAT verdict (rerank-decided even if graft defers)
                                     v
   exp4160 ARC EFFICIENCY (offline action-pruner vs baseline)  ||  exp4161 OBSERVABILITY fix
   exp4162 SOTA ingestion  ||  exp4163 registry/gaps hygiene   ||  exp4164 hardware continuity
                                     |
                                     v
                          exp4165 CAPSTONE .385 (ungated aggregation)
```

The key architectural change vs .382–.384: the moat verdict no longer hangs on the single
all-or-nothing `val ≥ 0.85` gate. **exp4158 (rerank) decides the moat at the intermediate baseline;
exp4159 (graft) is the publishable de-confound that runs when/if the baseline becomes faithful.**

---

## 4. Phase descriptions

### Phase A — Baseline (exp4157) — `track: trm-baseline`
Defensively HARVEST the live contiguous run's val (from the CSV, no OOM, no torch-to-GPU). If the
operator's run is alive, RECORD ONLY (never launch a competing run — GPU + checkpoint-write contention).
If dead and val < 0.85, resume ONE contiguous run (the working template; `+trainer.max_time` long;
`save_last`; write back to the stable dir; progress prints). Anti-no-op guard is **STEP-based**
(`manual_lr_step` advanced + a real val row written) — never the `epoch > seed` guard that misfired in
.384. The bounded-pass chain is forbidden.

### Phase B — The verifier moat (exp4158 rerank, exp4159 graft) — `track: verifier-moat` / `verifier-as-reward`
- **exp4158 (the decision-grade result):** snapshot the current checkpoint; headroom precheck
  (oracle@K > vote@1 — the positive control that makes a null informative, per FALSE_NEGATIVE_RISK);
  rerank K TRM candidates by the executable Sudoku verifier; report pass@1 lift vs majority vote
  (bootstrap CI95), the count of present-but-out-voted puzzles recovered (ARBITER), and the verifier's
  per-candidate cost vs an LLM-judge estimate (the EFFICIENCY axis). This is the moat signal for .385
  regardless of whether 0.85 is reached. The executable verifier is high-precision and ORTHOGONAL to
  the vote — exactly the property ARBITER says an external verifier needs to beat self-consistency.
- **exp4159 (the publishable de-confound, defensive):** IF val ≥ 0.85, run the Phase-0 precision gate
  (P(test-gold | demo-perfect) ≥ 0.85) then A=verifier-cert vs B=vote-cert RFT (N-matched, same TRM,
  held-out delta, CI95 excl 0). ELSE defer honestly. A positive result here resolves the DiffusionGemma
  gate.

### Phase C — North-star EFFICIENCY (exp4160) — `track: arc`
Offline, air-gapped: verifier-as-action-pruner vs a random/greedy baseline on the ARC-AGI-3-style
fixtures → `action_efficiency_ratio`. Attempt the next incremental level (honest no-solve = COMPLETE).
Advances the offline harness toward the offline-beats-baseline gate that precedes online play.

### Phase D — Mandated slots + close (exp4156, 4161, 4162, 4163, 4164, 4165)
Archive/activate (exp4156, codex), observability fix (exp4161, reserved infra + overdue priority),
SOTA-ingestion (exp4162, reserved bleeding-edge slot), registry/gaps hygiene (exp4163, reserved infra +
regression guard), hardware continuity (exp4164, GateMate + PolarFire non-terminal), capstone (exp4165).

---

## 5. Dependency graph

```
exp4156 (archive/activate) ──► everything (milestone active)
exp4157 (baseline harvest) ──► exp4158 (rerank: needs current_val + checkpoint)
                           └──► exp4159 (graft: needs baseline_faithful)
exp4158, exp4159 ──► exp4163 (registry records the moat outcomes)
exp4158, exp4159 ──► exp4165 (capstone aggregates the moat verdict)
exp4160, exp4161, exp4162, exp4164 ──► exp4165 (capstone aggregates)
```

All inter-task dependencies are handled by **defensive in-prompt reads** (read the upstream artifact;
branch on its fields), NOT `gated_on` — matching the .384 chain, so a soft-blocked upstream never
skip-cascades the downstream. The capstone is explicitly `NO gated_on` (aggregate whatever exists).

---

## 6. Hardware requirements

- **2× RTX 3090 (CUDA):** the contiguous baseline run owns GPU 1 (operator-launched). exp4157 must
  RECORD-ONLY while that run is alive; any conductor-launched continuation uses a free GPU. exp4158/4159
  load the snapshot checkpoint for decode/rerank/RFT (CPU-load scalars; GPU only for the forward decode).
- **Read val without OOM:** prefer the metrics CSV; if touching the checkpoint, `torch.load(..., map_location='cpu', weights_only=False)` scalar fields only — never the single-4227-row full-batch
  validation (it OOMs; the default 768-batch / 5-batch val path does not).
- **FPGA boards (exp4164):** GateMate (DirtyJTAG `--detect`) + PolarFire (`ssh polarfire`) non-terminal;
  KV260 (`ssh kria`) opportunistic-terminal. SSH / USB-detect preconditions ONLY (KV260 SSH-Not-SD-Card
  Discipline).

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The baseline does not reach 0.85 in .385 (likely) | **exp4158 decides the moat at the intermediate baseline** — the milestone is no longer all-or-nothing on 0.85. The graft defers honestly; .386 continues the contiguous run. |
| exp4157 launches a run that fights the operator's live run (GPU/checkpoint contention) | exp4157 RECORDS ONLY while the run is alive (PID + advancing-CSV liveness check); it resumes a run ONLY if the live run is confirmed dead. |
| Reading a half-written checkpoint while the live run is checkpointing | exp4158/4159 snapshot the checkpoint to a frozen path before loading. |
| Rerank null is uninformative because there is no headroom (the .383 FALSE_NEGATIVE) | Mandatory headroom precheck (oracle@K > vote@1) — a null is only reported as a moat finding when the positive control passes; otherwise `no_headroom_uninformative`. |
| The bounded-pass chain is re-proposed and no-ops again | Forbidden in every TRM prompt; the stale-Timer-guard root cause is recorded in exp4156's close-state; only the contiguous run is allowed. |
| Observability fix needs to touch `research_conductor.py` | exp4161 writes a precise patch PROPOSAL instead of editing that file (the standing constraint). |

---

## 8. Compliance with mandated disciplines

- **Codex-Default v2:** all 10 tasks `agent_type: codex` + `model: gpt-5.5`. No gemini. Archive runs on
  codex (mechanical), not Opus (2026-06-12 quota-conserve).
- **Verdict Terminal-Prefix:** every task's `honest_verdict` principle requires a terminal prefix.
- **Principle-Annotated Artifact Fields:** every REQUIRED ARTIFACT FIELD + every ACCEPTANCE GATE carries
  a `principle:`.
- **Failed-Experiment Rerun / Exclusion-Manifest:** every continuation carries an `operator_override`
  (≥10 chars) naming the false-positive scope-match and the forward difference; the contiguous/rerank/
  graft scope is NOT on the exclusion manifest.
- **Inference-Substrate Declaration:** each task declares its substrate (`gpu_nanotrm_native_training`,
  `verifier_ensemble_against_cached_candidates`, `offline_arc_explore_induce_verify`,
  `aggregation_from_upstream_artifacts`, `hardware_smoke`).
- **Pre-Launch Preconditions:** every compute-bound task has a PRECONDITIONS step 0 with `blocked_*`
  fallbacks; KV260 uses SSH-not-SD-card.
- **Reserved infra slots (≥2):** exp4161 (observability) + exp4163 (registry hygiene). **SOTA-ingestion
  slot:** exp4162. **Hardware continuity:** exp4164.
- **Adversarial Artifact Verification + FALSE_NEGATIVE_RISK:** exp4158's headroom precheck is the
  positive control; exp4159's Phase-0 precision gate guards label noise; the capstone SKIPs
  `flagged_adversarial` artifacts and cites upstream sha256.
- **Continuous self-learning (research-program.md):** exp4159 (verifier-as-reward RFT) is the
  self-learning experiment (the generator improves from verifier-certified data — the FR-11 / Phase-3
  endgame).
- **North star:** exp4158 (verifier accuracy moat) + exp4160 (verifier efficiency axis) directly
  advance the ARC-AGI-3 north-star metrics; exp4159 + the DiffusionGemma gate are the depth scale-up.
