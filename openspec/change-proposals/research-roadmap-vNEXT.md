# Research Roadmap — Milestone 2026.04.92

**Title:** Paper Integrity Remediation + GRPO v5 GPU Unblock + SC-Energy Regularization + Latent-GRPO

**CalVer:** 2026.04.92 (sequence increment from 2026.04.91)

**Date:** 2026-05-02

**Experiments:** exp1178 – exp1190 (13 tasks)

**Estimated wall time:** ~420 min

---

## What Milestone 2026.04.91 Proved

Milestone .91 completed 11/13 criteria (84.6%).

Key substantive findings:

1. **Phase 4 active inference pilot operational** (exp1165): Blocked Gibbs free-energy minimization
   on 10 synthetic 5x5 ARC-AGI-3 puzzles achieved 74.7% action reduction vs greedy baseline
   (phase4_mean_action_count: 2.1, baseline_mean_action_count: 8.3). Energy traces were monotone-
   decreasing in 78% of episodes. The F(z) = Σ_k w_k E_k(z) formulation is empirically viable.

2. **k=6 AND-compose degraded — SC-Energy overfit diagnosed** (exp1176): AUROC dropped from
   0.9240 (k=5 baseline, re-evaluated in .91) to 0.8973 when SC-Energy was added as a 6th verifier.
   Root cause: SC-Energy achieved AUROC=1.0 on the training corpus (exp1168) — a clear overfit signal.
   The k=5 ensemble remains the production default; k=6 requires regularized retraining.

3. **DoT (Diffusion of Thought) produced zero signal** (exp1171): dot_t1_auroc=0.5 and
   dot_t125_auroc=0.5 at all diffusion temperatures. The per-token energy-gradient masking
   approach has no discriminative power because sequence-level EBM energy functions have flat
   gradients at the token level. DoT requires a redesign using the EBM-diffusion formulation
   from arXiv 2410.21357 (continuous noise injection at sequence level).

4. **GRPO v5 root cause confirmed: llama.cpp no GPU offload** (exp1173): honest_verdict=
   training_wall_hit, training_completed=False. The routing-bug hypothesis in known-issues.md
   was a red herring. The actual cause: the conductor environment's llama.cpp build lacks GPU
   offload support. GRPO v5 needs --n-gpu-layers > 0 to load 35B models in adequate time.
   Fix is architectural (rebuild llama.cpp with CUDA/ROCm support), not a planner fix.

5. **BEAVER-lite sound with real logprobs** (exp1170): Mock issue from earlier milestones resolved.
   Real llama.cpp logprobs confirm prefix-closed bound holds. BEAVER-lite is operational.

6. **FoVer corpus extended to 7,329+ pairs** (exp1169): SC-Energy labels added. Corpus ready for
   regularized SC-Energy retraining in .92.

7. **Phase 4 publication hold conditions**: Phase 4 empirical result (exp1165) now provides the
   "EBM thinking advantage" demonstration required by the hold. However, 18 figure/claim integrity
   issues (5 critical, 5 high, 5 medium, 3 low) still block arXiv submission independently.
   .92's primary mission is resolving those integrity issues.

**NOT MET in .91:**
- `paper_v4_phase4_section_integrated`: exp1183 paper compilation not reached (gated on critical fixes)
- `grpo_v5_honest_result`: GATE_BLOCKED (llama.cpp GPU offload prerequisite not yet fixed)

**Open items carried into .92:**
- Figure integrity audit (18 issues documented in paper-v5-integrity-remediation.md)
- Hardware claim audit (FPGA KL=3.07 is software proxy, not bitstream-measured)
- GRPO v5 after llama.cpp GPU offload fix
- SC-Energy overfit diagnosis + regularized k=6 retraining
- DoT non-monotone diagnosis + EBM-diffusion redesign

---

## Three Biggest Gaps to PRD Vision

### Gap 1: Publication Hold — 18 Integrity Issues Block arXiv

The paper-v5-integrity-remediation.md audit identified 18 issues:
- **5 critical** (each individually blocks submission per CLAUDE.md "all headline results must have
  live GPU provenance"): fig3 11680x fabricated speedup, KL=3.07 software proxy, 15.6x hand-typed
  constant, HardNet++ apples-to-oranges, exp1121 SOSKANEnergyV3 collapse hidden
- **5 high**: GRPO confidence intervals missing, HumanEval baseline 0.0% harness failure, alpha_t
  24/100 rejection rate undisclosed, Phase-4 pilot trivial-greedy baseline, Seed IQ confirmed=false
- **5 medium**: ThinkPRM citation incomplete, holdout n=50 not stated, NRGPT non-monotone not
  disclosed, SOS-KAN AUROC reconciliation, fig2 binormal caveat missing
- **3 low**: bibliography stub entries, Table 1 caption, hardware portability theorem scope

Addressing all 18 in .92 is the milestone's primary deliverable. The 4-test framework from
paper-v5-integrity-remediation.md governs every fix:
1. SOURCE-ARTIFACT: every constant traces to a specific results/ JSON field
2. SAME-BASIS COMPARISON: speedup comparisons use same measurement basis both sides
3. PROMINENT CAVEATS: extrapolated/estimated claims carry equally-prominent caveats
4. NO HALLUCINATED CITES: every carnot.bib entry verified as real published work

### Gap 2: GRPO v5 Self-Learning Blocked — llama.cpp GPU Offload Missing

GRPO v4 established +10.0pp as the single best training result. GRPO v5 (adding TinyV v2
reward shaping + dual-GPU) has failed in exp1139 and exp1173. The root cause is now
definitively confirmed: llama.cpp lacks --n-gpu-layers support in the current build.
The fix is a one-time infrastructure investment: rebuild llama.cpp with CUDA support
(LLAMA_CUDA=1) on the RTX 3090 rig. Once that's done, GRPO v5 can run in its correct
environment. Every subsequent GRPO experiment benefits permanently.

### Gap 3: Verifier Ensemble Ceiling — SC-Energy Overfit Blocks k=6

The k=5 ensemble (AUROC=0.9402, post exp1128) is the current production ceiling. The k=6
attempt in .91 regressed to AUROC=0.8973 because SC-Energy overfits on training corpus
(AUROC=1.0). The path to k=6 requires regularized SC-Energy retraining on the extended
7,329-pair FoVer corpus using held-out evaluation. Separately, the DoT redesign (EBM-
diffusion formulation) is a speculative path toward finer-grained token-level energy signals.

---

## Milestone Architecture

```
Phase 0: Infrastructure (2 tasks, MANDATORY, unconditional)
         exp1178 Pytest memory watchdog (ops stability)
         exp1179 llama.cpp GPU offload fix (GRPO v5 unblock)

Phase 1: Paper Integrity (4 tasks, MANDATORY)
         exp1180 Critical ISSUE-1 to -5 + figure_integrity_audit.py (opus)
         exp1181 High ISSUE-6 to -10 (sonnet)
         exp1182 Medium/low ISSUE-11 to -18 + paper_claim_audit.py (sonnet)
         exp1183 Paper v5 recompile + arXiv bundle v6 (gated: 1180+1181 cleared)

Phase 2: Self-Learning MANDATORY
         exp1184 GRPO v5 + TinyV v2 (claude/opus, DualGPU, gated: 1179 verified)

Phase 3: Verifier Investigations
         exp1185 SC-Energy overfit + regularized k=6 (prior_failures: exp1176)
         exp1186 DoT EBM-diffusion redesign (prior_failures: exp1171)

Phase 4: New Research
         exp1187 Latent-GRPO energy reward (arXiv 2604.27998)
         exp1188 WOPR Hex game cartridge

Phase 5: Phase 4 Scale-Up + ISSUE-9 Fix
         exp1189 Phase 4 stronger baseline pilot + 10x10 grid (prior_failures: exp1165)

Phase 6: Retro
         exp1190 Milestone 2026.04.92 retro
```

Dependency graph:
```
exp1178 ──────────────────────────────────────────────────────────> (unconditional)
exp1179 ──────────────────────────────────────────────────────────> (unconditional)
exp1180 ──────────────────────────────────────────────────────────> (unconditional)
exp1181 ──────────────────────────────────────────────────────────> (unconditional)
exp1182 ──────────────────────────────────────────────────────────> (unconditional)
exp1183 ─── gated_on: exp1180.critical_issues_fixed AND exp1181.high_severity_fixed
exp1184 ─── gated_on: exp1179.llama_cpp_gpu_offload_verified
exp1185 ──────────────────────────────────────────────────────────> (unconditional, prior_failures)
exp1186 ──────────────────────────────────────────────────────────> (unconditional, prior_failures)
exp1187 ──────────────────────────────────────────────────────────> (unconditional)
exp1188 ──────────────────────────────────────────────────────────> (unconditional)
exp1189 ──────────────────────────────────────────────────────────> (unconditional, prior_failures)
exp1190 ─── gated after all others complete (retro)
```

---

## Phase 0: Infrastructure

### exp1178 — Pytest Memory Watchdog

**Problem:** 5 swap saturation spikes per session are the leading cause of conductor slowdown
and OOM kills in the test suite. The spikes happen when a test leaks a large JAX array or
llama.cpp model handle and pytest accumulates them across the session without gc.

**Fix:** Ship a pytest plugin (`conftest.py` hook) that:
- Records RSS before and after each test
- Kills the test immediately if RSS delta > 500 MB in a single test
- Emits a warning if cumulative RSS > 8 GB across the session
- Writes a per-test memory log to `results/pytest_memory_*.log`

**Acceptance gate:** `watchdog_operational=True`, `no_oom_kill_in_sample_run=True`.

### exp1179 — llama.cpp GPU Offload Fix

**Problem:** exp1173 confirmed that llama.cpp was built without GPU offload support.
`--n-gpu-layers 83` silently falls back to CPU, causing 35B model inference to time out
in the 2400s grace period.

**Fix:** Rebuild llama.cpp with `LLAMA_CUDA=1` (or `LLAMA_ROCM=1` on the ROCm path):
```bash
cd vendor/llama.cpp
LLAMA_CUDA=1 make -j$(nproc)
# verify:
./main -m <path-to-35B-gguf> -n 5 --n-gpu-layers 83 -p "hello"
```
Emit a JSON artifact confirming `gpu_offload_verified=True`, `layers_on_gpu >= 80`,
`throughput_tokens_per_sec >= 50.0`.

**Acceptance gate:** `gpu_offload_verified=True` AND `throughput_tokens_per_sec >= 50.0`.
GRPO v5 (exp1184) is gated on this result.

---

## Phase 1: Paper Integrity

### exp1180 — Critical ISSUE-1 to -5 + figure_integrity_audit.py

Address all 5 critical issues that individually block arXiv submission:

- **ISSUE-1** (fig3 11680x): Use path (c) from remediation.md — run real CPU benchmark (same
  N=64, per-sample basis as exp1068). Cite exp1094's 15.96µs C++ Glauber measurement. Headline
  becomes ~250x with honest same-basis comparison. Re-render fig3.
- **ISSUE-2** (KL=3.07 software proxy): Sweep main.tex for all "FPGA KL=3.07" mentions; rewrite
  to "software-proxy KL=3.07; bitstream KL not yet measured on-board".
- **ISSUE-3** (15.6x hand-typed constant): Use path (b) — retract 15.6x; rewrite to cite
  exp1094's measured 15.96µs CPU vs FPGA bitstream as the honest comparison (~249x).
- **ISSUE-4** (76,130x HardNet++): Rewrite to "117µs per violation vs 8.9s for prompt repair
  on same 20 cases (exp1147)". Drop multiplicative framing.
- **ISSUE-5** (SOSKANEnergyV3 collapse hidden): Add AUROC=0.3333 production-corpus finding to
  Section 5 as offensive remediation. Frame: two AUROCs, two corpora, one verifier.

Ship `scripts/figure_integrity_audit.py` that scans `docs/figures/*.py`, extracts numerical
constants, and flags any constant that doesn't trace to a `results/` artifact field.

**Acceptance gate:** `critical_issues_fixed=5`, `figure_integrity_script_active=True`,
`4_test_passes_critical=True`.

### exp1181 — High-Severity ISSUE-6 to -10

- **ISSUE-6** (GRPO confidence intervals): Rewrite GRPO claims with n=25/n=47 inline; add
  binomial CIs; add small-sample caveat at least as prominent as +8.51pp number.
- **ISSUE-7** (HumanEval 0.0% harness failure): Reframe as "after extraction-fix, +36pp" with
  explicit pre/post extraction caveat. Move from headline to anomaly section.
- **ISSUE-8** (alpha_t 24/100 rejection rate): Add 24/100 ground-truth-correct rejection rate
  to alpha_t=0.38 framing explicitly.
- **ISSUE-9** (Phase-4 trivial baseline): Reframe exp1165 pilot with honest characterization of
  puzzle difficulty. Add footnote: "Baseline is random legal action; stronger baselines tested
  in exp1189." Point forward to exp1189 results.
- **ISSUE-10** (Seed IQ confirmed=false): Add footnote to Table 5 explicitly stating "Seed IQ
  row is documented fallback evidence (ops/known-issues.md); not independently re-fetched in
  this work."

**Acceptance gate:** `high_severity_fixed=5`, `4_test_passes_high=True`.

### exp1182 — Medium/Low ISSUE-11 to -18 + paper_claim_audit.py

Address remaining 8 issues (ISSUE-11 through ISSUE-18). Ship `scripts/paper_claim_audit.py`
that reads main.tex, extracts every numerical claim, verifies each is followed within 200 chars
by an `(expNNNN)` citation, and checks the claimed value against the artifact JSON field.

**Acceptance gate:** `medium_low_issues_fixed=8`, `paper_claim_audit_script_active=True`.

### exp1183 — Paper v5 Recompile + arXiv Bundle v6

**Gated on:** `exp1180.critical_issues_fixed=True AND exp1181.high_severity_fixed=True`

Compile the corrected main.tex (with pdflatex or submit .tex source to arXiv directly), build
the final arXiv bundle v6 (main.tex + carnot.bib + 7 corrected figures), verify the bundle
passes the 4-test audit tool automatically.

**Acceptance gate:** `arxiv_bundle_v6_ready=True`, `4_test_full_pass=True`,
`pdf_compiles_without_error=True`.

---

## Phase 2: Self-Learning

### exp1184 — GRPO v5 + TinyV v2 (DualGPU MANDATORY)

**Gated on:** `exp1179.llama_cpp_gpu_offload_verified=True`

**Prior failures:**
- exp1139: GRPO v5 attempt, honest_verdict=gpu_offload_blocked
- exp1173: GRPO v5 retry after routing fix, honest_verdict=training_wall_hit (real cause:
  llama.cpp no GPU offload, not routing). Addressed_by: exp1179 rebuilds llama.cpp with
  LLAMA_CUDA=1; exp1184 verifies --n-gpu-layers ≥ 80 throughput ≥ 50 tok/s before training.

GRPO v5 adds two improvements over v4's +10.0pp:
1. **TinyV v2 reward shaping**: instead of binary correct/incorrect, uses calibrated
   ThinkPRM v2 energy score (AUROC=0.9946) as continuous reward signal.
2. **DualGPU**: split inference across both RTX 3090s for 35B model throughput.

Use SOTA local GGUF (unsloth/Qwen3.6-35B-A3B-GGUF as the training target).

**Acceptance gate:** `training_completed=True`, `grpo_v5_delta_pp > 0.0` (any positive delta
counts as non-regression; > 10.0pp would exceed v4 record), `dualgpu_confirmed=True`.

---

## Phase 3: Verifier Investigations

### exp1185 — SC-Energy Overfit Diagnosis + Regularized k=6

**Prior failures:**
- exp1176: k6_above_k5=False, AUROC=0.8973 (degraded from k=5's 0.9240).
  Addressed_by: exp1185 diagnoses overfit (sc_energy_auroc=1.0 on training corpus per exp1168)
  and retrains SC-Energy with dropout + L2 regularization on the full 7,329-pair FoVer corpus
  with 20% held-out evaluation set. Retire if regularized AUROC still degrades ensemble.

Steps:
1. Retrain SC-Energy with dropout=0.3 and L2 weight decay=1e-4 on 7,329 pairs.
2. Evaluate on held-out 20% split (1,466 pairs not seen during training).
3. Re-run k=6 AND-compose evaluation using the regularized SC-Energy.
4. If k=6 AUROC >= k=5 AUROC (0.9240), declare k=6 viable.

**Acceptance gate:** `sc_energy_regularized=True`, `sc_energy_holdout_auroc < 0.98` (overfitting
resolved), `k6_auroc_vs_k5_delta` reported honestly.
`retire_if_same_verdict: true` (if k6 still degrades ensemble, retire k=6 from future milestones).

### exp1186 — DoT Energy Gradient Diagnosis + EBM-Diffusion Redesign

**Prior failures:**
- exp1171: dot_t1_auroc=0.5, dot_t125_auroc=0.5 (zero discriminative signal at all T).
  Addressed_by: exp1186 diagnoses root cause (sequence-level EBM has flat gradients at token
  level) and redesigns using EBM-diffusion formulation from arXiv 2410.21357. Key idea: instead
  of masking tokens by token-level energy gradient, inject Gaussian noise at the sequence
  embedding level and use the EBM score function (∇_z E) for guided denoising.
  Retire if EBM-diffusion also produces AUROC ≤ 0.55 (near random).

**Acceptance gate:** `diagnosis_complete=True`, `redesigned_dot_auroc` reported honestly,
`retire_if_same_verdict: true`.

---

## Phase 4: New Research

### exp1187 — Latent-GRPO Energy Reward Integration

**Based on:** arXiv 2604.27998 (Latent-GRPO: invalid-sample masking + one-sided noise).
Latent-GRPO achieved +7.86pp on low-difficulty FoVer questions by masking invalid samples
before the GRPO policy gradient update (instead of allowing negative rewards from invalid
rollouts to corrupt the gradient).

**Integration plan:**
1. Implement `python/carnot/training/latent_grpo.py` adding:
   - `mask_invalid_samples(rollouts)` — filters rollouts where the verifier energy is undefined
     or degenerate (e.g., all-same logprob distribution)
   - `one_sided_noise_injection(rollout)` — adds noise only to the positive reward side
     (per arXiv 2604.27998 Eq. 3) to prevent reward hacking
2. Evaluate on 100-question FoVer subset (low-difficulty bucket).
3. Compare delta_pp vs GRPO v4's +10.0pp on same subset.

**Acceptance gate:** `latent_grpo_delta_pp > 0.0`, `invalid_mask_rate` reported,
`comparison_to_grpo_v4_honest=True`.

### exp1188 — WOPR Hex Game Cartridge

**Based on:** openspec/capabilities/wopr-games/spec.md

Implement the Hex game as a well-defined constraint satisfaction domain for testing WOPR
game-playing capabilities. Hex is ideal because:
- Perfect information, two-player, zero-sum
- Winning condition = graph connectivity (natural SAT constraint)
- No draws (Hex is always decisive)
- Carnot's k=5 verifier ensemble can evaluate "does this board position satisfy the
  winning constraints?" as a direct energy query

**Implementation:**
1. `python/carnot/games/hex.py`: HexBoard, HexGame, HexAction classes
2. `python/carnot/games/hex_verifier.py`: HexConstraintVerifier (wraps k=5 ensemble to
   evaluate whether a board position is a winning configuration)
3. 10-game evaluation: random vs greedy vs energy-minimizing player
4. Baseline: random play; energy-minimizing player uses Blocked Gibbs to find lowest-energy
   legal action

**Acceptance gate:** `hex_game_operational=True`, `energy_player_win_rate > 0.5` vs random.

---

## Phase 5: Phase 4 Scale-Up

### exp1189 — Phase 4 Stronger Baseline Pilot + 10x10 Grid

**Prior failures:**
- exp1165: phase4_better_than_baseline=True BUT baseline was random-legal-greedy (trivial).
  ISSUE-9 from paper-v5-integrity-remediation.md requires a non-trivial baseline before the
  result can be published. Addressed_by: exp1189 implements BFS-to-goal as the baseline,
  which is guaranteed to find the optimal solution if the puzzle is tractable, and tests on
  10x10 grids (harder than 5x5). Retire if Phase 4 cannot beat BFS on any puzzle size.

**Steps:**
1. Extend ARC3PuzzleEnv to support 10x10 grids with 5-8 legal actions per step.
2. Implement BFS baseline (optimal for small grids, tractable for 10x10 with branching ≤ 8).
3. Run Phase 4 (Blocked Gibbs free-energy minimization) vs BFS on 10 puzzles.
4. Capture full free_energy_values trace across all 10 puzzles (fixes ISSUE-9).
5. Report action_count_ratio Phase4/BFS (< 1.0 means Phase 4 is more efficient).

**Acceptance gate:** `phase4_vs_bfs_delta_reported=True`, `stronger_baseline_implemented=True`,
`free_energy_values_all_10_puzzles=True`.
`retire_if_same_verdict: true` (if Phase 4 cannot beat BFS on ≥5/10 puzzles, retire the pilot
from future milestones and document as limitation).

---

## Phase 6: Retro

### exp1190 — Milestone 2026.04.92 Retro

Standard post-milestone retrospective covering:
- Paper integrity: all 18 issues resolved? Which remain?
- GRPO v5: training completed? Delta vs v4?
- k=6 AND-compose: viable after regularization?
- DoT redesign: any AUROC signal?
- Latent-GRPO: integration result?
- Phase 4 vs BFS: defensible claim?
- What was the slowest task? Any recurring failure modes?
- ops/known-issues.md updates for .93

---

## Hardware Requirements

| Task | Hardware | Notes |
|------|----------|-------|
| exp1179 llama.cpp fix | RTX 3090 (CUDA) | Build + verify GPU offload |
| exp1180 paper figures | CPU | LaTeX + matplotlib only |
| exp1181-1183 paper | CPU | LaTeX edits only |
| exp1184 GRPO v5 | 2x RTX 3090 (DualGPU MANDATORY) | 35B model, grace_period_s:2400 |
| exp1185 SC-Energy retrain | RTX 3090 | JAX training + k=6 eval |
| exp1186 DoT redesign | RTX 3090 or CPU | EBM-diffusion prototype |
| exp1187 Latent-GRPO | RTX 3090 | 100-Q subset, lower GPU pressure |
| exp1188 WOPR Hex | CPU | Pure Python game logic |
| exp1189 Phase 4 10x10 | CPU or RTX 3090 | Blocked Gibbs + BFS comparison |

---

## Success Criteria (13 items, ≥11/13 = milestone complete)

1. `exp1178_watchdog_operational = True` (pytest memory watchdog active)
2. `exp1179_gpu_offload_verified = True` (llama.cpp GPU offload confirmed)
3. `exp1180_critical_issues_fixed = 5` (ISSUE-1 through ISSUE-5 all resolved)
4. `exp1181_high_severity_fixed = 5` (ISSUE-6 through ISSUE-10 all resolved)
5. `exp1182_medium_low_fixed = 8` (ISSUE-11 through ISSUE-18 all resolved)
6. `exp1183_arxiv_bundle_v6_ready = True` (paper compiles, 4-test passes; gated)
7. `exp1184_grpo_v5_result_honest = True` (any result is honest; gated on GPU fix)
8. `exp1185_sc_energy_regularized = True` (overfit diagnosed + retrained)
9. `exp1186_dot_diagnosis_complete = True` (root cause confirmed, redesign prototyped)
10. `exp1187_latent_grpo_delta_honest = True` (any reported delta)
11. `exp1188_hex_game_operational = True` (Hex cartridge running + win rate reported)
12. `exp1189_phase4_stronger_baseline = True` (Phase 4 vs BFS on 10x10 reported)
13. `exp1190_retro_complete = True` (retro artifact written)

**Publication hold lift conditions** (tracked separately from milestone criteria):
```
phase_1_critical_fixes_landed = 5/5     (exp1180 → ISSUE-1 through ISSUE-5)
phase_2_high_severity_landed  = 5/5     (exp1181 → ISSUE-6 through ISSUE-10)
phase_3_medium_landed         = 5/5     (exp1182 → ISSUE-11 through ISSUE-15)
phase_4_low_landed            = 3/3     (exp1182 → ISSUE-16 through ISSUE-18)
figure_integrity_script_active = True   (exp1180 ships scripts/figure_integrity_audit.py)
paper_claim_audit_hook_active  = True   (exp1182 ships scripts/paper_claim_audit.py)
4_test_passes_for_every_claim  = True   (exp1183 verifies end-to-end)
operator_explicit_approval     = False  (operator must say "submit it")
```

---

## Decentralization Check (CLAUDE.md Rules 1-7)

- Rule 1 (local-first): GRPO v5 (exp1184) uses unsloth/Qwen3.6-35B-A3B-GGUF via llama.cpp.
- Rule 2 (closed-weight optional): no closed-weight dependencies added.
- Rule 3 (distribution mirroring): arXiv bundle v6 targets both arXiv + IPFS.
- Rule 4 (multiple surfaces): paper integrity fixes touch docs/ only, no API surface changes.
- Rule 5 (hardware portability): GRPO v5 uses CUDA primary + ROCm fallback (existing paths).
- Rule 6 (data minimization): no new closed-weight LLM calls added.
- Rule 7 (no vendor abstractions in core): exp1187 Latent-GRPO uses abstract `SamplerBackend`.

All 7 rules satisfied.

---

## Cross-References

- Paper integrity remediation plan: `openspec/change-proposals/paper-v5-integrity-remediation.md`
- Current known issues + publication hold: `ops/known-issues.md`
- GRPO v5 prior failures: `results/experiment_1173_grpo_v5_tinyv_fn_correction.json`
- SC-Energy overfit evidence: `results/experiment_1176_k6_and_compose_validation.json`
- DoT zero-signal evidence: `results/experiment_1171_diffusion_of_thought_inference_v1.json`
- Phase 4 pilot: `results/experiment_1165_phase4_active_inference_pilot_v1.json`
- New arXiv papers: `research-references.md` — 2026-05-02 Scan section
