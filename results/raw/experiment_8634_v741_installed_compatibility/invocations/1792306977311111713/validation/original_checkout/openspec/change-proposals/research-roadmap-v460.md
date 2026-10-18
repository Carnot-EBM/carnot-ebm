# Research Roadmap v460 — FINAL-STRETCH SPRINT (6/30 deadline boundary): maintain the locked deliverable + keep the post-6/30 verifier-moat pivot turnkey

**Milestone:** 2026.06.460
**Planned by:** outer-loop Claude Opus 4.8 planner, 2026-06-29 (UTC).
**Status of the sprint:** ARC-AGI-3 Submission Sprint Forcing Function is ACTIVE through
**2026-06-30** (CLAUDE.md). `.460` is a final-stretch sprint milestone ON the deadline boundary —
the sprint retires 2026-06-30 ONLY (not on submissions), and the post-6/30 verifier-moat pivot
fires 7/1. `.460` may well be the LAST or second-to-last sprint milestone before the forcing
function retires.

---

## 1. What the previous milestone (.459) proved

`.459` completed cleanly and re-confirmed the locked submission deliverable for the 6/30 deadline.
The capstone (exp4989) landed:

> `complete_capstone_v459_submission_ready_levels_69_heldout_0.04_package_ready_pivot_turnkey_7_1`

The honest read, arm by arm:

| Arm | Scope | Outcome |
|---|---|---|
| A1 (exp4980) | tn36 L7→L8 deepen (fresh deepest lane) | **NO BANK** — `no_grounded_l8_delta`; honest no-bank rotation dead-end |
| A2 (exp4981) | g50t L2→L3 deepen | **NO BANK** — `no_grounded_l3_delta`; honest no-bank |
| A3 (exp4982) | ft09 self-play (FR-11) | **SUCCESS** — checkpoint refreshed; honest substrate (`verifier_ensemble_against_cached_candidates`, NOT DURATION_TOO_SHORT-flagged, `flag_resolved`) |
| A4 (exp4983) | FINAL held-out go/no-go | **0.04** full-25, `flag_resolved=true` (the honest hidden-state null) |
| D (exp4984) | distributional-energy-verifier pivot | **TURNKEY** for 7/1 + backlog extended to 11 papers (added 2510.14913 / 2603.04304) |
| B1 (exp4985) | adversarial audit | banks_trustworthy + pivot_readiness_trustworthy |
| B2 (exp4986) | submission package | **READY** (peak_vram 15.146 GB < 16 GB, frozen Qwen3.5-9B-MTP iGPU stack, operator-only) |
| B3 (exp4987) | stamping + mtime window | maintained the relaxed gate (NON-zero window) |
| C (exp4988) | KV260 continuity | reachable, SSH-only |

**The two load-bearing facts for `.460`:**

1. **`reproducible_total_levels` is FLAT at 69 for the SIXTH consecutive milestone**
   (`.454` sp80/su15, `.455` lf52/sb26, `.456` ar25/vc33, `.457` tr87/s5i5, `.458` tu93/bp35,
   `.459` tn36/g50t — all no-bank). `.457`/`.458`/`.459` rotated A1 to the *deepest, least-explored*
   lanes (tr87 L6→L7, tu93 L5→L6, tn36 L7→L8), and they STILL did not bank. The deepen well is
   **comprehensively dry across all depth regimes** (L2→L3, L3→L4, and the deep L5→L6 / L6→L7 /
   L7→L8 lanes). Honest no-bank rotation dead-ends are the expected outcome — the Level-Up Attempt
   Guarantee mandates the *attempt*, not the bank, and an honest no-bank is the correct result when
   the well is dry (do NOT fabricate a bank).

2. **The deliverable is LOCKED and the path is fully de-risked.** The first-win wall is
   B1-trusted CLOSED (`WALL_IS_HIDDEN_STATE` from `.453`; representation-invariant). The
   deliverable is the **~0.05 first-win agent + the publishable FoVer paper** (`paper_ready=true`).
   The submission package is ready. The post-6/30 verifier-moat pivot is turnkey. There is nothing
   left to *build* for 6/30; `.460` *maintains and confirms* on the deadline boundary, and stages
   the clean 7/1 pivot.

**Two concluded directions (do NOT re-propose):**

- **The S0 oracle-distinct-structural-energy program is CONCLUDED** (known-issues 2026-06-26):
  the staged program S0→S0'→S1→S2-v3→S3 ran to completion — the oracle-distinct structural
  energy is a real OFFLINE cross-game discriminator but adds NO live ARC agent value
  (selection OR generation). S4 is MOOT. Do NOT re-propose energy-as-ARC-lever stages.
- **The L2-deepening wall has shifted to the dynamics ENGINE** (known-issues 2026-06-25):
  the goal-quality half is solved; the binding residual is induced-engine reachability
  (~0.12 held-out engine accuracy on lp85). This is a multi-week post-sprint direction, not
  a 6/30 deliverable; it is NOT in scope for the final-stretch sprint milestone.

---

## 2. The `.460` thesis

`.460` is a **final-stretch sprint-day milestone ON the deadline boundary** (potentially the last
sprint milestone). It does NOT chase the closed first-win wall, does NOT propose representation #5,
does NOT reopen the concluded energy-as-ARC-lever program, and does NOT open a new world-model fork
(the final-stretch discipline established across `.454`–`.459`). It:

1. **Executes the mandatory ARC majority + Level-Up Attempt Guarantee on FRESH, rotated targets.**
   Because the deepen well is dry across all regimes, A1 rotates to a fresh deepest lane
   (sc25 L5→L6 — a depth-5 game never run as an A1/A2 deepen in any rotation; lp85 L5→L6 / cn04
   L3→L4 alternates) and A2 to a fresh L2→L3 (cd82 — named in the `.459` A2 title but never
   actually run, since `.459` A2 picked g50t; m0r0 / sk48 alternates). Both are reproduction-gated,
   `live_agent_self_discovery`, with honest no-bank rotation dead-ends if the well is dry.
2. **Runs the continuous-self-learning self-play loop EVERY milestone (FR-11).** A3 rotates
   to a banked, checkpointed game (su15 — has `models/arc_verifier_su15.json`; r11l / lf52
   alternates), warm-starts the learned verifier, runs the reproduction gate, and re-trains +
   checkpoints — maintaining the `.456` honest-substrate fix (declare
   `verifier_ensemble_against_cached_candidates` when the offline gate runs without the LLM, so
   the artifact is NOT DURATION_TOO_SHORT-flagged).
3. **Produces the FINAL 6/30 held-out go/no-go** (A4) — DELIVERABLE-FIRST, `max_turns ≤ 50`,
   carrying/light-resume-confirming the settled `.459` clean full-25 0.04 + CI with full
   methodology (anti-churn; do not re-run the full 25 from scratch). This is the operator's final
   pre-deadline decision number.
4. **Keeps the post-6/30 verifier-moat pivot TURNKEY and EXTENDS the SOTA backlog** (D) with
   two NEW verified-real papers — **arXiv:2504.13134** (Energy-Based Reward Models for Robust LM
   Alignment / EBRM — an energy reward model that models the reward *distribution* with uncertainty,
   the direct sibling/foundation of the distributional-energy verifier's learned-quality-scorer
   half) and **arXiv:2605.10158** (Unsupervised Process Reward Models / uPRM — a discriminative
   process verifier with NO human labels that BEATS majority-voting / self-consistency by up to
   6.9% = the cheap-discriminative efficiency frontier + the beats-SC win condition) — re-confirming
   the eleven already-ingested papers (now 13 total). It does NOT execute the real experiment
   (majority-ARC governs through 6/30); the experiment fires the instant the sprint retires.
5. **Maintains the reserved slots:** B1 (bank + pivot-readiness adversarial audit), B2 (FINAL
   pre-deadline submission-package harden + operator checklist), B3 (stamping backfill +
   relaxed mtime window), C (KV260 SSH-only continuity), E (capstone).

**The honest framing:** `.460` produces little *new* ARC progress (the well is dry, the wall is
closed, the deliverable is locked) — and that is the correct, honest state of a final-stretch
maintenance milestone on the deadline boundary. Its genuine forward value is producing the FINAL
6/30 go/no-go + package confirmation AND keeping the post-6/30 verifier-moat pivot turnkey with an
extended, real-citation backlog (now 13 papers) so the loop pivots cleanly to the oracle-distinct
moat experiment on 7/1.

---

## 3. Architecture (unchanged; the frozen submission stack)

```
LIVE SUBMISSION STACK (FROZEN for the sprint — B2):
  generator = Qwen3.5-9B-MTP (5.9 GB Q4, Apache) on the iGPU (NEVER the 3090s)
            + MTP + q8 KV + n_predict>=2048 + /no_think
  engine    = CUDA-12.8 llama-server binary (MTP in libllama-common)
  agent     = E3AgentPolicy (arc_competition_agent.py) — verifier-routed cascade
  Kaggle parity: peak_vram 15.146 GB < 16 GB

OFFLINE INDUCTION (A1/A2/A3/A4 — 2026-06-27 GPU-allocation directive):
  conductor owns GPU 0  (CARNOT_ARC_GENERATOR_CUDA_GPU=0)  — NOT iGPU-pinned
  outer loop owns GPU 1 (CUDA_VISIBLE_DEVICES=1)
  reproduction gate = arc_solver_kit.reproduce (the executable oracle; circularity-clean)

POST-6/30 PIVOT (D — turnkey, fires 7/1):
  decomposed energy = learned LoRA-ensemble quality scorer (MEAN ranks, STDDEV abstains)
                    + analytical FoVer constraint-penalty term
  target domains    = MuSR / TravelPlanner (self_consistency NOT near-ceiling)
  validation gate   = beats SC, CI95 excl 0, oracle-distinct (verifier_is_oracle=false),
                      no model-identity shortcut
  comparator backlog (13 papers): 2605.18871 (pivot) + energy-RM sibling 2504.13134 +
                      unsupervised-PRM 2605.10158 + discriminative budget frontier 2510.14913 +
                      generative frontier 2504.00891 / 2504.16828 + compute-optimal 2504.01005 +
                      unify-gen-and-verify 2603.04304 + 2502.01989 / 2508.16665 / 2508.10539 /
                      2502.11157 / 2509.24460
```

The submission stack is FROZEN — `.460` does not re-litigate model selection (settled 2026-06-19,
[[project_arc_live_generator]]).

---

## 4. Phases & tasks (11 tasks; conductor execution order)

| # | id | Phase | Track | Agent | Deliverable |
|---|---|---|---|---|---|
| 0 | exp4990-phase0 | TRANSITION — archive .459 → activate .460 | transition | codex | results/experiment_4990_archive_459_activate_460.json |
| 1 | exp4991-a1 | A1 ARC north star — DEEPEN a fresh deepest lane (sc25 L5→L6) | arc-north-star | codex | results/experiment_4991_levelup_attempt.json |
| 2 | exp4992-a2 | A2 Level-Up Attempt Guarantee — 2nd deepen (cd82 L2→L3) | arc-north-star | codex | results/experiment_4992_levelup_attempt.json |
| 3 | exp4993-a3 | A3 self-play EVERY milestone (su15; FR-11 / continuous self-learning) | arc-north-star | codex | results/experiment_4993_self_play_verifier_checkpoint.json |
| 4 | exp4994-a4 | A4 FINAL 6/30 held-out go/no-go (deliverable-first) | arc-north-star | codex | results/experiment_4994_heldout_first_win_readiness.json |
| 5 | exp4995-d | D SOTA-ingestion / keep post-6/30 pivot turnkey (+2 papers) | sota-ingestion | codex | results/experiment_4995_distributional_energy_verifier_turnkey.json |
| 6 | exp4996-b1 | B1 bank + pivot-readiness adversarial audit | infra | codex | results/experiment_4996_bank_and_pivot_audit.json |
| 7 | exp4997-b2 | B2 FINAL submission-package harden + operator checklist | infra | codex | results/experiment_4997_submission_package_harden.json |
| 8 | exp4998-b3 | B3 stamping backfill + relaxed mtime window | infra | codex | results/experiment_4998_stamping_backfill_and_wiring_readiness.json |
| 9 | exp4999-c | C KV260 SSH-only continuity | hardware | codex | results/experiment_4999_kv260_continuity.json |
| 10 | exp5000-capstone | E CAPSTONE .460 — submission-readiness scorecard + pivot handoff | capstone | codex | results/experiment_5000_capstone_v460.json |

### Dependency graph

```
exp4990 (transition)
   |
   +--> exp4991 (A1 deepen) ----+
   +--> exp4992 (A2 deepen) ----+
   +--> exp4993 (A3 self-play)  |
   +--> exp4994 (A4 go/no-go)   |
   +--> exp4995 (D pivot) ------+
                                |
                                v
                       exp4996 (B1 audit: reads A1/A2 banks + D pivot)
                                |
   exp4997 (B2) ---------------+
   exp4998 (B3) ---------------+
   exp4999 (C) ----------------+
                                v
                       exp5000 (E capstone: counts banks only if B1 banks_trustworthy;
                                states pivot only if B1 pivot_readiness_trustworthy)
```

B1 (exp4996) is ordered AFTER the arms it audits (A1/A2/D) so their artifacts exist when it runs
(the `.454` ordering bug stays fixed). The capstone (exp5000) is last and gates bank-counting +
pivot-statement on B1's trust flags.

---

## 5. Sprint-discipline compliance (CLAUDE.md ARC-AGI-3 Submission Sprint Forcing Function)

| Discipline | How `.460` satisfies it |
|---|---|
| **Majority ARC** | A1 + A2 (deepen) + A3 (self-play) + A4 (go/no-go) + B2 (deadline package) + E (capstone), plus D (post-6/30 pivot) — the majority of the milestone is ARC live-solving / submission. |
| **≥1 Level-Up Attempt** (`arc_levelup_guarantee_lint`) | A1 + A2 are both level-bank attempts (gate requires `reproduced_levels >= prior+1`); lint passes (2 ≥ 1). |
| **Self-play EVERY milestone** (FR-11) | A3 trains + checkpoints the learned verifier (continuous self-learning). |
| **3 reserved infra** | B1 (audit), B2 (package), B3 (stamping/window). |
| **1 hardware continuity** | C (KV260, SSH-only, always-write-blocked-artifact). |
| **1 SOTA-ingestion** | D (keep pivot turnkey + extend backlog with 2 NEW verified-real papers). |
| **Agent routing** | ALL experiments `agent_type: codex` / `gpt-5.5` (`CODEX_FORCE_EXPERIMENTS=1`); planner/retro stay Claude Opus 4.8 (operator sprint quality choice, 2026-06-19). |
| **Frozen live stack** | Qwen3.5-9B-MTP on the iGPU (B2); offline induction on GPU-0 CUDA generator (2026-06-27). |
| **solve_provenance** | every ARC solve/deepen task requires `live_agent_self_discovery`; registry-precheck forbids re-solving an already-banked level (duplicate = CRITICAL). |
| **Principle-annotated fields** | every REQUIRED ARTIFACT FIELD + gate carries a `principle:` line. |
| **operator_override** | every task carries the standing 2026-06-19 ARC-sprint override (clears the exclusion-manifest + doomed-rerun guards for these false-positive scope-matched continuation tasks); `exclusion_manifest_lint` passes clean. |
| **Transition durability** | exp4990's PRETEST_COMMAND must be its OWN test file + `--no-cov` (known-issues 2026-06-29; never the full coverage-instrumented suite, which re-introduces the codex 4800s cap). |

---

## 6. Hardware requirements

- **iGPU (Radeon 890M, ROCm/HIP)** — the FROZEN live submission generator (Qwen3.5-9B-MTP); B2.
- **RTX 3090 GPU 0** — the conductor's offline ARC CUDA generator (A1/A2/A3/A4 induction, if any runs).
- **RTX 3090 GPU 1** — outer-loop experiments (not used by this milestone's tasks directly).
- **KV260** — SSH-only continuity check (`ssh kria`); NEVER a host SD-card / block-device precondition.
- No FPGA-bitstream redesign, no PolarFire/GateMate mandate this milestone (KV260 is the one
  sovereignty board kept in the per-milestone rotation; north-star §3).

---

## 7. SOTA-ingestion detail (D — the two NEW papers)

Both verified real (arXiv abstract pages confirmed 2026-06-29), both NOT previously in the backlog,
both directly on-thesis for the distributional-energy-verifier moat:

| arXiv | Title | Carnot mapping (strongest method → cost → pitfall) |
|---|---|---|
| **2504.13134** | Energy-Based Reward Models for Robust Language Model Alignment (EBRM; Lochab & Zhang) | A post-hoc framework that models the reward **distribution** explicitly (not a scalar) and captures uncertainty → the DIRECT sibling/foundation of the distributional-energy verifier's learned-quality-scorer half (validates "energy verifier as distribution-over-rewards with epistemic uncertainty"; adoptable distribution-modeling head for the FoVer ensemble). **Cost:** low (post-hoc refinement on an existing RM). **Pitfall:** it is an *alignment* RM, not a per-step process verifier — the structured-reasoning constraint-penalty term is Carnot's addition; a post-hoc refinement can inherit the base RM's blind spots. |
| **2605.10158** | Unsupervised Process Reward Models (uPRM; Gadetsky et al., May 2026) | Trains a discriminative PROCESS verifier with NO human labels (from the generator's next-token probs); as a test-time-scaling verifier it matches SUPERVISED PRMs and BEATS majority-voting (self-consistency) by up to 6.9% → squarely the cheap-discriminative efficiency frontier AND the beats-SC win condition. **Cost:** medium (needs generator next-token probs). **Pitfall:** signal derives from the GENERATOR's own probs → model-identity-shortcut risk; a strong cheap BASELINE/comparator, not itself the oracle-distinct verifier. |

The validation gate for the post-6/30 experiment (stated, NOT claimed met): the distributional
energy verifier BEATS self-consistency with CI95 EXCLUDING zero, on a domain where SC is NOT
near-ceiling (MuSR / TravelPlanner), with NO model-identity shortcut and ORACLE-DISTINCT
(`verifier_is_oracle=false`).

---

## 8. Retirement / handoff

`.460` is a final-stretch maintenance milestone on the deadline boundary. The sprint retires
**2026-06-30**; after retirement the planner resumes the verifier-moat headline (the
distributional-energy-verifier experiment, which D keeps turnkey for 7/1, with the extended
13-paper real-citation backlog: 2605.18871 pivot + 2504.13134 energy-RM sibling + 2605.10158
unsupervised-PRM + 2510.14913 discriminative budget frontier + 2603.04304 unify-gen-and-verify +
2504.01005 compute-optimal + 2504.00891 GenPRM + 2504.16828 ThinkPRM + 2509.24460 ContextPRM +
2502.01989 / 2508.16665 / 2508.10539 / 2502.11157). The deliverable remains the **~0.05 first-win
agent + the publishable FoVer paper**. The first-win wall stays B1-trusted CLOSED; the deepen well
is dry; do NOT reopen either or queue representation #5 / the concluded energy-as-ARC-lever program.
The next genuine headline is the oracle-distinct verifier moat, NOT more ARC level-banking. The
first post-sprint roadmap (the milestone that falls after the deadline) should pivot the majority to
the distributional-energy-verifier head-to-head, reporting against BOTH the discriminative budget
frontier (2510.14913) / unsupervised-PRM baseline (2605.10158) and the generative frontier
(2504.00891 / 2504.16828) on a SC-not-saturated domain (MuSR / TravelPlanner), with the validation
gate (beats SC, CI95 excl 0, oracle-distinct, no model-identity shortcut).
