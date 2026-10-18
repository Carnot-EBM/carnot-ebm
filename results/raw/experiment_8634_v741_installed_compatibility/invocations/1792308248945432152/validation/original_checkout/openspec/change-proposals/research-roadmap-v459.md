# Research Roadmap v459 — FINAL-STRETCH SPRINT (6/30 deadline boundary): maintain the locked deliverable + keep the post-6/30 verifier-moat pivot turnkey

**Milestone:** 2026.06.459
**Planned by:** outer-loop Claude Opus 4.8 planner, 2026-06-29 (UTC).
**Status of the sprint:** ARC-AGI-3 Submission Sprint Forcing Function is ACTIVE through
**2026-06-30** (CLAUDE.md). `.459` is a final-stretch sprint milestone ON the deadline boundary —
likely the LAST or second-to-last sprint milestone before the forcing function retires. The sprint
retires 2026-06-30 ONLY (not on submissions), and the post-6/30 verifier-moat pivot fires 7/1.

---

## 1. What the previous milestone (.458) proved

`.458` re-confirmed the locked submission deliverable for the 6/30 deadline. The capstone
(exp4978) is expected to land in the same shape as `.457`:

> `complete_capstone_v458_submission_ready_levels_69_heldout_0.04_package_ready_pivot_turnkey_7_1`

The honest read, arm by arm:

| Arm | Scope | Outcome |
|---|---|---|
| A1 (exp4969) | tu93 L5→L6 deepen (fresh deep lane) | **NO BANK expected** — the deepen well is dry across all regimes; honest no-bank rotation dead-end |
| A2 (exp4970) | bp35 L2→L3 deepen | **NO BANK expected** — `no_grounded_l3_delta`; honest no-bank |
| A3 (exp4971) | sp80 self-play (FR-11) | **SUCCESS** — checkpoint refreshed; honest substrate maintained (NOT DURATION_TOO_SHORT-flagged) |
| A4 (exp4972) | FINAL held-out go/no-go | **0.04** full-25, `flag_resolved=true` (the honest hidden-state null) |
| D (exp4973) | distributional-energy-verifier pivot | **TURNKEY** for 7/1 + backlog extended (2504.01005 / 2504.00891 / 2509.24460) |
| B1 (exp4974) | adversarial audit | banks_trustworthy + pivot_readiness_trustworthy |
| B2 (exp4975) | submission package | **READY** (peak_vram 15.146 GB < 16 GB, frozen Qwen3.5-9B-MTP iGPU stack, operator-only) |
| B3 (exp4976) | stamping + mtime window | maintained the relaxed gate (NON-zero window) |
| C (exp4977) | KV260 continuity | reachable, SSH-only |

**The two load-bearing facts for `.459`:**

1. **`reproducible_total_levels` is FLAT at 69 for the FIFTH consecutive milestone**
   (`.454` sp80/su15, `.455` lf52/sb26, `.456` ar25/vc33, `.457` tr87/s5i5, `.458` tu93/bp35 —
   all no-bank). `.457` and `.458` rotated A1 to the *deepest, least-explored* lanes (tr87 L6→L7,
   tu93 L5→L6), and they STILL did not bank. The deepen well is **comprehensively dry across all
   depth regimes** (L2→L3, L3→L4, and the deep L5→L6 / L6→L7 / L7→L8 lanes). Honest no-bank
   rotation dead-ends are the expected outcome — the Level-Up Attempt Guarantee mandates the
   *attempt*, not the bank, and an honest no-bank is the correct result when the well is dry (do
   NOT fabricate a bank).

2. **The deliverable is LOCKED and the path is fully de-risked.** The first-win wall is
   B1-trusted CLOSED (`WALL_IS_HIDDEN_STATE` from `.453`; representation-invariant). The
   deliverable is the **~0.05 first-win agent + the publishable FoVer paper**
   (`paper_ready=true`). The submission package is ready. The post-6/30 verifier-moat pivot
   is turnkey. There is nothing left to *build* for 6/30; `.459` *maintains and confirms* on the
   deadline boundary, and stages the clean 7/1 pivot.

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

## 2. The `.459` thesis

`.459` is a **final-stretch sprint-day milestone ON the deadline boundary** (potentially the last
sprint milestone). It does NOT chase the closed first-win wall, does NOT propose representation #5,
does NOT reopen the concluded energy-as-ARC-lever program, and does NOT open a new world-model fork
(the final-stretch discipline established across `.454/.455/.456/.457/.458`). It:

1. **Executes the mandatory ARC majority + Level-Up Attempt Guarantee on FRESH, rotated targets.**
   Because the deepen well is dry across all regimes, A1 rotates to a fresh deepest lane
   (tn36 L7→L8 — the deepest game at 7 levels, 0 recorded deepen dead-ends, NOT yet attempted at
   this depth; cn04 L3→L4 / ft09 L3→L4 alternates) and A2 to a fresh L2→L3 (cd82; g50t / re86
   alternates). Both are reproduction-gated, `live_agent_self_discovery`, with honest no-bank
   rotation dead-ends if the well is dry.
2. **Runs the continuous-self-learning self-play loop EVERY milestone (FR-11).** A3 rotates
   to a banked, checkpointed game (ft09 — has `models/arc_verifier_ft09.json`; su15 / ls20
   alternates), warm-starts the learned verifier, runs the reproduction gate, and re-trains +
   checkpoints — maintaining the `.456` honest-substrate fix (declare
   `verifier_ensemble_against_cached_candidates` when the offline gate runs without the LLM, so
   the artifact is NOT DURATION_TOO_SHORT-flagged).
3. **Produces the FINAL 6/30 held-out go/no-go** (A4) — DELIVERABLE-FIRST, `max_turns ≤ 50`,
   carrying/light-resume-confirming the settled clean full-25 0.04 + CI with full methodology
   (anti-churn; do not re-run the full 25 from scratch). This is the operator's final pre-deadline
   decision number.
4. **Keeps the post-6/30 verifier-moat pivot TURNKEY and EXTENDS the SOTA backlog** (D) with
   two NEW real papers — arXiv:2510.14913 (Budget-aware Test-time Scaling via Discriminative
   Verification — the matched-compute *discriminative*-verifier-under-budget comparator = the
   efficiency-parity frontier, since the decomposed-energy verifier is itself discriminative) and
   arXiv:2603.04304 (V1: Unifying Generation and Self-Verification for Parallel Reasoners — the
   2026 unify-generation-and-self-verification comparator for the regenerate/abstain two-pass
   loop) — re-confirming the nine already-ingested papers. It does NOT execute the real experiment
   (majority-ARC governs through 6/30); the experiment fires the instant the sprint retires.
5. **Maintains the reserved slots:** B1 (bank + pivot-readiness adversarial audit), B2 (FINAL
   pre-deadline submission-package harden + operator checklist), B3 (stamping backfill +
   relaxed mtime window), C (KV260 SSH-only continuity), E (capstone).

**The honest framing:** `.459` produces little *new* ARC progress (the well is dry, the wall is
closed, the deliverable is locked) — and that is the correct, honest state of a final-stretch
maintenance milestone on the deadline boundary. Its genuine forward value is producing the FINAL
6/30 go/no-go + package confirmation AND keeping the post-6/30 verifier-moat pivot turnkey with an
extended, real-citation backlog (now 11 papers) so the loop pivots cleanly to the oracle-distinct
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
  comparator backlog (11 papers): 2605.18871 (pivot) + discriminative frontier 2510.14913 +
                      generative frontier 2504.00891 / 2504.16828 + compute-optimal 2504.01005 +
                      unify-gen-and-verify 2603.04304 + 2502.01989 / 2508.16665 / 2508.10539 /
                      2502.11157 / 2509.24460
```

The submission stack is FROZEN — `.459` does not re-litigate model selection (settled 2026-06-19,
[[project_arc_live_generator]]).

---

## 4. Phases & tasks (11 tasks; conductor execution order)

| # | id | Phase | Track | Agent | Deliverable |
|---|---|---|---|---|---|
| 0 | exp4979-phase0 | TRANSITION — archive .458 → activate .459 | transition | codex | results/experiment_4979_archive_458_activate_459.json |
| 1 | exp4980-a1 | A1 ARC north star — DEEPEN a fresh deepest lane (level bank) | arc-north-star | codex | results/experiment_4980_levelup_attempt.json |
| 2 | exp4981-a2 | A2 Level-Up Attempt Guarantee — 2nd deepen (L2→L3) | arc-north-star | codex | results/experiment_4981_levelup_attempt.json |
| 3 | exp4982-a3 | A3 self-play EVERY milestone (FR-11 / continuous self-learning) | arc-north-star | codex | results/experiment_4982_self_play_verifier_checkpoint.json |
| 4 | exp4983-a4 | A4 FINAL 6/30 held-out go/no-go (deliverable-first) | arc-north-star | codex | results/experiment_4983_heldout_first_win_readiness.json |
| 5 | exp4984-d | D SOTA-ingestion / keep post-6/30 pivot turnkey | sota-ingestion | codex | results/experiment_4984_distributional_energy_verifier_turnkey.json |
| 6 | exp4985-b1 | B1 bank + pivot-readiness adversarial audit | infra | codex | results/experiment_4985_bank_and_pivot_audit.json |
| 7 | exp4986-b2 | B2 FINAL submission-package harden + operator checklist | infra | codex | results/experiment_4986_submission_package_harden.json |
| 8 | exp4987-b3 | B3 stamping backfill + relaxed mtime window | infra | codex | results/experiment_4987_stamping_backfill_and_wiring_readiness.json |
| 9 | exp4988-c | C KV260 SSH-only continuity | hardware | codex | results/experiment_4988_kv260_continuity.json |
| 10 | exp4989-capstone | E CAPSTONE .459 — submission-readiness scorecard + pivot handoff | capstone | codex | results/experiment_4989_capstone_v459.json |

### Dependency graph

```
exp4979 (transition)
   |
   +--> exp4980 (A1 deepen) ----+
   +--> exp4981 (A2 deepen) ----+
   +--> exp4982 (A3 self-play)  |
   +--> exp4983 (A4 go/no-go)   |
   +--> exp4984 (D pivot) ------+
                                |
                                v
                       exp4985 (B1 audit: reads A1/A2 banks + D pivot)
                                |
   exp4986 (B2) ---------------+
   exp4987 (B3) ---------------+
   exp4988 (C) ----------------+
                                v
                       exp4989 (E capstone: counts banks only if B1 banks_trustworthy;
                                states pivot only if B1 pivot_readiness_trustworthy)
```

B1 (exp4985) is ordered AFTER the arms it audits (A1/A2/D) so their artifacts exist when it runs
(the `.454` ordering bug stays fixed). The capstone (exp4989) is last and gates bank-counting +
pivot-statement on B1's trust flags.

---

## 5. Sprint-discipline compliance (CLAUDE.md ARC-AGI-3 Submission Sprint Forcing Function)

| Discipline | How `.459` satisfies it |
|---|---|
| **Majority ARC** | A1 + A2 (deepen) + A3 (self-play) + A4 (go/no-go) + B2 (deadline package) + E (capstone), plus D (post-6/30 pivot) — the majority of the milestone is ARC live-solving / submission. |
| **≥1 Level-Up Attempt** (`arc_levelup_guarantee_lint`) | A1 + A2 are both level-bank attempts (gate requires `reproduced_levels >= prior+1`). |
| **Self-play EVERY milestone** (FR-11) | A3 trains + checkpoints the learned verifier (continuous self-learning). |
| **3 reserved infra** | B1 (audit), B2 (package), B3 (stamping/window). |
| **1 hardware continuity** | C (KV260, SSH-only, always-write-blocked-artifact). |
| **1 SOTA-ingestion** | D (keep pivot turnkey + extend backlog with 2 NEW real papers). |
| **Agent routing** | ALL experiments `agent_type: codex` / `gpt-5.5` (`CODEX_FORCE_EXPERIMENTS=1`); planner/retro stay Claude Opus 4.8 (operator sprint quality choice, 2026-06-19). |
| **Frozen live stack** | Qwen3.5-9B-MTP on the iGPU (B2); offline induction on GPU-0 CUDA generator (2026-06-27). |
| **solve_provenance** | every ARC solve/deepen task requires `live_agent_self_discovery`; registry-precheck forbids re-solving an already-banked level (duplicate = CRITICAL). |
| **Principle-annotated fields** | every REQUIRED ARTIFACT FIELD + gate carries a `principle:` line. |
| **operator_override** | every task carries the standing 2026-06-19 ARC-sprint override (clears the exclusion-manifest + doomed-rerun guards for these false-positive scope-matched continuation tasks). |

---

## 6. Hardware requirements

- **iGPU (Radeon 890M, ROCm/HIP)** — the FROZEN live submission generator (Qwen3.5-9B-MTP); B2.
- **RTX 3090 GPU 0** — the conductor's offline ARC CUDA generator (A1/A2/A3/A4 induction, if any runs).
- **RTX 3090 GPU 1** — outer-loop experiments (not used by this milestone's tasks directly).
- **KV260** — SSH-only continuity check (`ssh kria`); NEVER a host SD-card / block-device precondition.
- No FPGA-bitstream redesign, no PolarFire/GateMate mandate this milestone (KV260 is the one
  sovereignty board kept in the per-milestone rotation; north-star §3).

---

## 7. Retirement / handoff

`.459` is the final-stretch maintenance milestone on the deadline boundary. The sprint retires
**2026-06-30**; after retirement the planner resumes the verifier-moat headline (the
distributional-energy-verifier experiment, which D keeps turnkey for 7/1, with the extended
real-citation backlog: 2605.18871 pivot + 2510.14913 discriminative budget frontier + 2603.04304
unify-gen-and-verify + 2504.01005 compute-optimal allocation + 2504.00891 GenPRM + 2504.16828
ThinkPRM + 2509.24460 ContextPRM + 2502.01989 / 2508.16665 / 2508.10539 / 2502.11157). The
deliverable remains the **~0.05 first-win agent + the publishable FoVer paper**. The first-win wall
stays B1-trusted CLOSED; the deepen well is dry; do NOT reopen either or queue representation #5 /
the concluded energy-as-ARC-lever program. The next genuine headline is the oracle-distinct verifier
moat, NOT more ARC level-banking. The first post-sprint roadmap (`.460+`, if it falls after the
deadline) should pivot the majority to the distributional-energy-verifier head-to-head, reporting
against BOTH the discriminative budget frontier (2510.14913) and the generative frontier
(2504.00891 / 2504.16828) on a SC-not-saturated domain (MuSR / TravelPlanner), with the validation
gate (beats SC, CI95 excl 0, oracle-distinct, no model-identity shortcut).
