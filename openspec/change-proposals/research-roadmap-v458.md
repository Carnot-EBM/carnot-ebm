# Research Roadmap v458 — FINAL-STRETCH SPRINT (6/30 deadline): maintain the locked deliverable + keep the post-6/30 verifier-moat pivot turnkey

**Milestone:** 2026.06.458
**Planned by:** outer-loop Claude Opus 4.8 planner, 2026-06-29 (UTC).
**Status of the sprint:** ARC-AGI-3 Submission Sprint Forcing Function is ACTIVE through
**2026-06-30** (CLAUDE.md). `.458` is a final-stretch sprint milestone on the deadline boundary;
the sprint retires 2026-06-30 and the post-6/30 verifier-moat pivot fires 7/1.

---

## 1. What the previous milestone (.457) proved

`.457` re-confirmed the locked submission deliverable for the 6/30 deadline. The capstone
(exp4967) landed clean:

> `complete_capstone_v457_submission_ready_levels_69_heldout_0.04_package_ready_pivot_turnkey_7_1`

The honest read, arm by arm:

| Arm | Scope | Outcome |
|---|---|---|
| A1 (exp4958) | tr87 L6→L7 deepen (deep lane) | **NO BANK** — `no_grounded_l7_delta`; reproduced_levels stayed 6 |
| A2 (exp4959) | s5i5 L2→L3 deepen | **NO BANK** — `no_grounded_l3_delta`; reproduced_levels stayed 2 |
| A3 (exp4960) | dc22 self-play (FR-11) | **SUCCESS** — `success_self_play_checkpoint_refreshed`; substrate fix held (NOT DURATION_TOO_SHORT-flagged) |
| A4 (exp4961) | FINAL held-out go/no-go | **0.04** full-25, `flag_resolved=true` (the honest hidden-state null) |
| D (exp4962) | distributional-energy-verifier pivot | **TURNKEY** for 7/1 + backlog extended (2508.16665 / 2508.10539 / 2502.11157) |
| B1 (exp4963) | adversarial audit | **trusted** (no banks to count; `pivot_readiness_trustworthy=true`) |
| B2 (exp4964) | submission package | **READY** (peak_vram 15.146 GB < 16 GB, frozen Qwen3.5-9B-MTP iGPU stack, operator-only) |
| B3 (exp4965) | stamping + mtime window | maintained the relaxed gate (NON-zero window) |
| C (exp4966) | KV260 continuity | reachable, SSH-only |

**The two load-bearing facts for `.458`:**

1. **`reproducible_total_levels` is FLAT at 69 for the FOURTH consecutive milestone**
   (`.454` sp80/su15, `.455` lf52/sb26, `.456` ar25/vc33, `.457` tr87/s5i5 — all no-bank).
   `.457` rotated A1 to the *deepest, least-explored* lanes (tr87 L6→L7), and it STILL did
   not bank. The deepen well is now **comprehensively dry across all depth regimes** (L2→L3,
   L3→L4, and the deep L6→L7 / L7→L8 lanes). Honest no-bank rotation dead-ends are the
   expected outcome — the Level-Up Attempt Guarantee mandates the *attempt*, not the bank,
   and an honest no-bank is the correct result when the well is dry (do NOT fabricate a bank).

2. **The deliverable is LOCKED and the path is fully de-risked.** The first-win wall is
   B1-trusted CLOSED (`WALL_IS_HIDDEN_STATE` from `.453`; representation-invariant). The
   deliverable is the **~0.05 first-win agent + the publishable FoVer paper**
   (`paper_ready=true`). The submission package is ready. The post-6/30 verifier-moat pivot
   is turnkey. There is nothing left to *build* for 6/30; `.458` *maintains and confirms*.

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

## 2. The `.458` thesis

`.458` is a **final-stretch sprint-day milestone** on the deadline boundary. It does NOT chase
the closed first-win wall, does NOT propose representation #5, does NOT reopen the concluded
energy-as-ARC-lever program, and does NOT open a new world-model fork (the final-stretch
discipline established across `.454/.455/.456/.457`). It:

1. **Executes the mandatory ARC majority + Level-Up Attempt Guarantee on FRESH, rotated targets.**
   Because the deepen well is dry across all regimes, A1 rotates to a fresh deep lane
   (tu93 L5→L6 — not yet attempted at this depth; tn36 L7→L8 / cn04 L3→L4 alternates) and A2
   to a fresh L2→L3 (bp35; m0r0 / g50t alternates). Both are reproduction-gated,
   `live_agent_self_discovery`, with honest no-bank rotation dead-ends if the well is dry.
2. **Runs the continuous-self-learning self-play loop EVERY milestone (FR-11).** A3 rotates
   to a banked, checkpointed game (sp80; ft09 / su15 alternates), warm-starts the learned
   verifier, runs the reproduction gate, and re-trains + checkpoints — maintaining the `.456`
   honest-substrate fix (declare `verifier_ensemble_against_cached_candidates` when the
   offline gate runs without the LLM, so the artifact is NOT DURATION_TOO_SHORT-flagged).
3. **Produces the FINAL 6/30 held-out go/no-go** (A4) — DELIVERABLE-FIRST, `max_turns ≤ 50`,
   carrying/light-resume-confirming the settled clean full-25 0.04 + CI with full methodology
   (anti-churn; do not re-run the full 25 from scratch).
4. **Keeps the post-6/30 verifier-moat pivot TURNKEY and EXTENDS the SOTA backlog** (D) with
   three NEW real papers — arXiv:2504.01005 (When To Solve When To Verify — the compute-optimal
   efficiency frontier), 2504.00891 (GenPRM — generative process verifier comparator), and
   2509.24460 (ContextPRM — multi-domain cross-domain comparator) — re-confirming the six
   already-ingested papers. It does NOT execute the real experiment (majority-ARC governs
   through 6/30); the experiment fires the instant the sprint retires.
5. **Maintains the reserved slots:** B1 (bank + pivot-readiness adversarial audit), B2 (FINAL
   pre-deadline submission-package harden + operator checklist), B3 (stamping backfill +
   relaxed mtime window), C (KV260 SSH-only continuity), E (capstone).

**The honest framing:** `.458` produces little *new* ARC progress (the well is dry, the wall is
closed, the deliverable is locked) — and that is the correct, honest state of a final-stretch
maintenance milestone. Its genuine forward value is keeping the post-6/30 verifier-moat pivot
turnkey with an extended, real-citation backlog so the loop pivots cleanly to the oracle-distinct
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
```

The submission stack is FROZEN — `.458` does not re-litigate model selection (settled 2026-06-19,
[[project_arc_live_generator]]).

---

## 4. Phases & tasks (11 tasks; conductor execution order)

| # | id | Phase | Track | Agent | Deliverable |
|---|---|---|---|---|---|
| 0 | exp4968-phase0 | TRANSITION — archive .457 → activate .458 | transition | codex | results/experiment_4968_archive_457_activate_458.json |
| 1 | exp4969-a1 | A1 ARC north star — DEEPEN a fresh deep lane (level bank) | arc-north-star | codex | results/experiment_4969_levelup_attempt.json |
| 2 | exp4970-a2 | A2 Level-Up Attempt Guarantee — 2nd deepen (L2→L3) | arc-north-star | codex | results/experiment_4970_levelup_attempt.json |
| 3 | exp4971-a3 | A3 self-play EVERY milestone (FR-11 / continuous self-learning) | arc-north-star | codex | results/experiment_4971_self_play_verifier_checkpoint.json |
| 4 | exp4972-a4 | A4 FINAL 6/30 held-out go/no-go (deliverable-first) | arc-north-star | codex | results/experiment_4972_heldout_first_win_readiness.json |
| 5 | exp4973-d | D SOTA-ingestion / keep post-6/30 pivot turnkey | sota-ingestion | codex | results/experiment_4973_distributional_energy_verifier_turnkey.json |
| 6 | exp4974-b1 | B1 bank + pivot-readiness adversarial audit | infra | codex | results/experiment_4974_bank_and_pivot_audit.json |
| 7 | exp4975-b2 | B2 FINAL submission-package harden + operator checklist | infra | codex | results/experiment_4975_submission_package_harden.json |
| 8 | exp4976-b3 | B3 stamping backfill + relaxed mtime window | infra | codex | results/experiment_4976_stamping_backfill_and_wiring_readiness.json |
| 9 | exp4977-c | C KV260 SSH-only continuity | hardware | codex | results/experiment_4977_kv260_continuity.json |
| 10 | exp4978-capstone | E CAPSTONE .458 — submission-readiness scorecard + pivot handoff | capstone | codex | results/experiment_4978_capstone_v458.json |

### Dependency graph

```
exp4968 (transition)
   |
   +--> exp4969 (A1 deepen) ----+
   +--> exp4970 (A2 deepen) ----+
   +--> exp4971 (A3 self-play)  |
   +--> exp4972 (A4 go/no-go)   |
   +--> exp4973 (D pivot) ------+
                                |
                                v
                       exp4974 (B1 audit: reads A1/A2 banks + D pivot)
                                |
   exp4975 (B2) ---------------+
   exp4976 (B3) ---------------+
   exp4977 (C) ----------------+
                                v
                       exp4978 (E capstone: counts banks only if B1 banks_trustworthy;
                                states pivot only if B1 pivot_readiness_trustworthy)
```

B1 (exp4974) is ordered AFTER the arms it audits (A1/A2/D) so their artifacts exist when it runs
(the `.454` ordering bug stays fixed). The capstone (exp4978) is last and gates bank-counting +
pivot-statement on B1's trust flags.

---

## 5. Sprint-discipline compliance (CLAUDE.md ARC-AGI-3 Submission Sprint Forcing Function)

| Discipline | How `.458` satisfies it |
|---|---|
| **Majority ARC** | A1 + A2 (deepen) + A3 (self-play) + A4 (go/no-go) + B2 (deadline package) + E (capstone), plus D (post-6/30 pivot) — the majority of the milestone is ARC live-solving / submission. |
| **≥1 Level-Up Attempt** (`arc_levelup_guarantee_lint`) | A1 + A2 are both level-bank attempts (gate requires `reproduced_levels >= prior+1`). |
| **Self-play EVERY milestone** (FR-11) | A3 trains + checkpoints the learned verifier (continuous self-learning). |
| **3 reserved infra** | B1 (audit), B2 (package), B3 (stamping/window). |
| **1 hardware continuity** | C (KV260, SSH-only, always-write-blocked-artifact). |
| **1 SOTA-ingestion** | D (keep pivot turnkey + extend backlog with 3 NEW real papers). |
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

`.458` is a final-stretch maintenance milestone. The sprint retires **2026-06-30**; after retirement
the planner resumes the verifier-moat headline (the distributional-energy-verifier experiment, which
D keeps turnkey for 7/1, with the extended real-citation backlog: 2605.18871 pivot + 2504.01005
efficiency frontier + 2504.00891 GenPRM comparator + 2509.24460 ContextPRM cross-domain comparator +
2508.16665 / 2508.10539 / 2502.11157 / 2504.16828 / 2502.01989). The deliverable remains the **~0.05
first-win agent + the publishable FoVer paper**. The first-win wall stays B1-trusted CLOSED; the
deepen well is dry; do NOT reopen either or queue representation #5 / the concluded energy-as-ARC-lever
program. The next genuine headline is the oracle-distinct verifier moat, NOT more ARC level-banking.
