# Research Roadmap v457 — FINAL SPRINT DAY (6/30 deadline): execute the locked deliverable + keep the post-6/30 verifier-moat pivot turnkey

**Milestone:** 2026.06.457
**Planned by:** outer-loop Claude Opus 4.8 planner, 2026-06-29 (UTC).
**Status of the sprint:** ARC-AGI-3 Submission Sprint Forcing Function is ACTIVE through
**2026-06-30** (CLAUDE.md). `.457` is the final-stretch sprint milestone on the last day
before the deadline.

---

## 1. What the previous milestone (.456) proved

`.456` executed the locked submission deliverable for the 6/30 deadline. The capstone
(exp4956) landed clean:

> `complete_capstone_v456_submission_ready_levels_69_heldout_0.04_package_ready_pivot_turnkey_7_1`

The honest read, arm by arm:

| Arm | Scope | Outcome |
|---|---|---|
| A1 (exp4947) | ar25 L3→L4 deepen | **NO BANK** — `no_grounded_l4_delta`; reproduced_levels stayed 3 |
| A2 (exp4948) | vc33 L2→L3 deepen | **NO BANK** — `no_grounded_l3_delta`; reproduced_levels stayed 2 |
| A3 (exp4949) | lp85 self-play (FR-11) | **SUCCESS** — checkpoint refreshed, **substrate bug FIXED** (`flag_resolved=true`; the `.455` exp4938 DURATION_TOO_SHORT critical flag is gone) |
| A4 (exp4950) | FINAL held-out go/no-go | **0.04** full-25, `flag_resolved=true` (the honest hidden-state null) |
| D (exp4951) | distributional-energy-verifier pivot | **TURNKEY** for 7/1 (`pivot_turnkey=true`, `pivot_executable_on_7_1=true`) |
| B1 (exp4952) | adversarial audit | **trusted** (no banks to count; `pivot_readiness_trustworthy=true`) |
| B2 (exp4953) | submission package | **READY** (peak_vram 15.146 GB < 16 GB, frozen Qwen3.5-9B-MTP iGPU stack, operator-only) |
| B3 (exp4954) | stamping + mtime window | **FIXED** — relaxed gate (n_arms≥7) emits a NON-zero window (the `.454/.455` `blocked_insufficient` false-block is gone) |
| C (exp4955) | KV260 continuity | reachable, overlay `carnot_ising_v2_n64` |

**The two load-bearing facts for `.457`:**

1. **`reproducible_total_levels` is FLAT at 69 for the THIRD consecutive milestone**
   (`.454` sp80/su15 no-bank, `.455` lf52/sb26 no-bank, `.456` ar25/vc33 no-bank). The
   L2→L3 / L3→L4 deepen well is **dry** on the recently-rotated targets. Honest no-bank
   rotation dead-ends are the expected outcome — the Level-Up Attempt Guarantee mandates
   the *attempt*, not the bank, and an honest no-bank is the correct result when the well
   is dry (do NOT fabricate a bank).

2. **The deliverable is LOCKED and the path is de-risked.** The first-win wall is
   B1-trusted CLOSED (`WALL_IS_HIDDEN_STATE` from `.453`; representation-invariant — the
   hidden variable is winning-prefix-order state). The deliverable is the **~0.05 first-win
   agent + the publishable FoVer paper** (`paper_ready=true`). The submission package is
   ready. The post-6/30 verifier-moat pivot is turnkey. There is nothing left to *build*
   for 6/30; `.457` *confirms and maintains*.

The two recurring infra bugs that plagued `.454/.455` (A3 substrate mismatch, B3
window-too-strict) are BOTH fixed as of `.456`. `.457` keeps them fixed.

---

## 2. The `.457` thesis

`.457` is the **final sprint-day milestone**. It does NOT chase the closed first-win wall,
does NOT propose representation #5, and does NOT open any new world-model fork (the
final-stretch discipline established across `.454/.455/.456`). It:

1. **Executes the mandatory ARC majority + Level-Up Attempt Guarantee on FRESH, more-plausible
   targets.** Because the L2→L3 / L3→L4 well is dry on the hammered games, A1 rotates to the
   **deepest, least-explored lanes** (tu93 L5→L6 / tr87 L6→L7 / tn36 L7→L8 — 0 recorded
   deepen dead-ends each = the freshest headroom that exists), with ft09 L3→L4 as an
   adaptered alternate. A2 covers the Level-Up Attempt Guarantee floor with a fresh L2→L3
   (cd82). Both are reproduction-gated, `live_agent_self_discovery`, and record an honest
   no-bank rotation dead-end if dry.
2. **Runs self-play (A3) every milestone** — the continuous-self-learning / FR-11 loop, on
   a fresh banked+checkpointed game (dc22), with the `.456`-fixed honest substrate
   declaration maintained.
3. **Produces the operator's FINAL 6/30 go/no-go** (A4) by carrying/confirming the settled
   clean full-25 held-out first-win 0.04 (deliverable-first, anti-churn — no fresh full-25
   re-run).
4. **Keeps the post-6/30 verifier-moat pivot TURNKEY** (D) and **EXTENDS the SOTA-ingestion
   backlog** with the next verifier-moat papers (survey arXiv:2508.16665, variance-reduction
   arXiv:2508.10539, fast/slow Dyve arXiv:2502.11157) so the loop pivots into a rich method
   backlog the instant the sprint retires on 6/30.
5. **Maintains the reserved slots** — B1 adversarial audit (banks + pivot), B2 FINAL package
   harden, B3 stamping + relaxed mtime window, C KV260 continuity — and aggregates the FINAL
   submission-readiness scorecard (E).

**Do NOT re-propose** (all nulled/retired): representation #5, energy-as-ARC-lever (program
CONCLUDED 2026-06-26), macro/horizon-collapse, click-heatmap generator, trust-gate flip,
MATM similarity-retrieval (NULLED `.454`), TTT-on-code-engine, local code inducers,
decision-need targets, action-prefix latents, coverage/exploration/selection/
perception-from-grid.

---

## 3. Architecture (unchanged; the frozen submission stack)

```
                 ARC-AGI-3 LIVE AGENT (the scored deliverable — FROZEN for the sprint)
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  arc_competition_agent.py : make_carnot_agent -> E3AgentPolicy             │
   │    per-action verifier-routed cascade over the agent's OWN transitions:    │
   │    StepwiseExplorer -> online world-model induction (arc_live_ttt /        │
   │    LocalGGUFProposer, gated by WorldModelVerifier) -> e3.plan_in_model     │
   │  generator: Qwen3.5-9B-MTP on the iGPU  (MTP + q8 KV + n_predict>=2048 +   │
   │             /no_think)  — Kaggle-parity, peak_vram 15.146GB < 16GB         │
   └──────────────────────────────────────────────────────────────────────────┘
                              ▲  (registry / dev twin)
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  arc_loop_solve.py : OfflineSolver + GameAdapter (offline development twin)│
   │    raw-action verifier-routed best-first search in the offline sim,        │
   │    reproduction-gated (arc_solver_kit.reproduce); trains+checkpoints the   │
   │    learned verifier (models/arc_verifier_<game>.json) — FR-11 self-play    │
   └──────────────────────────────────────────────────────────────────────────┘

   OFFLINE induction (A1/A2/A3/A4) runs on the CONDUCTOR's dedicated GPU-0 CUDA
   generator (2026-06-27 GPU-allocation directive) — NOT iGPU-pinned. The iGPU-only
   constraint applies to the LIVE SUBMISSION stack (B2) only.

   POST-6/30 PIVOT (turnkey, fires 7/1):
   distributional-energy-verifier (arXiv:2605.18871) = learned quality-scorer LoRA
   ensemble on one frozen encoder (~3% params; MEAN ranks, STDDEV abstains) + the FoVer
   analytical constraint-penalty ensemble. Oracle-distinct (verifier_is_oracle=false) by
   design; validation gate = beats SC with CI95 excl 0 on an SC-not-saturated domain, no
   model-identity shortcut.
```

---

## 4. Phases & tasks (11 tasks; conductor execution order)

| # | id | Phase | Track | Agent | What |
|---|----|-------|-------|-------|------|
| 1 | exp4957 | PHASE 0 | transition | codex | archive `.456` → activate `.457`; record close-state; resolve any poison pre-test |
| 2 | exp4958 | A1 | arc-north-star | codex | **DEEP-lane** deepen (tu93 L5→L6 / tr87 L6→L7 / tn36 L7→L8 preferred; ft09 L3→L4 alt) — freshest headroom; +1 level or honest no-bank |
| 3 | exp4959 | A2 | arc-north-star | codex | **Level-Up Attempt Guarantee** — fresh L2→L3 (cd82 / s5i5 / g50t); +1 level or honest no-bank |
| 4 | exp4960 | A3 | arc-north-star | codex | **self-play / FR-11** — warm-start + train + checkpoint on a banked game (dc22 / lf52 / sb26); honest substrate |
| 5 | exp4961 | A4 | arc-north-star | codex | **FINAL 6/30 go/no-go** — carry/confirm the clean full-25 held-out 0.04 (deliverable-first, anti-churn) |
| 6 | exp4962 | D | sota-ingestion | codex | keep the post-6/30 verifier-moat pivot TURNKEY + INGEST the next backlog papers (2508.16665 / 2508.10539 / 2502.11157) |
| 7 | exp4963 | B1 | infra | codex | adversarial audit of A1/A2 banks + D pivot-turnkey (placed AFTER A1/A2/D) |
| 8 | exp4964 | B2 | infra | codex | FINAL pre-deadline submission-package harden + operator checklist (`submits=false`) |
| 9 | exp4965 | B3 | infra | codex | stamping backfill + relaxed mtime-window (maintain the `.456` fix) |
| 10 | exp4966 | C | hardware | codex | KV260 SSH-only continuity check |
| 11 | exp4967 | E | capstone | codex | FINAL submission-readiness scorecard + post-6/30 verifier-moat handoff |

### Dependency graph

```
exp4957 (PHASE 0 transition)
    │
    ├── exp4958 (A1 deep deepen)  ─┐
    ├── exp4959 (A2 fresh L2→L3)  ─┤
    ├── exp4960 (A3 self-play)     │
    ├── exp4961 (A4 go/no-go)      │
    ├── exp4962 (D pivot turnkey) ─┤
    │                              ▼
    │                       exp4963 (B1 audit: reads A1/A2/D artifacts)
    ├── exp4964 (B2 package harden)
    ├── exp4965 (B3 stamping/window)
    ├── exp4966 (C KV260)
    │                              ▼
    └────────────────────►  exp4967 (E capstone: reads all upstream; counts banks
                                     only if B1 banks_trustworthy; states pivot only
                                     if B1 pivot_readiness_trustworthy)
```

B1 is intentionally ordered AFTER A1/A2/D so it can read their artifacts (the `.454`
ordering bug, fixed in `.455/.456`, stays fixed).

---

## 5. Sprint-discipline compliance (CLAUDE.md ARC-AGI-3 Submission Sprint Forcing Function)

- **Majority-ARC:** 6 of 11 tasks are ARC/deadline (A1, A2, A3, A4, B2 deadline package, E
  capstone) + the D post-6/30 pivot = the ARC value stream dominates the non-reserved slots.
- **≥1 Level-Up Attempt Guarantee:** A1 + A2 are both banking attempts (the
  `arc_levelup_guarantee_lint.py` floor; ≥2 attempts across two depth regimes).
- **Self-play every milestone:** A3 (FR-11 continuous self-learning).
- **3 reserved infra:** B1 (audit), B2 (package), B3 (stamping/window).
- **1 hardware continuity:** C (KV260, SSH-only).
- **1 SOTA/pivot:** D (post-6/30 verifier-moat turnkey + SOTA-ingestion).
- **Agent routing:** all experiments `agent_type: codex` / `gpt-5.5`; planner + retro stay
  Claude Opus 4.8 (operator's sprint quality choice). All ARC/infra/SOTA tasks carry a
  standing-directive `operator_override:` (legit continuation; clears both the
  exclusion-manifest and doomed-rerun guards per CLAUDE.md).

## 6. Hardware requirements

- **OFFLINE induction (A1/A2/A3/A4):** conductor's dedicated GPU-0 CUDA `llama-server`
  (`CUDA_VISIBLE_DEVICES=0`, `CARNOT_ARC_GENERATOR_CUDA_GPU=0`) OR the iGPU HIP server —
  do NOT iGPU-pin or hard-reject `CUDA_VISIBLE_DEVICES` (the 2026-06-27 GPU-allocation
  directive; most deepens are offline search + reproduction gate with no LLM).
- **LIVE submission stack (B2):** FROZEN to Qwen3.5-9B-MTP on the iGPU (Kaggle parity,
  ~16 GB).
- **KV260 (C):** SSH-reachable board (`ssh kria`), SSH-only precondition (NEVER a host
  SD-card / block-device check).

## 7. Retirement / handoff

The ARC-AGI-3 Submission Sprint Forcing Function retires **2026-06-30** (the challenge
deadline) ONLY — not on submissions. The instant it retires, the loop pivots to the
**post-6/30 verifier-moat headline**: the distributional-energy-verifier experiment
(arXiv:2605.18871, which beats self-consistency on MuSR = the oracle-distinct,
SC-not-saturated win) runs from D's turnkey one-command entrypoint, against the validation
gate (beats SC with CI95 excl 0 + oracle-distinct + no model-identity shortcut). The
deliverable for the deadline remains the ~0.05 first-win agent + the publishable FoVer
paper.
