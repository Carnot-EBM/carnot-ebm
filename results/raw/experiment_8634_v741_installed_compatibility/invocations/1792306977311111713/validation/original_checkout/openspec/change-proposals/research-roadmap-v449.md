# Research Roadmap — Milestone 2026.06.449

**Planned by:** outer-loop Claude Opus 4.8 planner, 2026-06-27.
**Milestone doc:** this file.
**Theme:** **UNBLOCK the generation-wall fork (the GPU-precondition bug the 2026-06-27
operator directive fixes), measure it for real, then take the FIRST concrete swing at
the indicated root cause — the induced world-model inducer.**

The `.448 headline (A1, exp4861) was supposed to settle the L1-first-contact generation
wall by forking it — *is the winning multi-step prefix never assembled because the search
has no GUIDING signal (a planner gap, buildable), or because our air-gapped weak inducer
cannot induce an accurate-enough world model to plan through (a structural ceiling)?* It
**never ran the science.** It `blocked_generator_unavailable` on a **wrong GPU
precondition**: the experiment hard-rejected any `CUDA_VISIBLE_DEVICES` (`detail:
cuda_3090_generator_disallowed`, `igpu_required: true`) while the conductor — per the
**2026-06-27 operator GPU re-allocation** (conductor → GPU 0, outer loop → GPU 1) —
correctly offered its dedicated GPU-0 generator. B1 (exp4865) correctly classified A1 a
**non-test**; the capstone (exp4869) recorded `complete_a1_generation_wall_non_test`. The
fork is **still unmeasured**.

This milestone runs inside the **ARC-AGI-3 Submission Sprint** (CLAUDE.md forcing
function, active through **2026-06-30** — the deadline is ~3 days out). So `.449 both (a)
**unblocks and finally measures** the fork on the conductor's GPU-0 generator, and (b)
takes the **first real swing** at the strongly-indicated root cause — improving the
induced executable world-model's held-out transition accuracy via a counterexample-guided
(CEGIS) refinement loop — while still driving the realistic scored levers (a level-up
bank, self-play checkpointing, a *genuinely live* held-out first-win readiness number for
the 6/30 go/no-go, and final submission-package hardening).

---

## 1. What the previous milestone (.448) proved

Read from the capstone (exp4869) and its audited upstreams:

| Phase | Result | Reading |
|---|---|---|
| **A1 — generation-wall fork probe (HEADLINE)** | **BLOCKED — non-test.** `blocked_generator_unavailable`, root cause `cuda_3090_generator_disallowed` / `igpu_required: true` while the conductor launched `CUDA_VISIBLE_DEVICES=1`. `n_games_measured=0`, `fork_verdict=null`, `flagged_adversarial: true` (DURATION_TOO_SHORT — it never invoked the model). B1 (exp4865) confirmed non-diagnostic (`a1_genuinely_diagnostic: false`; positive control not migrated; live-path unreachable). | **The fork was never measured.** Pure infrastructure failure — the precondition wrongly vetoed the conductor's own dedicated GPU-0 generator. This is the *exact* incident named in CLAUDE.md's 2026-06-27 GPU-allocation rule ("cf. the `.448` exp4861 fork-probe non-test"). |
| **A2 — level-up attempt (scored lever)** | **PASS — banked r11l L2** (+1) via the click-template handle-average L2 delta. `reproducible_total_levels: 65 → 66`. `solve_provenance: live_agent_self_discovery`. | First bank in several milestones; the monotonic ARC metric moved. |
| **A3 — self-play (every milestone)** | **PASS.** Learned verifier checkpoint refreshed (`models/arc_verifier_re86.json`), re86 reproduction gate passed (`success_self_play_checkpoint_refreshed`). | The self-learning loop (train + checkpoint the learned verifier, FR-11) keeps working. |
| **A4 — held-out first-win readiness (deadline signal)** | **FLAT NULL at 0.04** (= 1/25), but **`live_agent_ran: false`** — it resumed from cache and did NOT run live. Positive control passed, parity green. | The 6/30 go/no-go number is stale (a cache hit, not a fresh live measurement). `.449 must run it GENUINELY LIVE. |
| **B2 — submission package** | **READY.** `vram_estimate_gb: 15.146` (< 16 GB Kaggle), builds, operator-only, no regression vs `.447. | The frozen Qwen3.5-9B-MTP stack packages for submission; the operator can submit. |
| **C — KV260 hardware** | **REACHABLE / graduated terminal.** 5 UIO devices, `success_kv260_continuity_ok`. | Keep in the per-milestone SSH-only continuity rotation. |
| **D — SOTA ingestion (.449 frontier)** | Mapped the operator-reserved **INDUCER_CEILING** branch (A1 was blocked, so D mapped the *likely* branch with that caveat): 3 inducer-quality tracks — **Family-B executable world-model inducer ladder** (2605.05138 + 2507.03160 + 2203.13474), **test-time world-model adaptation** (2506.02918 + 2509.03956 + 2507.15877), **CEGIS world-model refinement** (2606.11521 + 2502.07786 + 2507.15877). | "`.449 should target INDUCER_CEILING: improve the executable world-model inducer before adding planner/search complexity." `.449 picks up exactly this. |

**The independent evidence that points the same way (load-bearing).** Two separate
results, *outside* the blocked A1, already indicate the wall is the **inducer**:

- **`.437 exp4750** (the perception-grounded structural-alignment L2 goal): the detector
  over-segmentation IS fixed (`detector_goal_count` 42→2), but lp85 L2 still did not bank —
  the binding residual **shifted to ENGINE-reachability**: "the agent cannot PLAN to a
  correct goal because the induced dynamics ENGINE is wrong (free-form LLM engine held-out
  accuracy **~0.12** on lp85)." Operator note: "the next lever is the dynamics ENGINE /
  world-model accuracy."
- **exp4749** (the structured `ProductWorldModel` direction): nulled as a **dead/identity
  engine** — a more-structured engine did not become a more-*accurate* engine.

So the strongly-indicated fork is **INDUCER_CEILING** (low engine held-out accuracy → no
planning through it). `.449 measures it directly (A1) *and* attacks it (A1b).

**What is CONCLUDED / RETIRED (do NOT re-propose — these are forward-closed):**

- **The energy-as-ARC-lever program is CONCLUDED (negative), 2026-06-26 operator-directed.**
  S0→S0'→S1→S2-v3→S3 ran in full: the oracle-distinct structural energy is a real *offline*
  cross-game discriminator but adds **NO live ARC agent value** (selection OR generation).
  S4 is MOOT. Do NOT re-propose energy stages.
  (`docs/research-notes/oracle-distinct-structural-energy-program-2026-06-26.md`.)
- **Coverage / vocabulary levers** — macro-action vocabulary induction (`horizon collapse,
  guidance not depth`) and click-heatmap-as-generator (`premise falsified, guidance not
  coverage`) are RETIRED with empirical nulls. The primitive vocabulary is already
  *sufficient*; NEVER_ENUMERATED means the right sequence is never **assembled**, not that a
  primitive is missing. Do NOT re-propose any coverage/vocabulary lever.
- **Exploration-strategy** levers (NGU/RND/Go-Explore directed-exploration) and
  **perception-from-grid** are proven nulls (`.434/`.446). Do NOT re-propose.
- **Selection / ranking** levers (value-head, verifier-router, persistent-AEM, trust-energy
  gate, energy-fitness QD) all transfer-null on live solve-rate: they reorder a pool that
  does not contain the winner. The wall is **generation**, not selection.

---

## 2. The three biggest gaps (current state vs. the north star)

The north star (`ops/north-star.md` §0) is **solve ARC-AGI-3 accurately and efficiently**
as a LIVE agent that discovers hidden games at submission time. Against that:

1. **The L1-first-contact GENERATION wall is unbroken.** Generic first-win is **0.04**
   (1/25 games). The `.447 diagnostic localized it (NEVER_ENUMERATED — the winning prefix is
   never assembled), but the *cause* (planner gap vs inducer ceiling) is still unmeasured
   because A1 blocked. **Gap 1: measure the fork, then move it.** This is the headline.

2. **The induced world-model inducer is the indicated bottleneck and has never been
   *improved* (only swapped).** Free-form LLM induction nulls at ~0.12 held-out accuracy;
   a structured engine nulled as identity. No experiment has tried to **iteratively repair**
   the induced engine from its own held-out mispredictions. **Gap 2: the first inducer-
   accuracy lever — CEGIS refinement** (the most executable-native, fully air-gapped of the
   D candidates).

3. **The deadline go/no-go signal is stale.** `.448's held-out first-win readiness was a
   cache hit (`live_agent_ran: false`); the scored submission sits at ~0.08 with the
   diversity floor shipped. **Gap 3: a genuinely-live 6/30 readiness number + a final
   submission-package confirmation.**

---

## 3. The 2026-06-27 GPU directive — the precise fix (the headline unblock)

CLAUDE.md (ARC sprint, 2026-06-27): *"offline induction tasks must NOT hardcode an
`igpu_required: True` / `cuda_3090_generator_disallowed` precondition — that was sourced
from the live-stack constraint and WRONGLY blocks the conductor's own dedicated GPU-0
generator (cf. the `.448 exp4861 fork-probe non-test). Offline induction uses GPU 0
(conductor) / GPU 1 (outer loop); only the LIVE submission generator is iGPU-pinned."*

The mechanism already exists and is correct — only the experiment's *self-veto* is wrong:

- `arc_executable_world_model._generator_server_and_env()` already returns the **local CUDA
  llama-server pinned to GPU 0** when `CARNOT_ARC_GENERATOR_CUDA_GPU=0` is set (the conductor's
  systemd drop-in `40-arc-generator-3090-20260619.conf`) **and** the card has ≥13 GB free
  (both 3090s currently show ~24 GB free). It yields to the iGPU if the card is busy.
- The **bug** is in `experiment_4861_generation_wall_fork_probe.py:generator_available()`:
  it returns `ok: False, detail: cuda_3090_generator_disallowed` the moment
  `launch_env.get("CUDA_VISIBLE_DEVICES")` is set — i.e. it rejects exactly the generator the
  conductor offers. It also requires the iGPU HIP server.

**The `.449 fix (A1, A1b, A4):** the precondition must **accept** either the iGPU HIP server
**or** the conductor's GPU-0 CUDA server, set `igpu_required: False`, and health-check via
`_ensure_server()`. This is an **experiment-script** change only — it does NOT touch
`scripts/research_conductor.py` or the live submission stack (which stays iGPU-pinned for
Kaggle parity).

---

## 4. Milestone architecture (11 tasks)

```
                       .448 close-state (fork UNMEASURED; A1 blocked on GPU precond)
                                         │
   ┌── phase0 (exp4870) ── archive .448 → activate .449; record close-state ──┐
   │                                                                          │
   ▼                                                                          │
 ARC NORTH STAR (majority) ────────────────────────────────────────────────  │
   A1  (exp4871) HEADLINE  fork probe RE-RUN, GPU-FIXED (conductor GPU-0)      │
        │  measures the fork + per-game induced-engine held-out accuracy      │
        │                                                                     │
        ├─ gated_on median_engine_heldout_accuracy < 0.5  (INDUCER regime)    │
        ▼                                                                     │
   A1b (exp4872)  FIRST INDUCER SWING — CEGIS executable-world-model          │
                  refinement: repair the engine from its OWN held-out         │
                  mispredictions; re-measure held-out accuracy delta          │
   A2  (exp4873)  LEVEL-UP GUARANTEE — bank ≥1 new level (rotated target)     │
   A3  (exp4874)  SELF-PLAY (every milestone) — train + checkpoint verifier   │
   A4  (exp4875)  DEADLINE — held-out first-win readiness, GENUINELY LIVE     │
                                                                              │
 RESERVED SLOTS ───────────────────────────────────────────────────────────  │
   B1  (exp4876) INFRA-1  adversarial audit: A1 ran live on GPU-0 (not        │
                          iGPU-blocked again); A1b delta is real (held-out)   │
   B2  (exp4877) INFRA-2  submission-package FINAL harden (6/30; never submits)│
   C   (exp4878) HARDWARE KV260 SSH-only continuity (always write artifact)   │
   D   (exp4879) SOTA     .450 frontier per A1's ACTUAL fork verdict          │
                                                                              │
   E   (exp4880) CAPSTONE aggregate; headline = the measured fork + A1b delta ◄┘
```

**Slot accounting (ARC Submission Sprint Forcing Function):** majority ARC (A1, A1b, A2,
A3, A4 — 5 of 11) + Level-Up Attempt Guarantee (A2; rotates off `.448 r11l / re86) +
self-play every milestone (A3 = continuous self-learning / FR-11) + 2 reserved infra (B1,
B2) + 1 hardware continuity (C, KV260, SSH-only, always-write-blocked-artifact) + 1
SOTA-ingestion (D, `.450 frontier per the measured fork) + capstone (E) + transition
(phase0). Live stack FROZEN to Qwen3.5-9B-MTP; **offline induction (A1/A1b/A4) runs on the
conductor's dedicated GPU-0 CUDA generator** per the 2026-06-27 directive — the headline
unblock.

---

## 5. Phase descriptions

### Phase 0 — transition (exp4870)
Archive `.448 → activate `.449. Record the close-state: A1 **blocked on the GPU precondition
bug** (fork unmeasured); A2 banked r11l L2 (`reproducible_total_levels = 66`); A3 self-play
pass; A4 flat-0.04 **cache** (not live); B2 package ready (15.146 GB); energy program
CONCLUDED; INDUCER_CEILING strongly-indicated by exp4750/exp4749. Resolve any poison
pre-test. `aggregation_from_upstream_artifacts`.

### Phase A1 — HEADLINE: generation-wall fork probe, GPU-FIXED (exp4871)
Re-run the `.448 fork probe with the generator precondition corrected to accept the
conductor's **GPU-0 CUDA** generator (per §3). For the NEVER_ENUMERATED held-out games
(blind to the banked answer), induce → `plan_in_model` from cold-start transitions and emit
the JOINT FORK: induced-engine **held-out transition accuracy** × **coverage migration**
(NEVER_ENUMERATED → COVERED), with **tu93 as the positive control** (MUST come out HIGH
accuracy + COVERED). Verdict: `GUIDANCE_WALL` (migration with a good engine →`.450 builds a
planner) / `PLANNER_GAP` (good engine, no migration) / `INDUCER_CEILING` (low accuracy, no
migration → the engine is the wall). `verifier_is_oracle: true` (the reproduction gate is
the oracle — a grounding measurement, not a moat). `solve_provenance: development_proxy`.
The decisive diagnostic that redirects `.450. `prior_failures:` → exp4861 (blocked on GPU
precondition; addressed by the GPU-0 fix) + the free-form engine 0.12 null.

### Phase A1b — FIRST INDUCER SWING: CEGIS world-model refinement (exp4872)
**Gated on A1's `median_engine_heldout_accuracy < 0.5`** (the INDUCER_CEILING regime where
the intervention applies; skips cleanly if A1 blocked or the engine is already accurate).
Implement a **counterexample-guided (CEGIS) refinement loop** over the *executable* induced
world model: take A1's held-out **mispredicted** (s,a,s′) transitions as minimal failing
tests, ask the LLM inducer to **repair** the engine program (not re-induce from scratch),
accept a repair only when it fixes held-out counterexamples **without** regressing
observed-prefix replay, and **re-measure held-out transition accuracy vs A1's baseline**.
Live-path-reachable (it improves the live `e3` induction/repair path — NOT a parallel
solver; `arc_orphan_solver_lint` must pass). The first genuine attempt to *move* the ~0.12
engine accuracy. `solve_provenance: development_proxy`; `verifier_is_oracle: false` (the
held-out transition score is oracle-distinct from the env's level-up check).
`prior_failures:` → the free-form engine 0.12 null + exp4749 ProductWorldModel identity-null
(addressed by *iterative repair from counterexamples*, not one-shot induction).

### Phase A2 — Level-Up Attempt Guarantee (exp4873)
Bank ≥1 new reproducible level on a **rotated** target (rotate off `.448 r11l + re86). Pick a
shallow game with a *grounded* next-level delta via `recommend_approach` + `dead_ends`
(avoid the hidden-state-bound dead-ends ka59/wa30 and the no-grounded-delta rotations).
Reproduction-gated (`arc_solver_kit.reproduce`); `solve_provenance: live_agent_self_discovery`.

### Phase A3 — self-play, every milestone (exp4874)
Standing `arc_loop_solve` on a banked game (warm-started from the saved checkpoint):
verifier-routed solve → reproduction gate → **train + checkpoint** the learned verifier. The
continuous-self-learning / FR-11 experiment. `solve_provenance: live_agent_self_discovery`.

### Phase A4 — deadline go/no-go, GENUINELY LIVE (exp4875)
Re-run the held-out first-win readiness on the exp4605 variant harness **genuinely live**
(`live_agent_ran: true` REQUIRED — `.448 was a cache hit), on the conductor's GPU-0 generator
(drop the iGPU pin, same fix as A1). Report the rate + CI vs the 0.04 baseline and the
prior-best — the realistic 6/30 signal. Checkpoint/resume wall-clock-safe (the 2026-06-25
fix); a capped run emits a usable partial. For a flat null, `positive_control_passed` +
`null_delta_methodology_note`. `solve_provenance: development_proxy`.

### Phase B1 — INFRA-1: adversarial audit of A1 + A1b (exp4876)
The Phase-Prototype+Validation adversarial check. Verify A1 ran **genuinely live on the
GPU-0 generator** (NOT iGPU-blocked / not duration-too-short again), planner-blind (no banked
answer seeded), tu93 positive control genuinely migrated, numbers match the fork verdict,
live-path-reachable. Verify A1b's accuracy delta is **real** — measured on **held-out**
transitions (not the repair set), not a tautology, repairs did not just memorize the
counterexample row. `aggregation_from_upstream_artifacts`.

### Phase B2 — INFRA-2: submission package FINAL harden (exp4877)
Re-verify the Kaggle ARC-AGI-3 package builds and the frozen Qwen3.5-9B-MTP stack fits
~16 GB; diff against the `.448 ready package (regression check); produce the FINAL operator
submission checklist. **Never submits** (Operator-Only External Publication; no credentials).

### Phase C — hardware KV260 (exp4878)
SSH-reachability continuity (SSH ONLY — host SD-card device nodes permanently retired).
**Always write the deliverable** — `blocked_kv260_ssh_unreachable` on failure (never exit
with no file changes → 3-fail-skip). `hardware_smoke`.

### Phase D — SOTA ingestion, `.450 frontier (exp4879)
Given A1's **actual** (now-measured) fork verdict, ingest the matching SOTA. **INDUCER_CEILING**
(likely) → deepen the inducer ladder A1b did not take (Family-B reference inducer for ceiling
measurement, test-time dynamics adaptation, local open-code inducer A/B — and the
SWE-strong-but-cloud vs local-sovereign tension). **GUIDANCE/PLANNER** → pivot to
neural-guided planning / MCTS / search-verifier over the executable world model. Reliable
channel only (`sweep_clusters.py` / `sweep_semscholar.py` + low-concurrency WebSearch/WebFetch;
NO `/deep-research`). Real arXiv IDs only. Do NOT re-ingest the nulled coverage / exploration
/ energy classes.

### Phase E — capstone `.449 (exp4880)
Aggregate the scorecard. **Headline = the (finally-measured) fork verdict** (trusted only if
B1 confirmed A1 ran live + planner-blind + positive-control-migrated + numbers-match) **+
A1b's held-out-accuracy delta** (did CEGIS move the inducer?). Also report the scored deadline
levers (A2 bank, A3 self-play, A4 fresh-live readiness, B2 package). Skip any
`flagged_adversarial` upstream. `aggregation_from_upstream_artifacts`.

---

## 6. Dependency graph

```
phase0 ─▶ A1 ─▶ A1b (gated: A1.median_engine_heldout_accuracy < 0.5)
           │
           └────────────────────────────┐
A2 ─ A3 ─ A4 ─ B2 ─ C  (independent)     │
                                         ▼
A1, A1b ─────────────────────────────▶ B1 (audits A1 + A1b)
A1 ──────────────────────────────────▶ D  (frontier follows A1's fork)
all ─────────────────────────────────▶ E  (capstone)
```

## 7. Hardware requirements

| Resource | Used by | Notes |
|---|---|---|
| **Conductor GPU-0 (RTX 3090) CUDA generator** | A1, A1b, A4 | Qwen3.5-9B-MTP via the local CUDA llama-server pinned to GPU 0 (`CARNOT_ARC_GENERATOR_CUDA_GPU=0`, the 2026-06-27 drop-in; ~24 GB free now). The headline unblock — offline induction is NOT iGPU-pinned. |
| iGPU (AMD Radeon 890M) | — | Reserved for the LIVE submission stack only (Kaggle parity). NOT required for offline induction this milestone. |
| KV260 (SSH) | C | `ssh kria` reachability; graduated terminal, continuity-only. |
| Offline arcade (`arc_solver_kit.offline_arcade()`) | A1, A1b, A2, A3, A4 | Deterministic offline sim, all 25 games, zero quota. |

## 8. The fork → `.450 redirect (what this milestone decides)

- **INDUCER_CEILING** (low engine accuracy, no migration — strongly indicated): `.450
  continues the inducer ladder. If A1b's CEGIS moved held-out accuracy, scale it; else take
  the next D candidate (test-time dynamics adaptation, then Family-B/local-open-code inducer).
- **GUIDANCE_WALL / PLANNER_GAP** (high engine accuracy): `.450 builds a guided planner /
  neural-guided search over the (accurate) executable world model — A1b's CEGIS would have
  shown ~0 delta (nothing to repair), confirming the engine is not the wall.
- **A1 blocks AGAIN** (should not, with the GPU fix; both 3090s are idle): the fork-probe
  approach retires (`retire_if_same_verdict: true`) and `.450 forks the wall differently
  (operator escalation) — but B1 will pinpoint whether it was the precondition or a genuine
  generator outage.

---

*Discipline checklist:* every task `agent_type: codex` / `gpt-5.5` (ARC sprint; planner +
retro stay Opus). Every `honest_verdict` uses a terminal prefix
(`complete_`/`success_`/`blocked_`). Every compute-bound task has a PRECONDITIONS step 0 and
declares `inference_substrate`. Every REQUIRED ARTIFACT FIELD is principle-annotated. ARC
solve tasks declare `solve_provenance`. A1/A1b/A2/A3 carry `prior_failures:` (rerun
discipline) + the standing ARC-sprint `operator_override:`. No energy / coverage /
exploration / perception-from-grid re-proposals (all closed). Do NOT modify
`scripts/research_conductor.py`. Do NOT push.
