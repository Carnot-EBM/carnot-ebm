# Research Roadmap — Milestone 2026.06.451

**Planned by:** outer-loop Claude Opus 4.8 planner, 2026-06-27.
**Milestone doc:** this file.
**Theme:** **The executable-code world-model engine is a change-VALUE ceiling.
Attack the gap by REPLACING the world-model REPRESENTATION (the `.451 frontier the
.450 SOTA ingestion mapped), not by adapting the same code engine again.**

The `.450 headline (A1, exp4882) settled the dynamics-engine question that `.449's
metric-broken fork probe could not: on the corrigendum-corrected GRADED metric
(non-degenerate tu93 positive control, B1-audited `a1_genuinely_diagnostic: true`),
the induced **executable-code** world model predicts change-**LOCATION** well
(`engine_cell_recall_median = 0.727`) but **NOT** change-**VALUE**, and **test-time
dynamics adaptation does NOT close the value gap** (`tta_changed_cell_value_accuracy_
delta_median = -0.0087`, CI95 `[-0.178, 0.0]`). Fork verdict: **INDUCER_CEILING_HARD**.
The agent knows *where* cells change but not *what they become*, so `plan_in_model`
cannot assemble a winning multi-step prefix — the exact L1-first-contact generation
wall (generic first-win **0.04** = 1/25).

`.450's A1b (exp4883, the Family-B-vs-local inducer A/B that was meant to attribute the
ceiling to MODEL vs METHOD) was a **fabrication-flagged non-test** (`DURATION_TOO_SHORT`,
13.7 s — too fast to have run two live induction lanes; B1 `a1b_ab_trustworthy: false`).
So the METHOD-vs-MODEL attribution is **NOT established**. The `.450 SOTA ingestion (D,
exp4890) nonetheless mapped the **METHOD_IS_CEILING** branch — *alternative world-model
representations beyond executable code* — and that direction is the right one to test
*on its own merits*: A1 proved the code engine + TTA cannot predict change-VALUE, so
`.451 asks whether **a different representation can**. Testing alternative representations
IS the proper test of the METHOD ceiling (which A1b failed to run): if a non-code
representation closes the value gap, the executable-code METHOD was the ceiling and the
new representation is the first-win lever; if structurally-different representations
*also* fail, the change-VALUE gap is representation-invariant — a deep ceiling, and the
capstone escalates to the operator with the current 0.08 agent as the deliverable.

This milestone runs inside the **ARC-AGI-3 Submission Sprint** (CLAUDE.md forcing
function, active through **2026-06-30** — the deadline is ~3 days out). So `.451 (a) takes
the headline swing at the change-VALUE gap via the strongest `.451-frontier representation
(agent-authored decision-need targets), (b) keeps a structurally-different representation
in reserve (action-prefix latent adapter), and (c) drives the realistic scored levers (a
level-up bank, self-play checkpointing, a *genuinely-live* held-out first-win number for
the 6/30 go/no-go, final submission-package hardening).

---

## 1. What the previous milestone (.450) proved

Read from the audited upstreams + `results/operational_retro_2026_06_450.json`:

| Phase | Result | Reading |
|---|---|---|
| **A1 — TTA change-VALUE attack (HEADLINE)** | **INDUCER_CEILING_HARD, trustworthy.** Graded metric, `engine_cell_recall_median = 0.727` (change-LOCATION learnable), `tta_changed_cell_value_accuracy_delta_median = -0.0087`, CI95 `[-0.178, 0]` over 9 NEVER_ENUMERATED games + tu93 positive control. `a1_genuinely_diagnostic: true` (non-degenerate control, held-out-disjoint delta, planner-blind, numbers-match-fork, live on `gpu0_cuda`, 168 s, not flagged). | **The decisive, trustworthy finding.** The executable-code engine + test-time adaptation predict *where* cells change but not *what they become*. This is the diagnosed cause of the generation wall. |
| **A1b — inducer-ceiling A/B (MODEL vs METHOD)** | **FABRICATION-FLAGGED NON-TEST.** `complete_inducer_ceiling_neither_lane_lifts_method_is_ceiling` but `DURATION_TOO_SHORT` (13.7 s); B1 `a1b_ab_trustworthy: false`. | **The MODEL-vs-METHOD attribution is UNCONFIRMED.** The two live induction lanes did not genuinely run. `.451 must NOT trust `METHOD_IS_CEILING` as established — it tests it directly. |
| **A2 — level-up (scored lever)** | **PASS — banked g50t L2** (+1) via the `config_toggle_target_offset` delta. `reproducible_total_levels: 67 → 68`. `solve_provenance: live_agent_self_discovery`. | The monotonic ARC metric moved; the one clean win of `.450. |
| **A3 — self-play (every milestone)** | **PASS.** Learned-verifier checkpoint refreshed, reproduction gate green. | The self-learning loop (train + checkpoint the learned verifier, FR-11) keeps working. |
| **A4 — held-out first-win readiness (deadline)** | **PARTIAL ~0.0625, CI lower 0** (soft-budget partial, `live_agent_ran: true`), but **flagged** (`TAUTOLOGY` + `METHODOLOGY_MISSING`: no `model_specs` / `random_seed`). | Still a **flat null** vs the 0.04 baseline (CI includes 0). `.451 must run it clean: add `model_specs` + `random_seed`, report the delta honestly. |
| **B2 — submission package** | **READY.** `vram ~15.146 GB` (< 16 GB Kaggle), builds, operator-only. | The frozen Qwen3.5-9B-MTP stack packages for submission; operator can submit. |
| **C — KV260 hardware** | **REACHABLE / graduated terminal.** 5 UIO devices, `success_kv260_continuity_ok`. | Keep in the per-milestone SSH-only continuity rotation. |
| **D — SOTA ingestion (.451 frontier)** | Mapped **METHOD_IS_CEILING → alternative world-model representations beyond executable code**: (1) **agent-authored decision-need targets** (arXiv:2606.25421), (2) **action-prefix latent adapter** (arXiv:2606.26217 + 2603.19312), (3) latent-action interface (arXiv:2503.18938), (4) reverse counterfactual targeter (arXiv:2505.08073), (5) verification-calibrated abstraction (arXiv:2602.23997). All HTTP-200 verified; `research-references.md` / `research-studying.md` updated. | `.451 picks up frontier candidates (1) + (2) — the two strongest, structurally-different representations. |

**The independent evidence that points the same way (load-bearing).** The change-VALUE
ceiling is corroborated *outside* A1: `.437 exp4750 found lp85 L2 still did not bank after
the detector over-segmentation was fixed — the residual shifted to **engine reachability**
(free-form LLM engine held-out accuracy **~0.12**); and exp4749's structured
`ProductWorldModel` nulled as a **dead/identity engine** — a more-*structured* engine was
not a more-*accurate* engine. Three independent attempts on the **executable-code**
representation (free-form induction, structured product model, TTA) all hit the same
change-VALUE ceiling. `.451 changes the representation class, not the inducer.

**What is CONCLUDED / RETIRED (do NOT re-propose — forward-closed):**

- **The energy-as-ARC-lever program is CONCLUDED (negative), 2026-06-26 operator-directed.**
  The oracle-distinct structural energy is a real *offline* cross-game discriminator but adds
  **NO live ARC agent value** (selection OR generation). Do NOT re-propose energy stages.
- **Test-time dynamics adaptation on the executable-code engine** — nulled in `.450 A1
  (delta CI95 includes 0). Do NOT re-propose TTA-on-the-code-engine as the lever; `.451
  changes the *representation*, which is a different intervention.
- **Family-B-vs-local-open-code inducer A/B + stronger local code inducers** — the `.450 D
  mapping explicitly excludes "stronger local open-code inducers" because A1 + the
  exp4750/4749 corroboration show the ceiling is the executable-code *representation*, not
  the inducer's code-writing strength. (A1b's MODEL-vs-METHOD attribution was a non-test,
  but the representation-change test `.451 runs subsumes it.)
- **Coverage / vocabulary levers** (macro-action vocabulary, click-heatmap generator) —
  RETIRED with empirical nulls (`guidance not depth`, `guidance not coverage`).
- **Exploration-strategy** (NGU/RND/Go-Explore) and **perception-from-grid** — proven nulls.
- **Selection / ranking** levers (value-head, verifier-router, persistent-AEM, trust-energy
  gate, energy-fitness QD) all transfer-null on live solve-rate: they reorder a pool that
  does not contain the winner. The wall is **generation**, not selection.

---

## 2. The three biggest gaps (current state vs. the north star)

The north star (`ops/north-star.md` §0) is **solve ARC-AGI-3 accurately and efficiently**
as a LIVE agent that discovers hidden games at submission time. Against that:

1. **The change-VALUE gap blocks generation, and only ONE representation has been tried.**
   The executable-code engine knows *where* cells change but not *what they become*, so
   `plan_in_model` cannot assemble a winning prefix → generic first-win **0.04**. Three
   attempts on the *code* representation all failed. **Gap 1: try a fundamentally different
   representation** (agent-authored decision-need targets). This is the headline.

2. **No structurally-different second representation has been tested.** If a single
   alternative representation (A1) fails, we cannot conclude the gap is representation-
   invariant from one data point. **Gap 2: a second, structurally-different representation**
   (action-prefix latent adapter) — both to give the milestone two shots and to make
   "representation-invariant" a real (two-representation) conclusion if both fail.

3. **The deadline go/no-go signal is still a flat-null PARTIAL with quality flags.** `.450's
   held-out first-win was ~0.0625 (CI lower 0), `flagged_adversarial` (TAUTOLOGY +
   METHODOLOGY_MISSING). The scored submission sits at **0.08**. **Gap 3: a clean,
   headline-eligible 6/30 readiness number** (add `model_specs` + `random_seed`, honest
   delta) + a final submission-package confirmation.

---

## 3. The 2026-06-27 GPU directive (carried forward; the offline-induction unblock)

CLAUDE.md (ARC sprint, 2026-06-27): *"offline induction tasks must NOT hardcode an
`igpu_required: True` / `cuda_3090_generator_disallowed` precondition … Offline induction
uses GPU 0 (conductor) / GPU 1 (outer loop); only the LIVE submission generator is
iGPU-pinned."* The `.450 A1/A1b/A4 already ran correctly on `gpu0_cuda` (A1 168 s,
`generator_backend=gpu0_cuda`). `.451 keeps this:

- **Offline induction (A1, A1b, A4)** runs on the conductor's **GPU-0 CUDA generator**
  (`CARNOT_ARC_GENERATOR_CUDA_GPU=0`, the `40-arc-generator-3090-20260619.conf` drop-in;
  accept GPU-0 CUDA **or** the iGPU HIP server; health-check via `_ensure_server()`; do NOT
  block merely because `CUDA_VISIBLE_DEVICES` is set; do NOT iGPU-pin).
- **The LIVE submission stack (B2)** stays **FROZEN** to Qwen3.5-9B-MTP on the **iGPU**
  (Kaggle ~16 GB parity). This is an experiment-script discipline only — it does NOT touch
  `scripts/research_conductor.py`.

**`.450 retro action #3 carried into `.451 (load-bearing):** two of four `.450 compute-bound
arms were `flagged_adversarial` (A1b = DURATION_TOO_SHORT 13.7 s; A4 = TAUTOLOGY +
METHODOLOGY_MISSING). `.451 fixes both at the source: A1b MUST genuinely run live (>60 s,
checkpointed, real induction over ≥3 games); A4 MUST carry `model_specs` + `random_seed` and
report the delta without a self-referential tautology.

---

## 4. Milestone architecture (11 tasks)

```
                    .450 close-state (A1=INDUCER_CEILING_HARD trustworthy;
                       A1b non-test; code-engine change-VALUE ceiling)
                                         │
   ┌── phase0 (exp4891) ── archive .450 → activate .451; record close-state ──┐
   │                                                                          │
   ▼                                                                          │
 ARC NORTH STAR (majority) ────────────────────────────────────────────────  │
   A1  (exp4892) HEADLINE  agent-authored DECISION-NEED targets (2606.25421): │
        │   does a NON-code representation predict change-VALUE where the     │
        │   code engine + TTA could not? Same A1-graded held-out gate.        │
        │   emits BARE decision_need_value_accuracy_delta_median              │
        ├─ gated_on decision_need_value_accuracy_delta_median < 0.1           │
        ▼   (the value-gap-not-closed regime)                                 │
   A1b (exp4893)  SECOND REPRESENTATION — action-prefix LATENT adapter        │
                  (2606.26217 + 2603.19312); MUST run genuinely live (>60 s,  │
                  the .450 A1b 13.7 s non-test fix). Same A1-graded gate.     │
   A2  (exp4894)  LEVEL-UP GUARANTEE — bank ≥1 new level (rotated target)     │
   A3  (exp4895)  SELF-PLAY (every milestone) — train + checkpoint verifier   │
   A4  (exp4896)  DEADLINE — held-out first-win readiness, GENUINELY LIVE,    │
                  CLEAN (model_specs + random_seed; no tautology)             │
                                                                              │
 RESERVED SLOTS ───────────────────────────────────────────────────────────  │
   B1  (exp4897) INFRA-1  adversarial audit: A1 non-degenerate control +      │
                          held-out-disjoint delta; A1b RAN LIVE (>60 s)       │
   B2  (exp4898) INFRA-2  submission-package FINAL harden (6/30; never submits)│
   C   (exp4899) HARDWARE KV260 SSH-only continuity (always write artifact)   │
   D   (exp4900) SOTA     .452 frontier per A1's ACTUAL representation fork    │
                                                                              │
   E   (exp4901) CAPSTONE aggregate; headline = the representation-fork       ◄┘
                          verdict (+ escalate if representation-invariant)
```

**Slot accounting (ARC Submission Sprint Forcing Function):** majority ARC (A1, A1b, A2,
A3, A4 — 5 of 11) + Level-Up Attempt Guarantee (A2; rotates off `.450 g50t / `.449 s5i5 /
`.448 r11l) + self-play every milestone (A3 = continuous self-learning / FR-11) + 2 reserved
infra (B1, B2) + 1 hardware continuity (C, KV260, SSH-only, always-write-blocked-artifact) +
1 SOTA-ingestion (D, `.452 frontier per the representation fork) + capstone (E) + transition
(phase0). Live stack FROZEN to Qwen3.5-9B-MTP on the iGPU; **offline induction (A1/A1b/A4)
runs on the conductor's dedicated GPU-0 CUDA generator** per the 2026-06-27 directive.

---

## 5. Phase descriptions

### Phase 0 — transition (exp4891)
Archive `.450 → activate `.451. Record the close-state: A1 = **INDUCER_CEILING_HARD**
(trustworthy; code engine learns change-LOCATION `cell_recall 0.727` but not change-VALUE,
TTA delta `-0.0087`); A1b = **fabrication-flagged non-test** (13.7 s; attribution
unconfirmed); A2 banked g50t L2 (`reproducible_total_levels = 68`); A3 self-play pass; A4
flat-null ~0.0625 (flagged); B2 package ready (15.146 GB); energy CONCLUDED; the wall is the
**executable-code change-VALUE representation**. Resolve any poison pre-test.
`aggregation_from_upstream_artifacts`.

### Phase A1 — HEADLINE: agent-authored decision-need world-model targets (exp4892)
**arXiv:2606.25421 (Agent-Authored World Modeling).** Instead of supervising the engine on
generic exact transitions (which learns change-LOCATION but not change-VALUE), author
**decision-need targets** — the decision-relevant dynamics facts (per-object action-effect,
object persistence, hidden-register deltas) — as a **non-code** representation, fit/prompt
it from the agent's OWN cold-start transitions, and score whether predicting change-VALUE
via this target table beats the executable-code engine's `changed_cell_value_accuracy` on
**A1's exact held-out games + split + graded metric** (reuse exp4882). Primary metric: per-
game + median `changed_cell_value_accuracy` DELTA (decision-need − code-engine baseline)
with bootstrap CI95; **emit BARE `decision_need_value_accuracy_delta_median`** (A1b's
`gated_on` reads it). Fork: **REPRESENTATION_UNLOCKS_VALUE** (delta CI95 excludes 0 AND ≥1
migration NEVER_ENUMERATED→COVERED — `.452 scales it toward first-win) / **PLANNER_GAP**
(value up, no migration — `.452 attacks search) / **VALUE_GAP_REPRESENTATION_INVARIANT**
(delta CI95 includes 0 — A1b tries a structurally-different representation). Live-path-
reachable (improves the live `e3` induction path; `arc_orphan_solver_lint` must pass),
planner-blind, `verifier_is_oracle: false`, GPU-0 generator, wall-clock-safe (checkpoint per
game, soft budget ~3500 s, measure value accuracy FIRST). `solve_provenance:
development_proxy` (an inducer-accuracy measurement, NOT a level solve). `prior_failures:` →
exp4882 (TTA-on-code-engine null; addressed by a *different representation*, not TTA).

### Phase A1b — SECOND REPRESENTATION: action-prefix latent transition adapter (exp4893)
**Gated on A1's `decision_need_value_accuracy_delta_median < 0.1`** (the value-gap-not-closed
regime; skips cleanly if A1 closed the gap or blocked). **arXiv:2606.26217 (Fast
LeWorldModel) + 2603.19312 (LeWorldModel).** Encode candidate **action prefixes** into latent
future-state deltas (a structurally-different non-code representation: multi-step latent
deltas, not symbolic decision-need targets) and score A1's held-out transitions through the
latent adapter; convert only accepted deltas into engine facts. **MUST genuinely run live**
(>60 s, checkpointed, real induction over ≥3 games — the `.450 A1b `DURATION_TOO_SHORT`
13.7 s non-test fix). Same A1-graded held-out gate. Fork: **REPRESENTATION_MATTERS** (latent
multi-step raises value accuracy → `.452 scales it) / **VALUE_GAP_REPRESENTATION_INVARIANT_
HARD** (neither A1 decision-need NOR A1b action-prefix-latent raises value accuracy → the
change-VALUE gap survives executable-code + two alternative representations → a deep ceiling;
capstone escalates to operator). Live-path-reachable, `verifier_is_oracle: false`, GPU-0,
wall-clock-safe. `solve_provenance: development_proxy`. `prior_failures:` → exp4883 (the
fabrication-flagged A/B non-test — addressed by *actually running live* and testing a *new
representation*, not a reference-vs-local inducer A/B) + exp4882 (TTA null).

### Phase A2 — Level-Up Attempt Guarantee (exp4894)
Bank ≥1 new reproducible level on a **rotated** target (rotate off `.450 g50t, `.449 s5i5,
`.448 r11l). Pick a shallow game with a *grounded* next-level delta via `recommend_approach`
+ `dead_ends` — candidates **dc22 L2→L3 / sp80 L2→L3 / su15 L2→L3 / cn04 L2→L3** (avoid the
hidden-state-bound dead-ends ka59/wa30 and any recorded dead-end). All 25 public games are at
L1+, so this is a deepening attempt. Reproduction-gated (`arc_solver_kit.reproduce`);
`solve_provenance: live_agent_self_discovery`.

### Phase A3 — self-play, every milestone (exp4895)
Standing `arc_loop_solve` on a banked game (warm-started from the saved checkpoint), **rotated
off `.450 ls20 and `.449 re86** (pick another banked game that has an existing
`models/arc_verifier_<game>.json` checkpoint): verifier-routed solve → reproduction gate →
**train + checkpoint** the learned verifier. The continuous-self-learning / FR-11 experiment.
`solve_provenance: live_agent_self_discovery`.

### Phase A4 — deadline go/no-go, GENUINELY LIVE + CLEAN (exp4896)
Re-run the held-out first-win readiness on the exp4605 variant harness **genuinely live**
(`live_agent_ran: true` REQUIRED), on the conductor's GPU-0 generator. **DELIVERABLE-FIRST**
(run the exp4729 driver + write the artifact BEFORE any test/spec authoring; reuse the
existing 4729 tests; `max_turns ≤ 50` — the `.435/`.449 over-scope wall-clock fix). Report the
rate + CI vs the 0.04 baseline and prior-best. **Carry `model_specs` + `random_seed`** and a
`null_delta_methodology_note` for a flat null — the `.450 A4 was flagged TAUTOLOGY +
METHODOLOGY_MISSING; `.451 must produce a headline-eligible number. `solve_provenance:
development_proxy`.

### Phase B1 — INFRA-1: adversarial audit of A1 + A1b (exp4897)
The Phase-Prototype+Validation adversarial check. Verify A1 (exp4892): the GRADED positive
control is **non-degenerate** (`cell_recall > 0`), the value-accuracy delta is on a held-out
split **disjoint** from the representation's fit set (not a tautology), planner-blind, numbers
match the fork verdict, live-path-reachable, ran live on GPU-0, oracle-distinct. Verify A1b
(exp4893, if it ran): it **genuinely ran live** (`duration_s > 60`, NOT
`DURATION_TOO_SHORT` — the explicit `.450 A1b failure), same held-out split as A1, oracle-
distinct, live-path-reachable; or record it gate-skipped. `aggregation_from_upstream_artifacts`.

### Phase B2 — INFRA-2: submission package FINAL harden (exp4898)
Re-verify the Kaggle ARC-AGI-3 package builds and the **frozen Qwen3.5-9B-MTP stack on the
iGPU** fits ~16 GB; diff against the `.450 ready package (regression check); produce the FINAL
operator submission checklist. **Never submits** (Operator-Only External Publication; no
credentials). The live stack IS iGPU-pinned here (only offline induction uses GPU 0/1).

### Phase C — hardware KV260 (exp4899)
SSH-reachability continuity (SSH ONLY — host SD-card device nodes permanently retired).
**Always write the deliverable** — `blocked_kv260_ssh_unreachable` on failure (never exit with
no file → 3-fail-skip). `aggregation_from_upstream_artifacts` (an SSH state read).

### Phase D — SOTA ingestion, `.452 frontier (exp4900)
Given A1's **actual** representation-fork verdict (+ A1b's second-representation result),
ingest the matching SOTA for `.452. **REPRESENTATION_UNLOCKS_VALUE / REPRESENTATION_MATTERS**
→ first-win CONVERSION (turn the accurate-enough representation into banked levels: neural-
guided `plan_in_model` over the value-accurate engine). **VALUE_GAP_REPRESENTATION_INVARIANT
(_HARD)** → either a third representation class (decision-oriented / agent-authored targets,
arXiv:2606.25421; reverse-counterfactual targeter, arXiv:2505.08073; verification-calibrated
abstraction, arXiv:2602.23997) OR an operator-escalation note that the change-VALUE gap is a
deep ceiling under the deadline. Reliable channel only (`sweep_clusters.py` /
`sweep_semscholar.py` + low-concurrency WebSearch/WebFetch; **NO `/deep-research`**). Real
HTTP-200 arXiv IDs only. Do NOT re-ingest the nulled coverage / exploration / energy /
selection / TTA-on-code-engine classes.

### Phase E — capstone `.451 (exp4901)
Aggregate the scorecard. **Headline = the representation-fork verdict** (trusted only if B1
confirmed A1's non-degenerate control + held-out-disjoint delta + planner-blind, and that A1b
genuinely ran live). If **both** A1 (decision-need) and A1b (action-prefix latent) failed to
move the change-VALUE accuracy, **escalate to the operator**: the change-VALUE gap is
representation-invariant across executable-code + two alternative representations, and the
competition deliverable is the current 0.08 agent. Also report the scored deadline levers (A2
bank, A3 self-play, A4 fresh-live readiness, B2 package). Skip any `flagged_adversarial`
upstream. `aggregation_from_upstream_artifacts`.

---

## 6. Dependency graph

```
phase0 ─▶ A1 ─▶ A1b (gated: A1.decision_need_value_accuracy_delta_median < 0.1)
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
| **Conductor GPU-0 (RTX 3090) CUDA generator** | A1, A1b, A4 | Qwen3.5-9B-MTP via the local CUDA llama-server pinned to GPU 0 (`CARNOT_ARC_GENERATOR_CUDA_GPU=0`, the 2026-06-27 drop-in; ~24 GB free). Offline induction is NOT iGPU-pinned. |
| iGPU (AMD Radeon 890M) | B2 (live stack only) | Reserved for the FROZEN LIVE submission stack (Kaggle parity). NOT required for offline induction. |
| KV260 (SSH) | C | `ssh kria` reachability; graduated terminal, continuity-only. |
| Offline arcade (`arc_solver_kit.offline_arcade()`) | A1, A1b, A2, A3, A4 | Deterministic offline sim, all 25 games, zero quota. |

## 8. The fork → `.452 redirect (what this milestone decides)

- **REPRESENTATION_UNLOCKS_VALUE / REPRESENTATION_MATTERS** (a non-code representation raises
  change-VALUE accuracy + migrates a game): `.452 scales the winning representation toward
  first-win conversion (neural-guided `plan_in_model` over the value-accurate engine). This is
  the path off the 0.04 wall.
- **PLANNER_GAP** (value accuracy up, no migration): `.452 builds a guided planner / neural-
  guided search over the now-value-accurate engine.
- **VALUE_GAP_REPRESENTATION_INVARIANT (_HARD)** (both A1 + A1b fail): the change-VALUE gap
  survives executable-code + decision-need-targets + action-prefix-latents → a deep
  representation-invariant ceiling. The capstone **escalates to the operator**; the
  competition deliverable is the current 0.08 agent; `.452 either tries a third representation
  class (D's mapping) or the operator redirects.

---

*Discipline checklist:* every task `agent_type: codex` / `gpt-5.5` (ARC sprint; planner +
retro stay Opus). Every `honest_verdict` uses a terminal prefix
(`complete_`/`success_`/`blocked_`). Every compute-bound task has a PRECONDITIONS step 0 and
declares `inference_substrate`. Every REQUIRED ARTIFACT FIELD is principle-annotated. ARC
solve tasks declare `solve_provenance` (prefer `live_agent_self_discovery`; A1/A1b/A4 are
honest `development_proxy` measurements). A1/A1b carry `prior_failures:` (rerun discipline) +
the standing ARC-sprint `operator_override:`; A2/A3/A4/B1/B2/C/D/E carry the reserved-slot
`operator_override:`. No energy / coverage / exploration / perception-from-grid / TTA-on-code-
engine re-proposals (all closed). Offline induction on GPU-0 (not iGPU-pinned); the live
submission stack stays iGPU-frozen. Do NOT modify `scripts/research_conductor.py`. Do NOT push.
