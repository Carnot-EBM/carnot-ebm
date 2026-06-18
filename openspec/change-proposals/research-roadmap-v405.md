# Research Roadmap v405 — 2026.06.405

**Headline:** DEEPEN the one surviving oracle-distinct verifier vehicle — the
verifier-as-DETECTOR (exp4375, AUROC 0.918, beats chance where selection
headroom is ~0) — from "beats chance" into an ACTIONABLE capability:
**BiPRM-style bidirectional step-error LOCALIZATION + selective-prediction
abstention** (arXiv:2508.01682, the SOTA-ingestion `flagged_for_v405`). Probe
whether the detector GENERALIZES beyond FoVer (the domain-bound concern), and
whether it SELF-LEARNS / COMPOUNDS. Drive the ARC north star DEEPER (operator
MANDATORY 2026-06-17), now augmented by Mind-Studio lookahead-fidelity
(arXiv:2606.16070). `paper_ready=True` (FoVer 0.9131, G1–G4) carried unchanged.

**Provenance:** outer-loop planner (Claude Opus 4.8), 2026-06-18, after all 11
`.404 tasks completed. The `.404 SOTA-ingestion (exp4376) already mapped this
fork: `flagged_for_v405 = biprm_processbench_detector_localization_v405`.

---

## 1. What .404 proved (the honest scorecard)

Read via `scripts/summarize_artifact.py` + the `.404 capstone (exp4379,
`verifier_thesis_state: linear_settled_in_generation_retired_detector_positive`).
`.404 tested THREE vehicles for "the oracle-distinct verifier earns its place";
two closed, one is alive:

- **EFFICIENCY moat → SETTLED (`linear_is_settled`).** The stronger
  LLM-generated-heuristic function class (arXiv:2503.18809, exp4370) did NOT
  beat the deployed linear action-cost baseline: `llm_heuristic_beats_linear=false`,
  a clean powered null (linear, llm_generated, bfs all tied at 646 held-out
  actions). The efficiency moat is REAL and DEPLOYED (exp4364, 25→16 held-out
  env-actions, `verifier_is_oracle=false`) but the function class is now
  settled — no remaining headroom there. A2 (exp4371) was correctly gate-skipped
  (no win to validate).
- **IN-GENERATION (DiffusionGemma) moat → RETIRED (4th block).** exp4374:
  `retired_in_generation_conversion_unmeasurable` — the `.401 scorer is
  irreparably leaky on the generation corpus (`scorer_requalified_leak_clean=false`)
  AND the scorer-independent CoDiLA control could not differentiate the arms
  (`codila_control_differentiates=false`, `benchmark_n=0`). Block lineage
  `.399 → `.402 → `.403 → `.404 hit the falsifiable retirement gate. The
  in-generation-conversion-via-this-scorer direction is OUT of the autonomous
  loop (operator-owned if ever revived).
- **DETECTION → POSITIVE (the one live oracle-distinct vehicle).** exp4375:
  `detector_auroc=0.918`, CI95 lower `0.909`, `detector_beats_chance=true`,
  n=8,829, `selection_headroom.headroom=0.0`, `verifier_is_oracle=false`. The
  verifier ensemble DETECTS step-errors with AUROC 0.918 **even on a corpus
  where it cannot SELECT a better answer** (oracle@K == vote@1). This is the
  surviving, unexploited, oracle-distinct positive — a genuine Carnot capability
  ("I don't know" / precision filtering) that the selection-headroom nulls did
  NOT refute.
- **ARC north star → 34 reproducible levels / 17 games** (was 33; authoritative
  `ops/arc_solve_registry.yaml`). exp4372 advanced **lp85 +1** (→ L5).
  exp4373's blocked-mechanic tails (ar25/ka59/ft09 L2) produced NO new levels —
  `complete_e3_ar25_ka59_ft09_partial`, the named hidden-rule gaps persist.
- **`paper_ready=True`** (G1–G4, FoVer 0.9131, `unmet_gates: []`). Operator
  submission only.

**The SOTA-ingestion's own forward pointer (exp4376):**
`flagged_for_v405 = biprm_processbench_detector_localization_v405` — "make the
positive Exp 4375 detector signal actionable with BiPRM-style bidirectional
step-error localization and abstention" (arXiv:2508.01682). It also flagged the
LLM-generated-heuristic arm as **settled/null** (use only as a control, do NOT
re-headline) and the DiffusionGemma path as **retired** from the loop.

---

## 2. The three biggest gaps (current state vs PRD / north-star vision)

1. **The oracle-distinct verifier value is PROVEN but NOT ACTIONABLE.** north-star
   §0 names the oracle-distinct frontier as THE open claim; §5 names the verifier
   as Carnot's entire value-add. With efficiency SETTLED and in-generation
   RETIRED, **detection is the one live vehicle** — but "AUROC beats chance" is
   not a usable capability. The defensible second-headline-class result is a
   detector that (a) LOCALIZES the earliest error (ProcessBench-style step
   localization, bidirectional fusion) and (b) enables a USEFUL selective-
   prediction operating point (risk-coverage / "abstain on the lowest-scored k%").
   **Gap: turn the positive detector into an actionable localization + abstention
   capability.** → Phase A (headline).

2. **The detector is FoVer-domain-bound.** north-star §0 step 2 (verifier domain
   expansion) and memory `verifier-domain-bound-math-only`: the verifier is math-
   strong, facts/code weak. exp4375 measured detection on FoVer only. The SOTA-
   ingestion's own failure-mode note: "Math PRM gains can fail on FoVer or ARC
   traces." **Gap: does detection generalize beyond FoVer to a different domain's
   cached traces?** → Phase D (complementary, cheap, cached).

3. **ARC-AGI-3 north star — deep tails blocked on NAMED hidden-rule gaps.** 34/17
   levels; the un-reproduced next levels are blocked on specific mechanics
   (tn36 L8 program-editor control, ar25 L2 action7 undo-stack, ka59 L2 hidden
   step-counter HUD). Operator MANDATORY 2026-06-17 (incremental +1..+n per game).
   **Gap: close the named gaps, augmented by lookahead-fidelity.** → Phase B.

The **continuous self-learning** mandate (research-program.md) is folded into the
headline this milestone: does the DETECTOR self-improve / COMPOUND as labeled
traces accumulate (Phase C) — the same compounding-curve discipline that proved
the action-cost heuristic in exp4364, now on the live detector vehicle.

---

## 3. Milestone design — 11 experiments across 5 phases

| # | id | phase | track | what it answers | gate field |
|---|----|-------|-------|------------------|------------|
| 0 | exp4380 | 0 transition | infra | archive .404 → activate .405; record TRUE close-state | — |
| 1 | exp4381 | A headline | detector-actionable | does bidirectional fusion LOCALIZE the earliest error + give a useful abstention operating point? | `detector_localization_actionable` |
| 2 | exp4382 | A headline | detector-actionable | is the localization/abstention win GENUINE (not position/length bias, R2L future-context leak, single-split overfit)? | `localization_win_is_genuine` |
| 3 | exp4383 | B ARC | arc-north-star | E3 DEEPER high-headroom (lp85/tu93/tn36/tr87) + Mind-Studio lookahead | `new_levels_reproduced` |
| 4 | exp4384 | B ARC | arc-north-star | E3 blocked-mechanic tails (ar25/ka59/ft09 L2) + active-data + lookahead | `new_levels_reproduced` |
| 5 | exp4385 | C self-learning | detector-self-learning | does the DETECTOR COMPOUND as labeled traces accumulate? | `detector_compounds` |
| 6 | exp4386 | D complementary | oracle-distinct-detection | does detection GENERALIZE beyond FoVer to a different domain? | `detector_generalizes_cross_domain` |
| 7 | exp4387 | E infra | infra | SOTA-ingestion → .406 (reliable channel; verified arXiv IDs) | `flagged_for_v406` |
| 8 | exp4388 | E infra | infra | registry/gaps hygiene + GAP-4 guard + capstone-stamp durability | `gap4_regression_guard_passed` |
| 9 | exp4389 | E hardware | hardware | KV260 SSH continuity (opportunistic) | `kv260_reachable` |
| 10 | exp4390 | E capstone | capstone | the .405 scorecard + headline decision + G1–G4 | `detector_actionable_state` |

### Dependency graph

```
exp4380 (archive/activate)
   │
   ├── PHASE A (HEADLINE — detector made actionable; verifier_is_oracle=false)
   │     exp4381  BiPRM bidirectional localization + abstention   ─┐
   │     exp4382  skeptic-proof (GATED on exp4381 win)            ←┘
   │
   ├── PHASE B (ARC north star; verifier_is_oracle=true)
   │     exp4383  E3 deeper high-headroom + lookahead
   │     exp4384  E3 blocked-mechanic tails (prior_failures: exp4373)
   │
   ├── PHASE C (self-learning — detector compounds; verifier_is_oracle=false)
   │     exp4385  detector compounding curve
   │
   ├── PHASE D (complementary oracle-distinct; verifier_is_oracle=false)
   │     exp4386  cross-domain detection generalization
   │
   └── PHASE E (infra + hygiene + capstone)
         exp4387  SOTA-ingestion → .406
         exp4388  registry/gaps hygiene + GAP-4 guard
         exp4389  KV260 hardware continuity
         exp4390  capstone .405  (reads exp4381/4382/4383/4384/4385/4386)
```

Only one structured `gated_on` edge: exp4382 ← exp4381
(`detector_localization_actionable == true`). The capstone reads all upstreams
but is not hard-gated (robust aggregate-available-report-gaps helper).

---

## 4. Phase detail

### PHASE A — HEADLINE: make the oracle-distinct DETECTOR actionable (verifier_is_oracle=false)

The `.404 detector win (AUROC 0.918) proved the verifier *separates* correct from
incorrect where it cannot *select*. `.405 makes that signal USABLE:

- **exp4381 (A1) — BiPRM bidirectional localization + abstention.** Per
  arXiv:2508.01682 (Bidirectional Process Reward Model): run an L2R *and* an R2L
  detector pass over the cached, step-labeled FoVer corpus (the same corpus
  exp4375 scored), FUSE the two scores, and report (1) **earliest-error
  LOCALIZATION** — ProcessBench-style first-error-step accuracy/F1 — for the
  bidirectional fusion vs a unidirectional L2R baseline; (2) the
  **selective-prediction operating point** — the accuracy-vs-coverage /
  risk-coverage curve, precision@high-recall, and the selective risk at a useful
  coverage. **Online-actor honesty (the SOTA failure-mode):** the R2L pass uses
  future context unavailable to an in-loop actor → report bidirectional
  localization as an OFFLINE post-hoc detector, and keep a causal/L2R-only
  variant as the online-actionable number; never conflate the two.
  `detector_localization_actionable` BARE bool := (bidirectional fusion improves
  first-error localization over unidirectional L2R, CI95-excl-0, AND the
  risk-coverage curve yields a useful selective operating point). Cheap, cached,
  CPU — NOT hostage to any live LLM or the retired DiffusionGemma infra.
- **exp4382 (A2, GATED on A1) — skeptic-proof.** The twice-burned operator will
  ask: is the localization/abstention win genuine, or an artifact of
  (a) **position/length bias** (the detector just predicts "errors come late/early"
  — test against a position-only baseline), (b) **R2L future-context leakage**
  making the bidirectional gain online-invalid (report the causal-only delta), or
  (c) **single-split/seed overfit** (a held-out split + bootstrap CI95)?
  `localization_win_is_genuine` BARE bool. A fail QUARANTINES the A1 win and logs
  the residual as a missing-verifier gap.

### PHASE B — ARC NORTH STAR (accuracy; operator 2026-06-17 E3 MANDATORY; +1..+n per game; verifier_is_oracle=true)

- **exp4383 (B1) — E3 DEEPER high-headroom.** Extend the existing world models /
  solvers on the cracked high-headroom games to their next un-reproduced level:
  **lp85 L6, tu93 L5, tn36 L8** (the program-editor object-control — the named
  registry gap), **tr87 L7**. Augment the explore-verify-plan harness with
  **Mind-Studio lookahead-fidelity** (arXiv:2606.16070): entropy-selected traces +
  a lightweight per-game skill file + K-step lookahead fidelity checks BEFORE
  planning. Read `ops/arc_solve_registry.yaml` for each game's authoritative
  `levels_reproduced` and target +1 beyond it. HARD per-target gate:
  `arc_solver_kit.reproduce` on the OFFLINE env. Per-target checkpoint + wall-time
  cap (breadth-of-progress beats all-or-nothing).
- **exp4384 (B2) — E3 blocked-mechanic tails (prior_failures: exp4373).** Close the
  NAMED hidden-rule gaps to reach **ar25 L2** (action7 undo-stack), **ka59 L2**
  (hidden step-counter HUD), **ft09 L2** — extending the `.404 active-data lever
  (M2-v4) with Mind-Studio K-step lookahead-fidelity (the STATED forward
  difference vs the exp4373 partial). Per-game checkpoint + wall-time cap; an
  honest partial (refined model + sharper residual gap CLASS) is progress.

### PHASE C — CONTINUOUS SELF-LEARNING (mandated; the DETECTOR compounds; verifier_is_oracle=false)

- **exp4385 — does the detector COMPOUND?** The same compounding-curve discipline
  that proved the action-cost heuristic (exp4364), now on the live detector
  vehicle: the detector's operating threshold + the BiPRM fusion weights are
  LEARNED and updated online as labeled (output, is_correct, step-label) traces
  accumulate; measure detection localization-F1 / selective-risk vs accumulated-
  corpus size (a monotone-improving curve), reproduction-gated, with a positive
  control (a from-scratch detector on the full corpus as the ceiling).
  `detector_compounds` BARE bool := (held-out localization-F1 / AUROC rises with
  corpus size beyond the no-learning baseline, CI95-excl-0). A clean null (the
  detector is already saturated on FoVer) is decision-grade.

### PHASE D — ORACLE-DISTINCT cross-domain DETECTION (complementary; cheap; verifier_is_oracle=false)

- **exp4386 — does detection GENERALIZE beyond FoVer?** Score the detector against
  a NON-FoVer cached corpus (the GAP-4 ARC pool `results/arc3_trm_verifier_rerank.json`,
  and/or the cached code HumanEval/MBPP / GSM8K candidate pools from the headroom
  census, per the detector spec). Report detection AUROC + the selection-headroom
  PER domain → the headline is the **divergence** (detect where you cannot select).
  `detector_generalizes_cross_domain` BARE bool := (detection AUROC CI95 lower > 0.5
  on ≥1 non-FoVer domain). Log any domain where detection ≈ chance as a missing-
  verifier gap (`ops/verifier_gaps.md`) — that IS the product backlog. **This is
  cross-domain DETECTION (AUROC where selection headroom is ~0), NOT the retired
  cross-domain SELECTION axis (exp4314) — a different measurement mandated by the
  2026-06-14 P0 directive** (operator_override carries the distinction).

### PHASE E — INFRA + HYGIENE + CAPSTONE

- **exp4387 — SOTA-ingestion → .406.** Reliable channel only (sweep_clusters.py /
  sweep_semscholar.py + low-concurrency WebSearch/WebFetch); `/deep-research`
  BANNED in-loop. Every method carries a VERIFIED arXiv ID. Flag A2D2 (2606.13565)
  + SEPO (2502.01384) verifier-as-reward GENERATOR-training as OUT-OF-BAND.
  Condition the `.406 map on the `.405 outcomes.
- **exp4388 — registry/gaps hygiene + GAP-4 guard.** Reconcile
  `verifier_registry.yaml` + `verifier_gaps.md` + `arc_solve_registry.yaml` with
  the `.405 outcomes; run the GAP-4 regression guard; confirm the capstone
  `verifier_is_oracle` stamp fix is still durable. Audit-only.
- **exp4389 — KV260 hardware continuity.** SSH-reachability ONLY (never host SD
  card). Opportunistic per north-star §3 (KV260 = THE sovereignty story).
- **exp4390 — capstone .405.** The scorecard + headline decision: did the detector
  become ACTIONABLE (localization genuine + abstention useful)? does it COMPOUND?
  does it GENERALIZE cross-domain? the ARC reproducible-total; G1–G4 via
  `publication_gate.py`. SKIP flagged_adversarial; HONOR verifier_is_oracle.

---

## 5. HARD RULES (carried into every .405 task)

1. **Conductor STOOD-DOWN on TRM training.** No task launches TRM training, runs
   `pkill/kill` against `train.py`, or writes `results/trm_runs/`. Qwen FORBIDDEN
   as the TRAINED base (Spurious-Rewards confound); Qwen/Gemma GGUF as an
   off-policy judge/generator is fine.
2. **A2D2 (2606.13565) + SEPO (2502.01384)** verifier-as-reward GENERATOR-training
   are OUT-OF-BAND / operator-owned — flagged in SOTA-ingestion, NOT auto-run.
3. **Circularity / Oracle-Distinctness Discipline.** Every verifier-value task
   declares `verifier_is_oracle` honestly. The detector / efficiency / detection-
   generalization claims are oracle-DISTINCT (`verifier_is_oracle=false` + a
   matched control + CI95-excl-0). The ARC E3 SOLVEs are execution-grounded
   (`verifier_is_oracle=true`) — ARC progress, NOT a moat headline.
4. **DiffusionGemma in-generation conversion is RETIRED** from the autonomous loop
   (4th block, exp4374). Do NOT re-propose it. CoDiLA stays a diagnostic control
   only if the operator ever revives the path.
5. **The LLM-generated-heuristic efficiency arm is SETTLED** (exp4370 null). Do NOT
   re-headline another generated-heuristic sweep; use it only as a control.
6. **No autonomous edits** to `docs/index.html` / `README` / paper prose. Online
   ARC play stays operator-gated (NO leaderboard submission; only offline-
   reproduced levels count). DiffusionGemma (if ever) via the llama.cpp PR binary.
7. **Cross-game value TRANSFER (exp4342) + cross-domain SELECTION (exp4314) are
   RETIRED** — do NOT re-propose either. Phase D is cross-domain DETECTION (a
   distinct, mandated measurement).
8. **`paper_ready=True` (FoVer 0.9131) is the frozen headline** — `.405 adds the
   detector-actionable + ARC-depth + cross-domain-detection LENSES, never a
   substitute headline.

---

## 6. Hardware requirements

- **Phase A / C / D (detector):** CPU only — verifier scoring against cached,
  step-labeled candidate pools (`verifier_ensemble_against_cached_candidates`).
  Zero quota, infra-independent. This is a deliberate strength: the headline is
  NOT hostage to GPU/GGUF/PR-binary infra (which blocked the retired in-generation
  path 4×).
- **Phase B (ARC E3):** CPU offline arcade + codex world-model synthesis; no live
  GGUF load required (the codex agent is the proposer); `gemma-4-12B-it-GGUF`
  declared as the reproducible-alternative generator.
- **Phase E:** aggregation (CPU) + network (SOTA-ingestion) + KV260 SSH (hardware).
- **KV260:** SSH-reachable (`ssh kria`), opportunistic continuity only.

---

## 7. Success criteria (the .405 capstone reads these)

- **HEADLINE:** `detector_actionable_state ∈ {actionable_localization_and_abstention,
  detects_but_not_actionable, open}` — `actionable` iff exp4381
  `detector_localization_actionable==true` AND exp4382 `localization_win_is_genuine==true`.
  The oracle-distinct verifier graduates from "beats chance" to "localizes + abstains
  usefully" — a defensible second-headline-class capability.
- **SELF-LEARNING:** `detector_compounds` (exp4385) — does the live detector vehicle
  self-improve with accumulated data?
- **GENERALIZATION:** `detector_generalizes_cross_domain` (exp4386) — does detection
  transfer beyond FoVer, or is it domain-bound (a logged gap)?
- **ARC north star:** `reproducible_total_levels` ≥ 34, monotone; NEW levels/games
  from exp4383/4384.
- **Publication gate:** G1–G4 via `publication_gate.py` (`paper_ready` carried True,
  `unmet_gates` reported).
- **Discipline:** every moat read carries `verifier_is_oracle` correctly (no
  CIRCULAR_MOAT_OVERCLAIM); no flagged_adversarial artifact aggregated.
