# Research Roadmap — Milestone 2026.06.408

**PIVOT the ARC north star to verifier-grounded CONFIG-RULE INDUCTION (the operator's #1 lever — 9 toggle/config games) after the oracle-distinct first-error LOCALIZER RETIRED as position-bound. Plus: Agent2World adaptive E3 mechanic repair; a definitive hidden-state falsification of the localizer; a config-rule-vocabulary self-learning transfer; SteerConf code-domain calibration repair. paper_ready=True (FoVer 0.9131, G1–G4) stays the FROZEN headline.**

- **Milestone:** 2026.06.408
- **Planner:** Claude Opus 4.8 (outer-loop), 2026-06-18
- **Predecessor:** 2026.06.407 (exp4412 capstone, verdict `v407_localizer_position_bound_retired_compounds_false_calibrated_false_arc_levels_34_publication_ready`)
- **Design doc:** this file

---

## 1. What .407 proved (the close-state this milestone builds on)

`.407 took the honest next step on the `.406 quarantine: it replaced the SYNTHETIC template-injected
first-error data (which `.406's skeptic-proof exposed as pure position bias) with **REAL
verifier-checked intervention first-error data** (arXiv:2601.14209). The result retired the line.

| Axis | .407 outcome | The decisive number |
|---|---|---|
| **Oracle-distinct first-error LOCALIZER** | **`position_bound_retired`** | exp4403: FoVer position-only baseline F1 = **1.0** (first-error onset is near-deterministic in position → ~zero content headroom); real-intervention localizer F1 = 1.0, `delta_vs_position_only = 0.0`, `template_family_holdout_drop = 0.0`. GAP-4 ARC `delta = +0.019`, CI95 **[−0.13, +0.17]** (includes 0). `retire_if_same_verdict: true` fired. |
| **Self-learning / COMPOUNDS** | **false** | exp4407: active-vs-random learning curve FLAT at F1 = 1.0 across corpus 51→512 (saturated; positive control did not pass — no headroom on a position-bound signal). |
| **Cross-domain detection CALIBRATION** | **false** | exp4408: only FoVer detects (AUROC 0.918). **code_humaneval at chance (AUROC 0.577, CI95 [0.46, 0.69], n=539)** even after proper pools + semantic-confidence aggregation. The verifier is **domain-bound** (math strong, code weak). |
| **ARC north star (reproducible levels)** | **34 / 17 games, 0 new** | exp4405 (deeper) + exp4406 (tails): the per-mechanic EXECUTABLE UNIT TESTS **passed** (ar25 register test 1.0) but reproduction did NOT follow — `lookahead_fidelity` stuck ~0.73. **Static per-mechanic unit tests document blockers but do not deepen ARC.** |
| **Publication gate** | **paper_ready = True** | FoVer 0.9131, G1∧G2∧G3∧G4, `unmet_gates: []`. FROZEN headline. |

**The two load-bearing lessons:**

1. **The oracle-distinct LEARNED-verifier vehicles are now largely exhausted.** The first-error
   localizer is RETIRED this milestone (text intervention data); GAP-3 trained-content-energy was
   RETIRED earlier (Stage 2v2). Re-running either is a doomed rerun. The ONE remaining oracle-distinct
   question worth one bounded shot is whether the signal exists OFF-TEXT (in model hidden states) — a
   falsification, not a redeployment.

2. **Static unit tests are not the ARC lever.** Three milestones (.405/.406/.407) added 0 reproducible
   levels by gating on aggregate fidelity / static per-mechanic tests. The freshest, validated,
   operator-hands-on lever is **the verifier as the GROUNDING ORACLE for an LLM-PROPOSED win-RULE** —
   the Config Layer B result (2026-06-18): a local offline gemma-4-12B that cannot read a raw 64×64
   scene CAN induce a GROUNDED relational win-rule (ka59 → Tier-2 `count_4 == 32`, fires-on-win, 0 FP)
   when handed object-centric digests, and the verifier grounds/rejects the proposed predicate. The
   operator's first-contact audit names the **9 toggle/CONFIG games as "the clear #1 next investment
   and the genuinely hard one."**

---

## 2. The three biggest gaps (current state vs PRD vision)

1. **The ARC-AGI-3 solve count is STUCK at 34/17 — the north star is not moving.** The biggest
   untapped lever is the **9 toggle/config games** (bp35, dc22, g50t, ka59, lf52, s5i5, tn36, sc25,
   tr87) whose win is a relational target-PATTERN that random exploration cannot stumble. Config Layer B
   just proved verifier-grounded win-rule induction works on this class. **→ Phase A.**

2. **The verifier's value is DOMAIN-BOUND (math/FoVer only) and its learned-energy moat vehicles are
   exhausted.** The most credible remaining demonstration of value is the HYBRID (verifier grounds a
   cheap local generator's proposal) — for win-rules (Phase A) and for rule-execution on ARC with a
   LOCAL generator (the sovereignty tier of the GAP-4 forward protocol). The one open oracle-distinct
   question (off-text first-error signal) gets one definitive falisfication shot. **→ Phase A + Phase B.**

3. **Continuous self-learning has no live compounding signal** (the localizer-compounding line is dead,
   saturated). The PRD's FR-11 needs a substrate with REAL headroom: a **config-rule VOCABULARY** learned
   from solved config games that transfers to induce a new game's win-rule with less signal. **→ Phase C.**

---

## 3. Architecture: the .408 verifier-grounded-proposal loop

```
        ┌──────────────────────── ARC-AGI-3 config/toggle game (offline sim) ────────────────────────┐
        │                                                                                            │
        │   object-centric DIGEST            LLM PROPOSER (local, sovereign)         VERIFIER (Carnot)│
        │   (editable bbox + counts,   ──▶   gemma-4-12B-Q4 on the iGPU       ──▶    grounds is_win() │
        │    reference components)            "emit a RELATIONAL is_win(grid)"        on banked win + │
        │        (NOT a raw 64×64 grid)                                               non-wins        │
        │                                          │  propose                            │ ground     │
        │                                          ▼                                     ▼            │
        │                              candidate win-RULE predicate  ──reject-if-wrong──▶ GROUNDED    │
        │                                                                                 predicate   │
        │                                                                                    │        │
        │                                          OfflineSolver(verifier = grounded predicate)       │
        │                                          best-first search ──▶ +1 LEVEL ──▶ reproduce() GATE │
        └────────────────────────────────────────────────────────────────────────────────────────────┘
            Phase A1 (config-rule induction → solve)        Phase A2 (Agent2World adaptive E3 repair)

   Phase B   localizer falsified OFF-TEXT (hidden-state audit) │ GAP-4 rule-exec with a LOCAL generator
   Phase C   config-rule VOCABULARY from solved games ─transfer─▶ new game's win-rule (self-learning)
   Phase D   SteerConf steered-confidence ─rescue?─▶ code_humaneval detection (domain calibration)
```

**The thesis this milestone tests:** Carnot's verifier earns its place as the cheap, local,
execution-grounded ORACLE that grounds (accepts/rejects) a local generator's proposed predicate —
turning a model that cannot solve a game alone, paired with a verifier that cannot generate, into a
system that SOLVES the game. This is the hybrid (north-star §5), made concrete on the ARC class that has
resisted every blind-search approach.

---

## 4. Phases & experiments (exp4413–exp4424; 12 tasks)

### PHASE 0 — Transition
- **exp4413** — archive `.407 → activate `.408; record the TRUE close-state (localizer RETIRED
  position-bound; compounds false; calibration false; ARC 34/17; paper_ready True). codex, mechanical.

### PHASE A — ARC NORTH STAR: verifier-grounded config-rule induction + adaptive E3 (operator MANDATORY)
- **exp4414 (A1, PRIMARY)** — **Config/toggle SOLVE via verifier-grounded win-RULE induction.** Reuse
  the validated scaffolded inducer (`scripts/experiments/arc3_config_layerb_scaffolded.py`): for the
  config games, ground a relational win-rule predicate (object-centric digest → local gemma-12B proposes
  `is_win` → verifier grounds it on banked win/non-wins), then wire the grounded predicate as the
  `OfflineSolver` verifier to drive a solve to the next level. Targets: **ka59 L2** (rule known, drive
  the solve) + first-contact win-rule grounding on **≥2 UNSOLVED config games** (bp35/dc22/g50t/lf52/s5i5).
  Per-game checkpoint, wall-time cap. HARD gate: a NEW level `offline_reproduced` OR a Tier-2 grounded
  win-rule for a new config game (honest partial). `verifier_is_oracle: true` (the SOLVE is
  execution-grounded — ARC progress, NOT an oracle-distinct moat headline; the contribution is that the
  verifier GROUNDS the LLM-proposed rule).
- **exp4415 (A2)** — **Agent2World adaptive E3 mechanic repair** (`flagged_for_v408` #1, arXiv:2512.22336)
  + AERA speed-depth / leakage controls (arXiv:2605.25931) on the deep-tail games where `.407 static
  unit-tests passed but reproduction stalled: **ar25 L2, tn36 L8, lp85 L6**. Forward difference vs `.407:
  a "Testing Team" generates ADAPTIVE behavior tests from failing rollout traces (not static named-register
  tests), feeds them to the world-model developer, reruns executable checks before solving. Incremental
  +1, per-game checkpoint. `verifier_is_oracle: true`.

### PHASE B — ORACLE-DISTINCT FRONTIER (operator P0 2026-06-14) + verifier-earns-its-place (sovereignty)
- **exp4416 (B1)** — **Hidden-state first-error localization AUDIT** (arXiv:2605.13772). The DEFINITIVE
  falsification of the retired text localizer: does a hidden-state transport margin carry ANY recoverable
  NON-position first-error signal? One bounded diagnostic vs the content-blind position-only baseline.
  Emit `hidden_state_localizer_has_nonposition_signal` BARE bool. `verifier_is_oracle: false`. A clean
  null conclusively CLOSES the first-error-localizer program (don't revisit); a weak signal is LOGGED as
  a gap (not a redeployment). `retire_if_same_verdict: true`.
- **exp4417 (B2)** — **GAP-4 rule-execution verifier with a LOCAL open-weight generator arm** (the
  known-issues GAP-4 forward protocol — the decentralization tier). Replace the contaminated codex/gpt-5.5
  inducer with a LOCAL Gemma-4/Qwen3.6 generator; apply the graded min-hamming gate (τ=0.005) + k-consistency
  agreement; measure pass@2 on the TRM rerank pool vs the matched no-verifier control. This is the
  north-star §5 win-condition (does the verifier earn its place with a SOVEREIGN generator?), framed
  honestly as `verifier_is_oracle: true` (execution gate) — NOT an oracle-distinct moat claim.

### PHASE C — CONTINUOUS SELF-LEARNING (mandated; the substrate that now has real headroom)
- **exp4418** — **Config-rule VOCABULARY transfer from previously-seen games** (operator's "derive
  config-rules from previously-seen games"). Build a config-rule vocabulary (relational primitives:
  count-equality, position-match, glyph-map, …) from the GROUNDED win-rules of solved config games, then
  measure whether seeding the inducer with the vocabulary raises the grounding rate on HELD-OUT config
  games vs cold-start. Emit `config_rule_vocabulary_transfers` BARE bool := (vocabulary-seeded grounding
  rate > cold-start, delta CI95-excl-0, held-out-game control). `verifier_is_oracle: false`. NOT gated.
  Replaces the dead localizer-compounding line with a substrate that has real headroom (9 games).

### PHASE D — CROSS-DOMAIN DETECTION CALIBRATION REPAIR (complementary; cached; CPU)
- **exp4419** — **SteerConf steered-confidence to rescue code_humaneval detection** (arXiv:2503.02863).
  `.407 (exp4408) left code detection at chance (0.577). Add conservative/optimistic steering probes +
  confidence-consistency features beside the verifier score; fit leave-domain-out calibration with a
  random-score control. Emit `detection_calibrated_multi_domain` BARE bool. `verifier_is_oracle: false`.
  `retire_if_same_verdict: true` (if steered confidence ALSO leaves code at chance, the multi-domain
  detector contract retires + logs a domain-bound gap).

### PHASE E — INFRA + SOTA + HARDWARE + CAPSTONE
- **exp4420** — SOTA-ingestion → `.409 (mandated). Reliable channel ONLY (sweep_clusters / sweep_semscholar
  + low-concurrency WebSearch/WebFetch; `/deep-research` BANNED in-loop). Emit `flagged_for_v409`.
- **exp4421** — registry/gaps hygiene + GAP-4 regression guard; reconcile `ops/verifier_registry.yaml`,
  `ops/verifier_gaps.md`, `ops/arc_solve_registry.yaml` with the `.408 outcomes. Emit
  `regression_guard_passed` BARE bool. Audit-only (no production verifier edits).
- **exp4422** — KV260 continuity (opportunistic per north-star §3). SSH-reachability precondition ONLY
  (NEVER a host SD-card precondition). Clean documented skip if unreachable.
- **exp4423** — CAPSTONE `.408: the milestone scorecard + the headline decision. Emit `verifier_thesis_state`
  + `config_rule_induction_state` + `arc_reproducible_total_levels` + the G1–G4 publication gate.

> Note: 11 experiment tasks + 1 capstone = 12 (exp4413–exp4423; exp4424 reserved if a Phase-A split is
> needed). Tasks renumber contiguously in the YAML.

---

## 5. Dependency graph

```
exp4413 (transition)
   │
   ├── exp4414 (A1 config-rule SOLVE) ─────────────┐
   ├── exp4415 (A2 Agent2World adaptive E3) ────────┤
   ├── exp4416 (B1 hidden-state localizer audit) ───┤
   ├── exp4417 (B2 GAP-4 local generator) ──────────┤
   ├── exp4418 (C  config-rule vocabulary transfer) ┤   (exp4418 prefers exp4414's grounded rules;
   ├── exp4419 (D  SteerConf calibration repair) ───┤    runs on solved-game rules if A1 is partial)
   ├── exp4420 (E SOTA-ingestion → .409) ───────────┤
   ├── exp4421 (E registry/gaps hygiene + GAP-4) ───┤
   └── exp4422 (E KV260 continuity) ────────────────┤
                                                     ▼
                                          exp4423 (CAPSTONE .408)
```

No hard `gated_on` chains (every axis is independently decision-grade; the capstone aggregates available
artifacts and reports per-axis gaps — no hard-block-all-False). exp4418 has a SOFT preference for
exp4414's grounded rules but falls back to the registry's existing solved-config-game rules.

---

## 6. Hardware requirements

- **Local iGPU (AMD Radeon 890M, gfx1150)** for the offline gemma-4-12B-Q4 win-rule proposer
  (Config Layer B server, ~4.2 tok/s, offline-legal, zero quota — NEVER the 3090s). Phase A1 + C.
- **2× RTX 3090 (CUDA)** available for any hidden-state extraction (Phase B1) / local-generator
  candidate generation (Phase B2) if a GGUF needs the discrete GPU.
- **KV260** opportunistic SSH-reachability check only (Phase E). No host SD-card precondition.
- Phases C/D/E are CPU + cached (zero new live inference where possible).

---

## 7. Disciplines honored (HARD rules for every task)

- **Conductor STOOD-DOWN on TRM/generator training** — NO task launches `train.py`, runs pkill/kill
  against it, or writes `results/trm_runs/`. A2D2 (2606.13565) + SEPO (2502.01384) are OUT-OF-BAND /
  operator-owned. Qwen FORBIDDEN as a TRAINED base; Qwen/Gemma GGUF as off-policy judge/generator is fine.
- **Circularity / Oracle-Distinctness Discipline** — every verifier-value task declares
  `verifier_is_oracle` honestly. Config-rule SOLVEs + Agent2World E3 + GAP-4 are execution-grounded
  (`verifier_is_oracle: true`, ARC progress NOT a moat headline); the localizer audit + calibration +
  vocabulary-transfer are oracle-distinct (`verifier_is_oracle: false` + matched control + CI95-excl-0).
- **ARC-AGI-3 Incremental-Progress Scoping** — +1..+n per game, NEVER "full solve / all levels."
- **ARC Solve Reproducibility** — a level counts only if `arc_solver_kit.reproduce` re-derives it
  offline; `total_levels` cites the registry's `reproducible_total_levels`.
- **Do NOT re-propose** the RETIRED lines: the SYNTHETIC or REAL-intervention first-error TEXT localizer
  (`position_bound_retired`, exp4392/4403), GAP-3 trained-content-energy selector, cross-game ARC value
  transfer (exp4318/4331/4342), in-generation DiffusionGemma (exp4374), LLM-heuristic efficiency
  (exp4370), cross-domain SELECTION (exp4314). The hidden-state audit (B1) is a one-shot FALSIFICATION,
  explicitly NOT a localizer redeployment.
- **Adversarial-verify + sample-size rigor** — n≥1000 for distributional claims (Phase B1/D); the
  IMPLAUSIBLE_PERFECT / FALSE_NEGATIVE_RISK detectors apply (an F1=1.0 that does not beat the
  position-only baseline is the .407 tell — report honestly).
- **Public-doc discipline** — NO autonomous edits to `docs/index.html` / README / paper prose.
  Online ARC stays operator-gated (NO leaderboard submission; only offline-reproduced levels count).
- **paper_ready=True (FoVer 0.9131) is the FROZEN headline** — `.408 adds lenses, never a substitute.

---

## 8. Success criteria for .408

| Outcome | Win | Honest null (still decision-grade) |
|---|---|---|
| **ARC config-rule induction (A1)** | ≥1 NEW reproducible level on a config game via a verifier-grounded win-rule | Tier-2 grounded win-rules for ≥1 new config game (the rule grounds; the solve is the residual) + the search-blocker logged |
| **Agent2World E3 (A2)** | ≥1 NEW reproducible level on ar25/tn36/lp85 via adaptive repair | adaptive tests that PASS + the residual world-model gap logged (sharper than .407's static tests) |
| **Hidden-state localizer audit (B1)** | a measured NON-position first-error signal off-text (logged as a gap, not redeployed) | clean null → the first-error-localizer program is CONCLUSIVELY CLOSED |
| **GAP-4 local generator (B2)** | a SOVEREIGN (local-generator) rule-exec gate that holds pass@2 lift vs control | the local generator cannot induce demo-perfect programs → logged decentralization gap |
| **Config-rule vocabulary (C)** | vocabulary-seeded grounding beats cold-start (CI95-excl-0) | no transfer → the config-rule vocabulary is logged as not-yet-compounding |
| **SteerConf calibration (D)** | code_humaneval detection > chance + LODO ECE below baseline on ≥2 non-FoVer domains | code stays at chance → multi-domain detector contract retired, domain-bound gap logged |
| **Publication gate** | paper_ready stays True (G1–G4) | — |

The milestone advances the north star if it raises ARC reproducible levels, OR grounds a new config-game
win-rule (de-risking the #1 lever), OR conclusively settles an open oracle-distinct question. A milestone
that re-measures a settled/retired axis without moving any of those is churn.
