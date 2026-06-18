# Research Roadmap v407 — 2026.06.407

**Headline:** `.406 attacked the .405 first-error localization NULL (F1 0.096) with *synthetic*
verifiable process data and came back mostly null — the synthetic localizer reported a "win"
(FoVer F1 1.0) but the skeptic-proof **quarantined it as pure POSITION BIAS**
(`localizer_win_is_genuine=false`, `beats_position_only_baseline=false`,
`template_ablation_drop=0.0`). `.407 gives the one alive **oracle-distinct** vehicle ONE genuine
attempt: train the first-error localizer on **REAL intervention data** (arXiv:2601.14209) with the
deconfounding controls .406 failed (content-blind position-only baseline + template-family holdout),
**retire-if-it-ties-position-only-again**. The PRIMARY work is the operator-MANDATORY **ARC north
star**: the .406 lookahead-fidelity gate (0.73–0.875) only *confirmed* the blocker — `.407
DECOMPOSES it into per-mechanic **executable unit tests** (which transition the world model gets
wrong), fixes the engine test-by-test, and plans deeper. Plus active-learning self-learning
(arXiv:2504.10559) and a repaired cross-domain calibration contract (arXiv:2602.07842).

**`paper_ready = True` (FoVer 0.9131, G1–G4) is the FROZEN headline — `.407 adds genuine lenses,
never a substitute.**

**Provenance:** outer-loop planner (Claude Opus 4.8), 2026-06-18, after all 11 `.406 tasks
completed. SOTA mapping from the `.406 ingestion exp4398 (`flagged_for_v407 =
intervention_active_real_first_error_deconfounding_v407`), arXiv-verified via the reliable channel.

---

## 1. What .406 proved (the honest scorecard)

Read via `scripts/summarize_artifact.py`; capstone exp4401
(`v406_localizer_localizes_but_not_genuine_compounds_false_calibrated_false_arc_levels_34_publication_ready`).

| Axis | `.406 task | Result |
|---|---|---|
| **LOCALIZER (oracle-distinct)** | exp4392 → exp4393 | **NOT GENUINE.** Synthetic-data localizer reported FoVer F1 1.0 / GAP-4 ARC 0.69 vs the 0.096 baseline, but the skeptic-proof **quarantined it**: `beats_position_only_baseline=false`, `template_ablation_drop=0.0`. Pure position bias; the F1=1.0 is the IMPLAUSIBLE_PERFECT tell. **Synthetic template-injected data does not carry the REAL first-error distribution.** |
| **SELF-LEARNING / compounds** | exp4396 | **false (saturated null).** Learning curve F1=1.0 FLAT across corpus 566→5661 — the artifact saturates trivially; SIZE-only growth has no headroom. |
| **CROSS-DOMAIN calibration** | exp4397 | **false.** code_humaneval detection AUROC 0.98 but underpowered (n=100); multi-valid-output / base-rate confounds unseparated; leave-one-domain-out ECE not below the uncalibrated baseline. |
| **ARC north star** | exp4394 + exp4395 | **STILL 34 levels / 17 games (0 new).** Lookahead-fidelity gate UNREACHABLE: ar25 0.73, tu93 0.80, lp85 0.83, tr87 0.86, tn36 0.875 — all below ~0.95. Gating on AGGREGATE fidelity only confirmed the blocker. |
| **publication gate** | exp4401 | **paper_ready=True** (G1∧G2∧G3∧G4, FoVer 0.9131), `unmet_gates: []` — carried unchanged. |

**The SOTA-ingestion's forward pointer (exp4398):** `flagged_for_v407 =
intervention_active_real_first_error_deconfounding_v407` — the synthetic route is a dead end; use
REAL intervention first-error data (arXiv:2601.14209), stratified by position + template family,
with the deconfounding controls built into the headline.

**Carried RETIRED / SETTLED (do NOT re-propose):** cross-game ARC learned-verifier / value transfer
(3 nulls exp4318/4331/4342, exclusion manifest); in-generation DiffusionGemma conversion (4th block
exp4374); LLM-generated-heuristic efficiency (exp4370 null — the deployed linear action-cost
heuristic exp4364 is the moat); cross-domain SELECTION (exp4314); the SYNTHETIC localizer route
(.406 artifact-confounded). **HARD RULE:** conductor STOOD-DOWN on TRM training (no task launches
TRM training, pkill/kill against train.py, or writes results/trm_runs/). A2D2 (2606.13565) + SEPO
(2502.01384) verifier-as-reward generator-training are OUT-OF-BAND / operator-owned.

---

## 2. The three biggest gaps (current state vs PRD / north-star vision)

1. **The oracle-distinct verifier is still UNPROVEN, and the localizer keeps confounding.** North-star
   §0/§5 + the 2026-06-14 P0 directive: the deep, defensible, still-open claim is an oracle-DISTINCT
   (learned/energy) verifier. The detector DETECTS (AUROC 0.92–0.98) but the two attempts to make it
   LOCALIZE the earliest error came back null (.405 F1 0.096; .406 position-bias artifact). **Gap:**
   a localizer trained on *real* first-error structure that survives the position/template controls —
   or an honest retirement of the localizer-as-headline. `.407 closes this either way (exp4403/4404,
   `retire_if_same_verdict: true`).

2. **The ARC-AGI-3 north star is fidelity-blocked, and aggregate gating cannot fix it.** North-star §0:
   ARC-AGI-3 solve-rate is THE destination. We are stuck at 34/17 because world-model lookahead
   fidelity is 0.73–0.875. **Gap:** a per-mechanic decomposition — which *specific* transition each
   world model gets wrong — turned into executable unit tests and fixed test-by-test, raising fidelity
   past the planning threshold (exp4405/4406, the operator-MANDATORY E3 work).

3. **Continuous self-learning has no live compounding vehicle on the accuracy axis, and cross-domain
   detection is uncalibrated.** research-program.md mandates ≥1 self-learning experiment/milestone;
   the .406 size-only localizer-compounding saturated. The verifier is domain-bound (math strong);
   ARC-AGI-3 needs calibrated multi-domain detection (north-star §0 step 2). **Gap:** ACTIVE-learning
   selection that compounds where size-only growth could not (exp4407), and a calibrated multi-domain
   detector contract with proper pools + base-rate separation (exp4408).

---

## 3. Architecture — where .407 acts

```
                          ┌──────────────────────────────────────────────────┐
   GENERATOR (commodity)  │  open local LLM (Qwen/Gemma GGUF) · TRM refiner   │   ← NOT trained in-loop
                          └──────────────────────────────────────────────────┘
                                              │ candidates / traces
                                              ▼
   ┌───────────────────────────  CARNOT ENERGY VERIFIER (the value-add)  ───────────────────────────┐
   │                                                                                                  │
   │  A. oracle-distinct LOCALIZER (FoVer/math)        B. ARC executable world-model verifier         │
   │     REAL intervention first-error data               per-mechanic EXECUTABLE UNIT TESTS           │
   │     (arXiv:2601.14209) + position-only +             decompose the fidelity blocker               │
   │     template-family controls → exp4403/4404            (arXiv:2606.16070) → exp4405/4406           │
   │     verifier_is_oracle=FALSE                          verifier_is_oracle=TRUE (execution-grounded) │
   │                                                                                                    │
   │  C. ACTIVE-learning self-learning (exp4407)        D. cross-domain detection CALIBRATION (exp4408) │
   │     uncertainty + position-diversity                  proper pools + base-rate separation          │
   │     (arXiv:2504.10559) verifier_is_oracle=FALSE       (arXiv:2602.07842) verifier_is_oracle=FALSE  │
   └────────────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
            E. infra/hygiene/capstone: SOTA-ingest→.408 · registry+GAP-4 guard · KV260 · scorecard
```

---

## 4. Phase descriptions

### Phase 0 — Transition (exp4402)
Archive `.406 → activate `.407; assert YAML parses; pre-test gate green; record the TRUE .406
close-state (localizer not genuine = position bias; compounds/calibration false; ARC 34/17 fidelity-
blocked; flagged_for_v407 = real-intervention deconfounding; paper_ready=True). `agent_type: codex`.

### Phase A — HEADLINE: deconfound the oracle-distinct first-error localizer (verifier_is_oracle=false)
- **exp4403 (A1)** — REAL intervention first-error data (arXiv:2601.14209): collect verifier-checked
  real traces where a single-step intervention redirects a failed trajectory; stratify by first-error
  position + template family; train a contrastive localizer; evaluate on a HELD-OUT REAL split with
  the **deconfounding controls built in** (a content-blind position-only baseline + a template-family
  holdout). Emit `localizer_genuinely_beats_position_only` BARE bool. `prior_failures: exp4392`,
  `retire_if_same_verdict: true` (a third position-bound tie retires the localizer-as-headline).
- **exp4404 (A2, gated on A1)** — typed first-error taxonomy cross-check (arXiv:2603.25412) + cross-
  domain (GAP-4 ARC + held-out FoVer family) generalization of the genuine localizer. Emit
  `localizer_generalizes_typed` BARE bool. The .406 lesson: a "win" without the controls is meaningless.

### Phase B — ARC north star (operator MANDATORY 2026-06-17; verifier_is_oracle=true)
- **exp4405 (B1)** — E3 deeper (lp85 L6, tu93 L5, tn36 L8, tr87 L7) via **per-mechanic executable
  unit tests** (arXiv:2606.16070): the .406 aggregate fidelity gate only confirmed the blocker; here
  decompose it — write an executable unit test for each mismatched transition, fix the engine to pass
  it, raise per-mechanic fidelity, THEN plan deeper. `prior_failures: exp4394`.
- **exp4406 (B2)** — E3 blocked-mechanic tails (ar25 L2 action7-undo-stack, ka59 L2 step-counter-HUD +
  object-relevance, ft09 L2) via per-mechanic unit tests on the NAMED registers. `prior_failures:
  exp4395`. Both report mechanic-test pass/fail SEPARATELY from any solve claim; only offline-
  reproduced levels count; +1..+n per game (Incremental-Progress Scoping).

### Phase C — Continuous self-learning (mandated; verifier_is_oracle=false)
- **exp4407** — ACTIVE-learning compounding (arXiv:2504.10559): the .406 SIZE-only growth saturated
  (exp4396). NEW mechanism = active selection (uncertainty + first-error-POSITION diversity) of REAL
  first-error traces — does active selection compound where size-only could not, on the best available
  first-error localizer (A1's if genuine, else the deconfounded corpus)? Emit `localizer_compounds`
  BARE bool. `prior_failures: exp4396`. Keeps template-family holdouts + a position-only control IN the
  active loop. NOT gated (the mandated slot must produce a reading).

### Phase D — Cross-domain detection calibration + pool expansion (verifier_is_oracle=false)
- **exp4408** — repair the false calibrated-multi-domain contract (arXiv:2602.07842): build PROPER
  pools (code HumanEval n≥300, GSM n≥300 — not the .406 underpowered n=100) from EXISTING cached
  candidates; semantic confidence aggregation across multi-valid answers + per-domain base-rate
  separation; leave-one-domain-out calibration. Emit `detection_calibrated_multi_domain` BARE bool.
  `prior_failures: exp4397`. A domain at chance is LOGGED as a missing-verifier gap.

### Phase E — Infra + hygiene + capstone
- **exp4409** — SOTA-ingestion → `.408 (reliable channel only; `/deep-research` banned in-loop; every
  method a real arXiv ID). Emit `flagged_for_v408`.
- **exp4410** — registry/gaps hygiene + GAP-4 execution regression guard + verify the capstone
  verifier_is_oracle stamping fix is durable. Emit `regression_guard_passed`.
- **exp4411** — KV260 continuity (opportunistic per north-star §3; SSH-reachability only, NEVER a host
  SD-card precondition).
- **exp4412** — capstone `.407: the scorecard + headline decision (localizer genuine? compounds?
  calibrated multi-domain? ARC reproducible_total_levels?) + G1–G4 via `publication_gate.py`; robust
  aggregate-available-report-gaps helper; HONOR verifier_is_oracle (no CIRCULAR_MOAT_OVERCLAIM).

---

## 5. Dependency graph

```
exp4402 (archive/activate)
   ├─ exp4403 (A1 localizer: REAL intervention data) ──gated──▶ exp4404 (A2 typed + cross-domain)
   ├─ exp4405 (B1 ARC deeper: per-mechanic unit tests)          [verifier_is_oracle=true]
   ├─ exp4406 (B2 ARC tails: per-mechanic unit tests)           [verifier_is_oracle=true]
   ├─ exp4407 (C active-learning self-learning)                 [reads A1 if present; not gated]
   ├─ exp4408 (D cross-domain calibration repair)
   ├─ exp4409 (E SOTA-ingest → .408)
   ├─ exp4410 (E registry/gaps hygiene + GAP-4 guard)
   ├─ exp4411 (E KV260 continuity, opportunistic)
   └─ exp4412 (E capstone .407)  ◀── aggregates 4403/4404/4405/4406/4407/4408
```

`exp4404` carries a structured `gated_on` (exp4403 `localizer_genuinely_beats_position_only == true`)
so the conductor skips the Sonnet call when A1 does not produce a genuine localizer. The capstone is
UNGATED and uses the robust aggregate-available helper (no hard-block-all-False).

---

## 6. Hardware requirements

| Phase | Substrate | Notes |
|---|---|---|
| A (localizer) | CPU, cached candidates | symbolic prefix-checks + a contrastive localizer fit; a SOTA GGUF only if needed to realize an intervention trace into NL (cache-checked first) |
| B (ARC E3) | codex/gpt-5.5 induction + offline sim | `live_llm_inference`; offline `environment_files/<game>/`; zero leaderboard submission |
| C / D | CPU, cached candidates | active-learning fit + calibration; zero new live inference |
| E | CPU (aggregation) + KV260 SSH (opportunistic) | capstone is aggregation-only; KV260 reachability via `ssh kria`, never `/dev/mmcblk*` |

No GPU is REQUIRED for the headline (A is cached/CPU) — deliberately not hostage to GPU/GGUF/PR-binary
infra. The dual RTX 3090s are available for any optional NL-realization of intervention traces.

---

## 7. SOTA mapping (`.406 ingestion exp4398, verified 2026-06-18)

| arXiv | Method | `.407 use | verifier_is_oracle |
|---|---|---|---|
| 2601.14209 | InT self-proposed interventions for first-error credit assignment | exp4403 HEADLINE — REAL intervention first-error data, deconfounded | false |
| 2603.25412 | Reasoning Safety Monitor typed step-localization taxonomy | exp4404 — typed cross-check + cross-domain audit | false |
| 2504.10559 | ActPRM active learning for PRM training | exp4407 — active selection where size-only saturated | false |
| 2602.07842 | Semantic Confidence Aggregation (multi-answer calibration) | exp4408 — repair the multi-domain calibration contract | false |
| 2606.16070 | Mind-Studio executable world models + lookahead | exp4405/4406 — per-mechanic executable unit tests | true |
| 2605.05138 / 2605.25931 / 2512.22336 | Executable World Models / AERA / Agent2World | carried ARC baselines | true |
| 2606.13565 / 2502.01384 | A2D2 / SEPO (verifier-as-reward generator training) | OUT-OF-BAND, operator-owned, NOT auto-run | — |

---

## 8. Invariants carried from .406

- **HARD RULE — no TRM training in-loop.** No task launches TRM training, runs pkill/kill against
  train.py, or writes results/trm_runs/. Qwen FORBIDDEN as the TRAINED base (Spurious-Rewards
  confound); Qwen/Gemma GGUF as an off-policy judge/generator is fine.
- **Circularity / Oracle-Distinctness Discipline.** Every verifier-value task declares
  `verifier_is_oracle` honestly: the localizer / calibration / compounding claims are oracle-DISTINCT
  (`false` + matched control + CI95-excl-0); ARC E3 solves are execution-grounded (`true`, ARC
  progress, NOT a moat headline).
- **paper_ready=True (FoVer 0.9131) is the FROZEN headline** — `.407 adds genuine lenses, never a
  substitute. NO autonomous edits to `docs/index.html` / README / paper prose.
- **Online ARC stays operator-gated** — NO leaderboard submission; only offline-reproduced levels
  count.
- **Do NOT re-propose** cross-game ARC value transfer (retired), in-generation DiffusionGemma
  (retired), LLM-heuristic efficiency (settled), cross-domain SELECTION (retired), or the SYNTHETIC
  localizer route (artifact-confounded).
- **SOTA models** — any task needing an LLM names a SOTA GGUF in MODEL_SPECS
  (unsloth/Qwen3.6-35B-A3B-GGUF, unsloth/gemma-4-31B-it-GGUF, unsloth/gemma-4-26B-A4B-it-GGUF),
  cache-checked in a PRECONDITIONS step; the GGUF tokenizer is embedded (load via the `.gguf` path,
  never `AutoTokenizer.from_pretrained` on a `-GGUF` repo).
