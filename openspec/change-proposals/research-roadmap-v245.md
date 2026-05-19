# Research Roadmap v245: IsingVerifier Fix + Phase 4 Final + Ensemble v7b/7c + GateMate Flash + arXiv Gate

**Milestone:** 2026.05.245
**Status:** PROPOSED
**Date:** 2026-05-19
**Previous milestone:** 2026.05.244 — 5/13 tasks at terminal verdict;
phase4_final_status=blocked_precondition (exp2531 IsingVerifier + exp2532 Phase 4 v4 both
Gemini-CLI-failed); ensemble_v7b_regression unfixed (exp2533 also Gemini-failed); LaTeX
compile fixed (exp2536, abstract=205 words, submission_package_ready=True mechanically);
GateMate bitstream generated (exp2537, gatemate_bitstream_generated=True); Tier 0u viable
(exp2535, AUROC=0.96); JEPA pipeline integrated (exp2539); paper updated (exp2540);
arxiv_ready=False.

---

## What .244 Proved

Milestone .244 completed 5 of 13 tasks at terminal verdict (8 tasks either gate-blocked or
Gemini-CLI-failed). The five tasks that ran were a mix of successes:

**Major wins:**
- **arXiv LaTeX compile fixed** (exp2536): tectonic compiled `docs/arxiv-paper/main.tex`
  cleanly (main.pdf 393.9 KB). Abstract trimmed from 522 words → 205 words while preserving
  all five numbered contributions and headline numerics. The three-milestone LaTeX/abstract
  blocker is resolved. submission_package_ready=True at the mechanical layer.
- **GateMate A1 LUT mapping resolved + bitstream generated** (exp2537): The yosys↔nextpnr
  LUT mismatch turned out to be a false premise — yosys 0.64 emits CC_LUT2 for the XOR-heavy
  n=16 Ising tile and nextpnr-himbaechel 0.10 accepts it under `--vopt allow-unconstrained`
  with no workaround needed. `rtl/gatemate_ising_n16.cfg` (16392 bytes, max F=514.67 MHz,
  33/40960 CPE_LT utilization) is ready to flash. Only the openFPGALoader physical flash step
  remains for GateMate terminal-state graduation.
- **Tier 0u Logical Consistency Verifier implemented** (exp2535): Tier0uVerifier implemented
  in `python/carnot/verify/tier0u_logical_consistency.py`. Solo AUROC=0.96 on n=100 synthetic
  corpus, viable=True. Ready for ensemble integration (Group E, self-consistency class).
- **JEPA fast-path integrated into VerifyRepairPipeline** (exp2539): jepa_pipeline_integrated=True;
  JEPAFastPathPredictor + extract_response_features shipped in
  `python/carnot/pipeline/jepa_fast_path.py` with 14 passing tests. Caveat: fast_path_rate=1.0
  on synthetic corpus (feature-vector proxy not discriminative on natural text; trained JEPA
  model from exp2525 AUC=0.8889 needed for real discrimination).
- **Paper updated** (exp2540): paper_updated=True; citations_added=[arXiv:2512.02080,
  arXiv:2605.12484, arXiv:2605.03971, arXiv:2605.05134]; phase4_section_updated=True.

**Root cause of critical-path failures (new finding from .244):**
Milestones .244 activated with conductor running `AGENT_TYPE=gemini`. Tasks exp2530, exp2531,
and exp2533 (all `agent_type: claude`) were routed to the Gemini CLI, which was throwing a
JavaScript runtime error: `at async file:///usr/lib/node_modules/@google...`. This is a
DIFFERENT failure mode from the prior .242 `artifact_not_updated_past_bootstrap` failures and
from the prior 429 rate-limit failures documented in known-issues.md. The Gemini CLI is
currently non-functional at the JS runtime layer. Consequence: exp2531 (IsingVerifier) and
exp2533 (Ensemble v7b) both failed 3× at the execution layer before reaching any code-writing
step. Their dependent tasks (exp2532, exp2534) were gate-blocked.

**Gaps confirmed (entering .245):**
1. **IsingVerifier still a stub** — `class IsingVerifier: pass` in
   `python/carnot/verify/semantic_energy.py`. Four consecutive milestones (.241/.242/.243/.244)
   have been blocked at the Phase 4 empirical validation step because IsingVerifier has no
   `.energy()` method. The root code fix is unchanged from exp2531's plan; the execution
   routing must use codex (not Gemini) for the task to reach the write step.
2. **Ensemble v7b regression unfixed** — AUROC carries forward 0.9607 (regression from 0.9750)
   for the fourth consecutive milestone. The Tier 0r Group D fix planned as exp2533 never ran.
3. **Phase 4 empirical still unresolved** — the retire_if_same_verdict gate on exp2532 never
   fired. Phase 4 has been in `blocked_precondition` status across four milestones. The .245
   milstone is the final structured attempt; if Phase 4 fails again after IsingVerifier is
   implemented, it is permanently retired and paper §4 documents the honest negative.

---

## Three Biggest Gaps vs PRD Vision (entering .245)

### Gap 1: IsingVerifier Stub + Broken Execution Routing (Four Consecutive Milestones)

The identical code fix (implement `energy(step_text: str) -> float`) has been planned for four
milestones (.241 Phase 4 proxy, .242 step-level, .243 exp2519, .244 exp2531). Each time, either
the agent fell back to a proxy metric (exp2508) or was blocked at execution routing (exp2519
blocked_ising_verifier_not_available, exp2531 Gemini CLI JS error). **The fix for .245:** (a)
IsingVerifier implementation task (exp2544) uses `agent_type: codex` to bypass Gemini; (b) the
implementation is simple regex-based arithmetic claim checking — no GGUF needed, no hardware
dependency, no complex precondition chain. The code can be written and tested in a single codex
session against a Python interpreter.

**exp2544** implements IsingVerifier. **exp2545** runs the Phase 4 ARM-EBM v5 test with the
real IsingVerifier — the final structured attempt before permanent retirement
(`retire_if_same_verdict=true`).

### Gap 2: Ensemble v7b Regression — Three Milestones Without Fix

The group-conditional ensemble has been at AUROC=0.9607 (regression from 0.9750) since
exp2521 in .243. The fix (Tier 0r → Group D) has been planned three times (exp2510, exp2521,
exp2533) but executed only once (exp2521, which caused the regression by placing Tier 0r in
Group C). exp2533 never ran due to Gemini CLI error. **exp2546** (codex) finally executes the
Group D fix. Once restored, **exp2547** (also codex, gated on exp2546) integrates Tier 0u
(AUROC=0.96, viable from exp2535) into a new Group E (self-consistency class), potentially
pushing headline AUROC above 0.9750.

### Gap 3: GateMate Terminal-State Graduation — Only Physical Flash Remaining

GateMate has been "almost terminal" since exp2537 generated the bitstream. The only remaining
step is `openFPGALoader -c dirtyJtag -b olimex_gatemateevb rtl/gatemate_ising_n16.cfg`.
The DirtyJTAG programmer is confirmed onboard (1209:c0ca, /dev/ttyACM0/1); the board profile
is confirmed in openFPGALoader; user is in the `uucp` serial group. **exp2549** executes the
flash. Per CLAUDE.md Hardware-Task Continuity, GateMate terminal state =
`gatemate_bitstream_flashed=True` (n=16 Ising tile flashed + smoke-tested on hardware).

---

## Architecture Snapshot (entering .245)

```
Carnot Verification Stack (2026.05.245)

Verifier Ensemble (Group-Conditional Conformal, 9-10 verifiers):
  Group A (logprob):    Tier 0a (0.810), Tier 0b (0.8539), Tier 0f (0.8669)
  Group B (semantic):   Tier 0c (0.8831), Tier 0e (0.8896), Tier 0g (0.854)
  Group C (logic):      Tier 0d (0.588), Tier 0h (0.678)
  Group D (proof-path): Tier 0r (0.8256 synthetic) — REGRESSION PENDING FIX
  Group E (self-cons.): Tier 0u (0.96 synthetic) — PENDING INTEGRATION [exp2547]
  Headline AUROC = 0.9750 (carry-forward from .241 adversarially-verified)
  Regression from exp2521: ensemble v7 AUROC=0.9607 (Tier 0r Group C) — unfixed 4 milestones

IsingVerifier Status: CLASS IsingVerifier: PASS — STUB, NO METHODS
  Four milestones of blocked Phase 4 attempts trace to this single stub.
  Fix: exp2544 implements energy(step_text: str) -> float [CRITICAL PATH]

Phase 4 ARM-EBM Empirical Status: blocked_precondition (4 consecutive milestones)
  Next attempt: exp2545 with real IsingVerifier — FINAL attempt, retire_if_same_verdict=true

FR-11 Self-Learning Architecture:
  Tier 1 (online weights): operational (exp2500)
  Tier 2 (constraint memory): operational (exp2463, cross-session)
  Tier 3 (JEPA predictor): integrated into VerifyRepairPipeline (exp2539)
                            fast_path_rate=1.0 on synthetic — calibration needed [exp2553]
  Tier 4 (adaptive structure): prototype (exp2488)

arXiv Status:
  Gate 1 (phase1_ship): True
  Gate 2 (integrity_audit): True
  Gate 3 (phase4_validated): False — blocked; IsingVerifier fix → Phase 4 v5 → resolve
  Gate 4 (auroc_adversarially_verified): True (exp2498)
  LaTeX: COMPILES (exp2536, tectonic, abstract=205 words)
  paper_updated: True (exp2540, citations + Phase 4 section)

Hardware:
  KV260: kv260_hwh_generated=True, sd_card_detected=True; PYNQ URL unreachable [exp2550]
  GateMate A1: bitstream_generated=True (exp2537); flash pending [exp2549]
  PolarFire: TERMINAL (exp2501, energy_sanity_check_passed=True)
```

---

## Phase Structure

### Phase 0: Archive + Activate
- **exp2543**: Archive .244 → research-complete.yaml, activate .245

### Phase 1: Critical Path — IsingVerifier + Phase 4 Final
- **exp2544**: Implement IsingVerifier.energy(step_text: str) -> float
  — Fix the stub class. Regex-based arithmetic claim extraction + consistency check.
  No GGUF dependency. Test: energy('2+3=5')=0.0, energy('2+3=6')=1.0.
  [codex, 40 turns, prior_failures: exp2531 Gemini error]

- **exp2545**: Phase 4 ARM-EBM v5 — IsingVerifier NO FALLBACK (FINAL ATTEMPT)
  — Compute IsingVerifier.energy(step_text) per CoT step from telemetry manifest.
  Compute ARM-EBM energy per step (-sum token_logprobs). Pearson correlation test.
  Gate: |pearsonr|>0.30 AND p<0.05 AND n>=100 AND step_granularity_achieved=True.
  **retire_if_same_verdict=true** — if still below threshold, Phase 4 is permanently retired.
  [codex, 50 turns, gated on exp2544, prior_failures: exp2532/2519/2508/2486]

### Phase 2: Ensemble Fixes + Calibration
- **exp2546**: Ensemble v7b — Tier 0r → Group D (fix AUROC=0.9607 regression)
  — Move Tier 0r from Group C to dedicated Group D; re-run group-conditional calibration
  across 5 seeds. Gate: ensemble_v7b_auroc >= 0.975.
  [codex, 45 turns, prior_failures: exp2533 Gemini error, exp2521 regression]

- **exp2547**: Ensemble v7c — Add Tier 0u to Group E (11th verifier)
  — Integrate Tier 0u (AUROC=0.96, viable from exp2535) as Group E (self-consistency class).
  Gate: ensemble_v7c_auroc >= 0.975 (no regression from v7b baseline).
  [codex, 40 turns, gated on exp2546.regression_resolved==true]

- **exp2548**: Adaptive Conformal v2 + ACSE — finally unblocked
  — Prompt-adaptive calibration (arXiv:2604.13991) + ACSE semantic entropy (arXiv:2605.04295).
  Baseline: ensemble_v7b_auroc from exp2546.
  [codex, 45 turns, gated on exp2546.ensemble_v7b_auroc>=0.975]

### Phase 3: Hardware Continuity
- **exp2549**: GateMate A1 Physical Flash (openFPGALoader)
  — Flash `rtl/gatemate_ising_n16.cfg` via onboard DirtyJTAG. Smoke-test Ising n=16 tile.
  Terminal state gate: gatemate_bitstream_flashed=True.
  [codex, 35 turns, track: hardware, prior_failures: none — first flash attempt]

- **exp2550**: KV260 SD Card Flash v3 — Alternate PYNQ Image Source
  — SD card detected (exp2538). PYNQ v3.0 GitHub URL was unreachable. Try:
  (a) alternate PYNQ mirrors/releases, (b) xilinx.com direct download,
  (c) document updated operator commands if URL still unreachable.
  [codex, 30 turns, track: hardware, prior_failures: exp2538/2526]

### Phase 4: New Verifier + arXiv Package
- **exp2551**: Tier 0t Dynamical System Verifier (arXiv:2605.05134)
  — Implement trajectory-deviation score from token logprob sequences as Tier 0t.
  No GGUF — computed from existing telemetry manifest logprob fields.
  Gate: tier0t_auroc > 0.70.
  [codex, 40 turns, prior_failures: none — new verifier class]

- **exp2552**: arXiv Pre-Submission Package v3 — Final Operator Checklist
  — LaTeX compiles (exp2536), paper updated (exp2540). Incorporate Phase 4 final outcome
  from exp2545. Generate final operator action checklist (CCS tagging, author list,
  submission form fields). Gate: submission_package_ready=True.
  [codex, 35 turns, prior_failures: exp2536/2527]

### Phase 5: Continuous Self-Learning + Capstone
- **exp2553**: FR-11 JEPA Fast-Path Threshold Calibration (Continuous Self-Learning)
  — JEPA fast_path_rate=1.0 on synthetic corpus because feature-vector proxy scores
  all natural text as uniformly low-violation. Fix: (a) load trained JEPA model from exp2525
  and calibrate threshold against real telemetry data, OR (b) improve feature extraction to
  include token-level logprob_variance from existing telemetry manifest.
  Gate: jepa_fast_path_rate < 0.95 on mixed corpus (shows non-trivial discrimination).
  [codex, 40 turns, continuous_self_learning_task: true, prior_failures: exp2539]

- **exp2554**: Capstone v245 — Milestone 2026.05.245 Final Synthesis (NO HARD GATE)
  — Synthesize all exp2543-exp2553 outcomes. Determine Phase 4 final status, headline AUROC,
  arXiv readiness, operator_recommendation.
  [claude+opus, 100 turns, requires_claude: true, NO HARD GATE]

- **exp2555**: Operational Retrospective v245
  [codex, 20 turns]

---

## Dependency Graph

```
exp2543 (archive)
    |
    ├─── exp2544 (IsingVerifier impl) ──── exp2545 (Phase 4 v5, FINAL) ──┐
    |                                                                       |
    ├─── exp2546 (Ensemble v7b Group D) ──┬── exp2547 (Ensemble v7c Tier 0u)│
    |                                      └── exp2548 (Adaptive Conformal) │
    |                                                                        |
    ├─── exp2549 (GateMate flash)                                            |
    ├─── exp2550 (KV260 SD flash v3)                                         |
    ├─── exp2551 (Tier 0t Dynamical System)                                  |
    ├─── exp2552 (arXiv package v3) ← depends on exp2545 outcome ───────────┘
    ├─── exp2553 (FR-11 JEPA calibration)
    |
    └─── exp2554 (Capstone v245, reads all above) ──── exp2555 (Retro)
```

---

## Hardware Requirements

| Board | State entering .245 | Target this milestone | Terminal state |
|---|---|---|---|
| GateMate A1-EVB-2M | bitstream_generated=True (exp2537) | Physical flash via openFPGALoader | gatemate_bitstream_flashed=True |
| KV260 | hwh_generated=True, SD detected, URL unreachable | Alternate PYNQ source flash | board-level latency transcript |
| PolarFire SoC | TERMINAL (exp2501) | Optional/opportunistic | ALREADY GRADUATED |

---

## Decentralization Compliance (Rules 1-7)

- **Rule 1 (local-first):** All experiments use locally-available code, corpus, and tools.
  Phase 4 uses telemetry manifest logprob fields (already on disk). No closed-weight GGUF
  required for critical path (IsingVerifier is pure Python regex).
- **Rule 2 (closed models optional):** No experiment mandates a closed-weight model.
- **Rule 3 (distribution mirroring):** No new artifacts published this milestone.
- **Rule 4 (multiple integration surfaces):** JEPA calibration (exp2553) touches pipeline
  surface; ensemble updates touch Python API; hardware track touches FPGA surface.
- **Rule 5 (hardware portability):** GateMate flash (exp2549) advances open-FPGA sovereignty.
- **Rule 6 (data minimization):** No closed-weight calls in any experiment.
- **Rule 7 (no vendor abstractions in core):** All new code in verify/ and pipeline/ follows
  BaseVerifier protocol; no vendor SDK imports in core.

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml` for retired experiment IDs matching .245 task scopes:
- exp2091 (Tier 1 CSL Grammar Updates, Gemini bail-out): not matched by any .245 task.
- Legacy slowest-5 retirements (exp260, 308, 309, 346, etc.): not matched.
- HalluSAEGeometricProbe (scope-retired): not matched (Tier 0t is dynamical system, different
  from SAE geometry probe).
- JEPA discriminative OOD (exp887/883/799/804/809): not matched (exp2553 calibrates existing
  pipeline JEPA, does not re-propose the OOD discriminative approach).

**Result: 0 retired experiment ID patterns matched by .245 tasks.** No prior_failures blocks
required by manifest; prior_failures blocks present for .244 execution failures.

---

## Failed-Experiment Rerun Compliance Table

| Task | Prior failure(s) | What changed | retire_if_same_verdict |
|---|---|---|---|
| exp2544 (IsingVerifier) | exp2531 (Gemini CLI JS error) | agent_type: codex bypasses Gemini | false |
| exp2545 (Phase 4 v5) | exp2532 (gate-blocked), exp2519 (blocked_ising_verifier_not_available), exp2508 (methodology fallback), exp2486 (pearsonr=0.108 noise floor) | exp2544 implements IsingVerifier; agent_type: codex; NO FALLBACK; retire if same result | true |
| exp2546 (Ensemble v7b) | exp2533 (Gemini CLI JS error), exp2521 (AUROC=0.9607 regression) | agent_type: codex; Group D approach unchanged from .244 plan | false |
| exp2547 (Ensemble v7c Tier 0u) | None (new scope — first Tier 0u ensemble integration) | — | false |
| exp2548 (Adaptive Conformal v2) | exp2534 (gate-blocked by exp2533 failure), exp2524 (not_run), exp2511 (blocked_ensemble_v7_not_available) | exp2546 builds ensemble v7b prerequisite | false |
| exp2549 (GateMate flash) | None (first flash attempt; exp2477 erroneously declared terminal) | exp2537 generated bitstream; openFPGALoader confirmed | false |
| exp2550 (KV260 SD flash v3) | exp2538 (blocked_pynq_url_unreachable), exp2526 (sd_card_not_detected) | SD card detected (exp2538); try alternate image source | false |
| exp2551 (Tier 0t) | None (new scope — dynamical system verifier) | — | false |
| exp2552 (arXiv v3) | exp2536 (submission_not_ready pending Phase 4), exp2527 (latex_compile_failure) | exp2536 fixed LaTeX; exp2545 resolves Phase 4 | false |
| exp2553 (FR-11 JEPA calibration) | exp2539 (fast_path_rate=1.0, no discrimination) | exp2553 calibrates threshold; uses real telemetry logprob_variance | false |

---

## Agent Routing

| Task | agent_type | model | Justification |
|---|---|---|---|
| exp2543 (archive) | codex | gpt-5.5 | Mechanical archival task |
| exp2544 (IsingVerifier) | codex | gpt-5.5 | Single-file Python implementation with clear spec |
| exp2545 (Phase 4 v5) | codex | gpt-5.5 | Compute-bound; pearsonr is deterministic numerical check |
| exp2546 (Ensemble v7b) | codex | gpt-5.5 | Calibration code follows established pattern |
| exp2547 (Ensemble v7c) | codex | gpt-5.5 | Same pattern as v7b; Tier 0u already implemented |
| exp2548 (Adaptive Conformal) | codex | gpt-5.5 | Established conformal calibration pattern |
| exp2549 (GateMate flash) | codex | gpt-5.5 | Shell command sequence; openFPGALoader syntax known |
| exp2550 (KV260 SD flash v3) | codex | gpt-5.5 | URL search + dd flash; deterministic |
| exp2551 (Tier 0t) | codex | gpt-5.5 | Single-class implementation; pattern matches Tier 0u |
| exp2552 (arXiv v3) | codex | gpt-5.5 | Template-driven checklist generation |
| exp2553 (FR-11 JEPA calibration) | codex | gpt-5.5 | Feature extraction + threshold calibration; deterministic |
| exp2554 (Capstone) | claude | opus | Multi-artifact synthesis across 12 files + cross-phase judgment; meets all 3 positive criteria |
| exp2555 (Retro) | codex | gpt-5.5 | Templated retrospective |

**12 codex/gpt-5.5 (92.3%), 1 claude+opus (7.7%)**

Positive-criterion justification for exp2554 (capstone):
1. Codex has consistently been unable to synthesize cross-phase multi-artifact reasoning (12+ files, open-ended arXiv readiness judgment under ambiguity).
2. Requires reading 12 distinct JSON artifacts + cross-phase reasoning — meets the 5+ files multi-tool-choreography criterion.
3. Phase 4 final status determination and arXiv recommendation under ambiguity requires open-ended judgment — no deterministic gate substitutes.

---

## Completed Milestones Reference

| Milestone | Key result | AUROC | Phase 4 |
|---|---|---|---|
| 2026.05.239 | FR-11 JEPA complete; paper integrity fixed | 0.9351 (flagged) | Failed ODAR |
| 2026.05.240 | Group-conditional 0.975 confirmed | 0.9750 | ARM-EBM r=0.108 |
| 2026.05.241 | AUROC adversarially verified | 0.9750 | MISSING+noise |
| 2026.05.242 | Phase 4 step-level (fallback) | 0.9750 | Fallback used |
| 2026.05.243 | Tier 0r impl; KAN rebuilt | 0.9750 | blocked_precondition |
| 2026.05.244 | LaTeX fixed; GateMate bitstream; Tier 0u | 0.9750 | blocked_precondition (Gemini failure) |
| **2026.05.245** | **IsingVerifier + Phase 4 FINAL + Ensemble v7b/7c + GateMate flash** | **target ≥0.975** | **FINAL ATTEMPT** |
