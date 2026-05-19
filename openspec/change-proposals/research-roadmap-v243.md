# Research Roadmap v243: Phase 4 ARM-EBM v3 (No Fallback) + Tier 0r Implementation + Ensemble v7 + KAN Restore + arXiv Submission Prep

**Milestone:** 2026.05.243
**Status:** PROPOSED
**Date:** 2026-05-19
**Previous milestone:** 2026.05.242 — 6/9 capstone-input experiments at terminal verdict; phase4_validated_any=True per literal gates (exp2508 pearson_r=-0.4266) BUT step_granularity_achieved=False (methodology fallback); KV260 .hwh generated; FR-11 Tier 2 memory operational; paper-v6 updated; arxiv_ready=True per literal gates with Gate 3 methodology caveat.

---

## What .242 Proved

Milestone .242 had 6 of 9 capstone-input tasks at a terminal verdict (3 blocked: exp2509, exp2510, exp2511).

**Major wins:**
- **KV260 .hwh generated** (exp2514): vivado v2025.2.1 successfully generated the hardware handoff file from the Vivado block design. Physical SD card flash documented as a manual operator step. KV260 status: hwh_generated_flash_pending_operator.
- **FR-11 Tier 2 memory-augmented threshold learning** (exp2512): memory_augmented_auroc=1.0000 vs no-memory baseline 0.7803 on a 32-example per-domain synthetic corpus. SQLite schema v1.0 instantiated; FR-11 Tier 2 cross-session memory operational.
- **Paper-v6 final write-through** (exp2515): 5 corrigendum items resolved, §3 and §6 updated with .241+.242 findings. paper_updated=True, arxiv_ready=True per literal gate check.

**Gaps confirmed (entering .243):**
1. **Phase 4 Gate 3 methodology caveat** — exp2508 self-declared phase4_validated_step_level=True with pearson_r=-0.4266, BUT step_granularity_achieved=False: the designed raw-logprob per-CoT-step methodology was not executed; the experiment fell back to response-level SemanticEnergy proxy (same class as the previously-failed exp2486). Gate 3 flips literally but the structural test was not performed. Operator decision needed: (a) accept literal gate and proceed to arXiv, (b) re-run with proper IsingVerifier step-level energy, or (c) revise paper §4 to reflect Phase 4 as empirically unsupported.
2. **Ensemble v7 chain completely stalled** — exp2509 (HalluGuard Tier 0s) blocked_no_eval_corpus; exp2510 (Tier 0r integration) blocked_tier0r_not_implemented; exp2511 (Adaptive Conformal v2) blocked_ensemble_v7_not_available. Three-task chain blocked at its root: Tier 0r was never actually implemented (exp2504 showed viability but the code was not captured). AUROC headline still carried forward from .241 at 0.9750.
3. **KAN model missing from disk** — exp2513 (KAN Multilevel) blocked_kan_not_found. The certified-deployment-ready KAN (AUROC=0.994, certified_coverage=0.833 from exp2489) cannot be located. The AUROC=0.994 headline is at risk if the model cannot be reconstructed.

---

## Three Biggest Gaps vs PRD Vision (entering .243)

### Gap 1: Phase 4 Methodology Caveat — Clean Re-Run Required for Unambiguous Gate 3

The literal Gate 3 flips on exp2508's self-declared field, but the capstone flags a critical corrigendum: `energy_proxy_used='semantic_energy_fallback'`, `step_granularity_achieved=False`. The designed methodology — compute E_step = -sum(log_p(token_i)) directly from raw token logprobs at per-CoT-step granularity and correlate with IsingVerifier.energy(step_text) — was NOT executed. The experiment fell back to the same response-level SemanticEnergy proxy class that exp2486 used and which failed (pearsonr=0.108).

**What is structurally different in .243 exp2519:**
- The experiment MUST check for IsingVerifier.energy() importability as a PRECONDITION
- If IsingVerifier.energy() is available: use it directly on each CoT step text
- If IsingVerifier.energy() is not available: emit blocked_ising_verifier_not_available — DO NOT fall back to SemanticEnergy
- The ARM-EBM formula E_step = -sum(log_p(token_i)) is computed from raw token_logprobs in the telemetry manifest — this requires the token_logprob field to be populated, not the semantic embedding of the text
- A positive result (step_granularity_achieved=True, |pearsonr| > 0.30) provides a clean Gate 3 flip
- A negative result (retire_if_same_verdict=true) retires Phase 4 and triggers the arXiv proceed-without-Phase-4 path

**arXiv implication:** the arXiv submission package prep (exp2527) proceeds regardless of exp2519 outcome — if Phase 4 validates, the paper §4 is strengthened; if Phase 4 retires, the paper §4 is revised to document the negative result transparently.

### Gap 2: Tier 0r Implementation (Root Cause of Three-Task Chain Block)

Tier 0r (Curry-Howard soft-typed proof-path, arXiv:2510.01069 ICLR 2026) showed AUROC=0.9123 viability in exp2504, but that experiment tested the verifier concept without persisting the implementation as an importable Python class. When exp2510 tried to import `from carnot.verify.tier0r_curry_howard import Tier0rVerifier`, the file didn't exist.

**exp2520 must actually write the code:** implement Tier0rVerifier in `python/carnot/verify/tier0r_curry_howard.py` following the existing BaseVerifier interface. Once implemented, exp2521 can run group-conditional calibration with 10 verifiers, exp2524 can run adaptive conformal calibration, and the ensemble headline AUROC can advance beyond the 0.9750 carry-forward.

### Gap 3: KAN Model Restoration

The KAN Tier 1 verifier (AUROC=0.994, certified_coverage=0.833, established in exp2489) cannot be located on disk. exp2523 must search all likely paths, and if not found, retrain from the available training corpus. The multilevel training improvement (arXiv:2603.04827) can only be tested once the KAN baseline is restored.

---

## Architecture Snapshot (entering .243)

```
Tier 0 Verifiers (conformal p-value ensemble — v6, 9 active verifiers):
  Group A (logprob-class):
    Tier 0a: SemanticEnergy (AUROC=0.810)
    Tier 0b: HALT (AUROC=0.8539)
    Tier 0f: PCIB (AUROC=0.8669)
  Group B (semantic-class):
    Tier 0c: FregeLogic (AUROC=0.8831)
    Tier 0e: LogCons Hierarchical (AUROC=0.8896)
    Tier 0g: LaaB Meta-Judgment (AUROC=0.854)
  Group C (logic-class):
    Tier 0d: DiffuTruth (AUROC=0.588, marginal but included)
    Tier 0h: NCO (AUROC=0.678)
  Excluded (non-viable / noise-floor):
    Tier 0i: ODAR routing (AUROC=0.5584)
    Tier 0p: LLM-as-Judge (AUROC=0.6412)
    Tier 0q: Spilled Energy (AUROC=0.4903, retired .241)
  PENDING IMPLEMENTATION for ensemble v7:
    Tier 0r: Curry-Howard (viable, exp2504 AUROC=0.9123 — code NOT written yet)
  PENDING PROTOTYPE for ensemble v8:
    Tier 0s: HalluGuard NTK-based (arXiv:2601.18753 — needs eval corpus)
  Group-conditional calibration (Fisher combination):
    mean AUROC = 0.9750 (adversarially verified, cite-safe, .241/.242 carry-forward)
    HIVE peer baseline: 0.9236 (+0.0514 gap, BREACHED)

Tier 1: KAN (AUROC=0.994 from exp2489, certified_coverage=0.833) [MODEL MISSING FROM DISK]
FR-11 Self-Learning Loop:
  Tier 1: Online constraint reweighting (operational)
  Tier 2: Cross-session SQLite memory (operational, exp2512, schema v1.0)
  Tier 3: JEPA predictor (operational, exp2475, jepa_violation_auc=0.7633)
  Tier 4: Adaptive energy landscape (operational, exp2488/exp2500)
Paper-v6: updated through .242 (exp2515), arxiv_ready=True per literal gates
Gate 3 caveat: Phase 4 exp2508 methodology fallback — clean re-run pending

Hardware:
  KV260: kv260_hwh_generated=True (exp2514), flash pending manual operator SD card
  PolarFire: TERMINAL (exp2501, energy_sanity_check_passed=True)
  GateMate: TERMINAL (exp2453, bitstream_flashed=True)
```

---

## Milestone .243 Phases

### Phase 0: Archive + Activate (1 task)
Standard milestone activation. Archives .242 results to research-complete.yaml and swaps to .243 roadmap.

### Phase 1: Phase 4 ARM-EBM v3 — IsingVerifier NO FALLBACK (1 task)
**Critical path to unambiguous arXiv Gate 3.** Run the ARM-EBM step-level energy correlation with IsingVerifier.energy() on step text. If IsingVerifier not available → emit blocked_ising_verifier_not_available (NO SemanticEnergy fallback). If the same methodology fallback happens again → retire_if_same_verdict=true fires and Phase 4 is retired permanently.

### Phase 2a: Ensemble Chain Unblock — Tier 0r → Ensemble v7 → Adaptive Conformal (3 tasks)
**exp2520** implements Tier0rVerifier class (the code that exp2504 validated but never wrote). **exp2521** integrates it as the 10th verifier in group-conditional calibration. **exp2524** adds prompt-adaptive calibration (arXiv:2604.13991) and ACSE semantic entropy (arXiv:2605.04295) to the v7 ensemble.

### Phase 2b: KAN Restore + HalluGuard Corpus (2 tasks)
**exp2523** searches for the KAN model and retrains from scratch if not found. **exp2522** constructs an evaluation corpus from available results files and prototypes HalluGuard NTK-based Tier 0s.

### Phase 2c: Continuous Self-Learning (1 task, MANDATORY per CLAUDE.md)
**exp2525** extends FR-11 Tier 3 JEPA to incorporate the Phase 4 step-level energy signal as a JEPA training feature. If exp2519 produces step_granularity_achieved=True, the JEPA gain signal is augmented; otherwise the task improves JEPA with other available signals.

### Phase 3: Hardware — KV260 SD Card Flash (1 task)
**exp2526** attempts physical SD card preparation using the generated .hwh file from exp2514. Documents exact operator commands if physical flash cannot be automated.

### Phase 4: arXiv Submission Package Prep (1 task)
**exp2527** runs LaTeX compile check, prepares the submission package (abstract, author list, CCS concepts), incorporates exp2519 Phase 4 result into paper §4 (either clean validation or documented negative), and produces a pre-submission checklist. This runs regardless of Phase 4 outcome — the paper is ready to submit either way.

### Phase 5: Synthesis (2 tasks)
**exp2528** (claude+opus, NO HARD GATE): synthesizes all exp2518-exp2527 results, delivers final arxiv_ready determination, surfaces operator decision needed if Gate 3 methodology issue persists. **exp2529** (codex): operational retrospective.

---

## Dependency Graph

```
exp2518 (archive)
    |
    ├── exp2519 (Phase 4 v3)
    |       |
    |       └──> exp2527 (arXiv prep, reads Phase 4 result)
    |
    ├── exp2520 (Tier 0r implementation)
    |       |
    |       └── exp2521 (Ensemble v7, gated: tier0r_implemented==true)
    |                   |
    |                   └── exp2524 (Adaptive Conformal v2, gated: ensemble_v7_auroc>=0.970)
    |
    ├── exp2522 (HalluGuard + eval corpus)  [independent]
    ├── exp2523 (KAN restore)               [independent]
    ├── exp2525 (FR-11 Tier 3 JEPA)        [independent, reads exp2519 if available]
    ├── exp2526 (KV260 SD card)             [independent]
    |
    └──> exp2528 (capstone, reads exp2518-2527)
              |
              └── exp2529 (retro)
```

---

## Hardware Requirements

| Board | Terminal State | Status entering .243 | .243 task |
|---|---|---|---|
| AMD/Xilinx KV260 | Board-level latency transcript + kv260_synthesis_succeeded=true | kv260_hwh_generated=True, flash pending | exp2526: SD card flash attempt |
| Microchip PolarFire SoC | energy_sanity_check_passed=True | TERMINAL (exp2501) | graduated to optional |
| GateMate A1-EVB-2M | n=16 Ising tile flashed + smoked | TERMINAL (exp2453) | graduated to optional |

---

## Failed-Experiment Rerun Compliance Table

| .243 Task | Prior failed exp | Root cause | What is different | retire_if_same_verdict |
|---|---|---|---|---|
| exp2519 Phase 4 v3 | exp2508 (methodology fallback, step_granularity_achieved=False) | SemanticEnergy fallback used instead of IsingVerifier.energy() | PRECONDITION check for IsingVerifier; NO fallback path; blocked_* if not available | true (if step_granularity_achieved=False AGAIN) |
| exp2519 Phase 4 v3 | exp2486 (SemanticEnergy embeddings, pearsonr=0.108) | Wrong energy proxy (embedding distance not raw logprob) | IsingVerifier.energy() on step text, not SemanticEnergy embeddings | false (exp2508's pearsonr=-0.4266 did not trigger this retire) |
| exp2520 Tier 0r impl | exp2510 (blocked_tier0r_not_implemented) | Tier0rVerifier class not written in prior exp2504 | This task WRITES the implementation; not a re-test of viability | false |
| exp2521 Ensemble v7 | exp2510 (blocked_tier0r_not_implemented) | Same root cause as exp2520 | Assumes exp2520 success; gated on tier0r_implemented==true | false |
| exp2522 HalluGuard | exp2509 (blocked_no_eval_corpus) | No labeled evaluation corpus available at runtime | This task constructs the corpus from available results files first | false |
| exp2523 KAN restore | exp2513 (blocked_kan_not_found) | KAN model file missing from disk | This task retrains from scratch if model not found | true (if kan_model_rebuilt=False AND training fails) |
| exp2524 Adaptive Conformal | exp2511 (blocked_ensemble_v7_not_available) | Ensemble v7 not built (blocked by Tier 0r missing) | Gated on exp2521.ensemble_v7_auroc>=0.970 | false |
| exp2526 KV260 SD card | exp2514 (hwh_generated, flash pending operator) | Physical SD card flash is a manual operator step | This task attempts automated SD card prep using PYNQ | false |
| exp2527 arXiv prep | exp2515 (paper_updated=True, arxiv_ready per literal) | Paper updated but Gate 3 methodology caveat unresolved | This task incorporates exp2519 result; either clean Gate 3 or documented negative | false |

---

## Decentralization Compliance (CLAUDE.md Rules 1-7)

- **Rule 1 (local-first):** All AUROC experiments use existing on-disk data + CPU-only computation. exp2519 reads from existing telemetry manifest. exp2520-2524 use Python + existing carnot package. No closed-weight dependency.
- **Rule 2 (closed frontier models optional):** No task requires a closed-weight model. GGUF models would only be used if exp2525 JEPA training explicitly needs inference, which it does not.
- **Rule 3 (distribution mirroring):** paper-v6 + arXiv submission prep (exp2527) will include IPFS CID documentation for any new artifacts per CLAUDE.md Rule 3.
- **Rule 4 (multiple integration surfaces):** No API surface changes in this milestone.
- **Rules 5-7 (hardware portability, data minimization, no vendor abstractions in core):** No violations. All new code (Tier0rVerifier, FR-11 JEPA update) follows the BaseVerifier protocol.

---

## Exclusion Manifest Cross-Check

The following task scopes were checked against `ops/exclusion_manifest.yaml`:
- Phase 4 ARM-EBM: exp2486/exp2474/exp2487/exp2496/exp2497/exp2508 are NOT on the exclusion manifest. Only exp2497 (Spilled Energy Tier 0q) is retired. Phase 4 ARM-EBM scope remains live.
- Tier 0r implementation: no retired exp_id for this scope.
- KAN multilevel: no retired exp_id; exp2476 and exp2513 were "blocked" artifacts, not retired.
- GRPO/VPRM and WOPR puzzle cartridge scope are excluded — none of these appear in .243 tasks.

Zero exclusion manifest violations found.

---

## Agent Routing

| Task | Agent | Justification |
|---|---|---|
| exp2518 archive | codex/gpt-5.5 | Mechanical, no judgment |
| exp2519 Phase 4 v3 | codex/gpt-5.5 | Formulaic correlation computation with strict PRECONDITIONS |
| exp2520 Tier 0r impl | codex/gpt-5.5 | Code generation following BaseVerifier pattern |
| exp2521 Ensemble v7 | codex/gpt-5.5 | Mechanical calibration re-run with new verifier |
| exp2522 HalluGuard | codex/gpt-5.5 | Corpus construction + NTK approximation (formulaic) |
| exp2523 KAN restore | codex/gpt-5.5 | Search + retrain follows established pattern |
| exp2524 Adaptive Conformal | codex/gpt-5.5 | Calibration with prompt-type classifier |
| exp2525 FR-11 Tier 3 | codex/gpt-5.5 | Extending JEPA with new signal (mechanical) |
| exp2526 KV260 SD card | codex/gpt-5.5 | Hardware documentation + PYNQ SD prep |
| exp2527 arXiv prep | codex/gpt-5.5 | LaTeX compile + checklist (mechanical) |
| exp2528 capstone | claude+opus | Multi-artifact synthesis + open-ended arXiv decision |
| exp2529 retro | codex/gpt-5.5 | Templated retro |

**Agent allocation:** 11 codex (91.7%), 1 claude+opus (8.3%). Compliant with CLAUDE.md Codex-Default discipline (1 of 12 = 8.3% claude, below 16.7% threshold).
