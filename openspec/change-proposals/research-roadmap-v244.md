# Research Roadmap v244: IsingVerifier Fix + Phase 4 v4 + Ensemble v7b + arXiv LaTeX Fix + Continuous Self-Learning JEPA Integration

**Milestone:** 2026.05.244
**Status:** PROPOSED
**Date:** 2026-05-19
**Previous milestone:** 2026.05.243 — 7/12 capstone-input experiments at terminal verdict;
phase4_final_status=blocked_precondition (IsingVerifier is a stub); ensemble_v7_auroc=0.9607
(regression from 0.9750); latex_compile_success=False; JEPA improved 0.7633→0.8889;
KAN rebuilt with persisted checkpoint; Tier 0r implemented (AUROC=0.8256 synthetic);
arxiv_ready=False.

---

## What .243 Proved

Milestone .243 completed 7 of 12 tasks at terminal verdict (5 either gate-blocked or not run).

**Major wins:**
- **Tier 0r Curry-Howard verifier implemented** (exp2520): `python/carnot/verify/tier0r_curry_howard.py`
  written following the BaseVerifier interface; solo AUROC=0.8256 on synthetic 50-example corpus.
  The three-task dependency chain (exp2510 blocked since .241) is now unblocked at the root.
- **FR-11 Tier 3 JEPA strengthened** (exp2525): jepa_violation_auc advanced from 0.7633 to
  0.8889 (+0.126) by adding ising_energy_response_level + logprob_variance features. The
  continuous self-learning loop compounds with each milestone.
- **KAN Tier 1 rebuilt from scratch** (exp2523): kan_model_rebuilt=True, checkpoint saved to
  `models/kan_tier1_restored.safetensors`, multilevel_auroc=1.0 on available training corpus.
  The blocked_kan_not_found gap from .242/.243 is resolved.

**Gaps confirmed (entering .244):**
1. **IsingVerifier is a stub** — `class IsingVerifier: pass` (no methods, no arguments). Every
   Phase 4 ARM-EBM attempt that requires IsingVerifier.energy() is blocked until this stub is
   implemented. This is the root cause of three consecutive blocked_precondition verdicts.
2. **Ensemble v7 regression to 0.9607** — Adding Tier 0r to Group C dragged the Fisher
   combination down by -0.0143 (0.9750 → 0.9607). Root cause: Tier 0r's nonconformity scores
   are in a different range from Group C members (Tier 0d, Tier 0h), causing calibration mismatch.
   Fix: assign Tier 0r to a dedicated Group D (proof-path class).
3. **arXiv LaTeX compile failure** — `latex_compile_success=False` in exp2527. The paper source
   is at `docs/arxiv-paper/main.tex`. The abstract is 522 words (exceeds 250-word arXiv limit).
   Compile errors and abstract length must both be fixed before submission.

---

## Three Biggest Gaps vs PRD Vision (entering .244)

### Gap 1: IsingVerifier Stub Must Be Implemented

`python/carnot/verify/semantic_energy.py` defines `class IsingVerifier: pass` — completely empty.
This class was intended to compute constraint-violation energy from text, but was never implemented.
Every Phase 4 ARM-EBM attempt (exp2486, exp2508, exp2519) that checked
`IsingVerifier(n_spins=4).energy("text")` failed because the class accepts no arguments and has no
`.energy()` method. The root fix: implement IsingVerifier with a real `energy(step_text: str) -> float`
method that extracts arithmetic/logical claims from text, checks them for consistency, and returns a
normalized violation energy in [0, 1].

**exp2531 must write the real implementation.** Once IsingVerifier.energy(text) is functional, exp2532
(Phase 4 ARM-EBM v4) can run the structural Phase 4 test for the first time. retire_if_same_verdict=true
on exp2532 vs exp2519 — if Phase 4 still fails after IsingVerifier is implemented, Phase 4 is
permanently retired and paper §4 documents the honest negative outcome.

### Gap 2: Ensemble v7 Regression — Tier 0r Needs Dedicated Group D

The ensemble v7 run (exp2521) used Group C (logic-class) for Tier 0r, alongside Tier 0d
(DiffuTruth, AUROC=0.588) and Tier 0h (NCO, AUROC=0.678). The Fisher group-conditional
combination averages calibration offsets within each group; Tier 0r's proof-path scores
(range ~0.0-0.6) are systematically lower than Tier 0d/0h scores, shifting the group mean
and reducing Fisher discriminability. The fix: assign Tier 0r to a new Group D (proof-path
class) with its own independent calibration. Expected result: Group D calibration sees only
Tier 0r's clean score distribution; Group C recovers its prior mean_cal_C ≈ 0.47-0.53;
ensemble v7b AUROC should restore to ≥0.975.

**exp2533 (Ensemble v7b)** implements Tier 0r Group D. If AUROC ≥ 0.975, exp2534
(Adaptive Conformal v2 + ACSE, blocked since .242) is unblocked.

### Gap 3: arXiv LaTeX Compile Failure

The paper at `docs/arxiv-paper/main.tex` does not compile (`latex_compile_success=False` in exp2527).
Two confirmed issues: (a) the abstract is 522 words (arXiv limit is 250), and (b) the pdflatex
pipeline itself may have missing package dependencies or syntax errors. Both must be resolved before
the submission package is ready. The submission checklist from exp2527 shows all 4 gates are met
(gate_1 through gate_4=True) and Phase 4 will be documented as either positive (if exp2532 validates)
or negative (honest negative per CLAUDE.md). The LaTeX fix unblocks the final arXiv submission package.

**exp2536 (LaTeX compile fix)** diagnoses and fixes the compile errors and abstract length.

---

## Architecture Snapshot (entering .244)

```
Tier 0 Verifiers (conformal p-value ensemble — current state):
  Group A (logprob-class):
    Tier 0a: SemanticEnergy (AUROC=0.810)
    Tier 0b: HALT (AUROC=0.8539)
    Tier 0f: PCIB (AUROC=0.8669)
  Group B (semantic-class):
    Tier 0c: FregeLogic (AUROC=0.8831)
    Tier 0e: LogCons Hierarchical (AUROC=0.8896)
    Tier 0g: LaaB Meta-Judgment (AUROC=0.854)
  Group C (logic-class):
    Tier 0d: DiffuTruth (AUROC=0.588)
    Tier 0h: NCO (AUROC=0.678)
  Group D (proof-path class — BEING ADDED in .244):
    Tier 0r: Curry-Howard (implemented exp2520, solo AUROC=0.8256 synthetic)
  Ensemble status:
    v6: AUROC=0.9750 (adversarially verified, cite-safe, .241/.242 carry-forward)
    v7: AUROC=0.9607 (REGRESSION — Tier 0r in Group C dragging Group C down)
    v7b (target .244): AUROC>=0.975 (Tier 0r → Group D, independent calibration)
  HIVE peer baseline: 0.9236 (gap +0.0514 if v6 baseline; gap +0.0371 vs v7)

Core EBM infrastructure:
  IsingVerifier: STUB (class IsingVerifier: pass) — must implement energy(text) in exp2531
  KAN Tier 1: REBUILT checkpoint at models/kan_tier1_restored.safetensors (exp2523)
    - multilevel_auroc=1.0 on training corpus (not adversarially verified on original eval yet)
  FR-11 JEPA Tier 3: jepa_violation_auc=0.8889 (exp2525 — response-level energy features)
    - Pipeline integration pending (exp2539)

Hardware (terminal states):
  GateMate A1-EVB-2M:
    - DirtyJTAG onboard (1209:c0ca) verified, openFPGALoader JTAG detect works (0x20000001)
    - yosys↔nextpnr LUT mapping mismatch: CC_LUT3/CC_LUT2 vs CC_LUT4 BLOCKING bitstream
    - Terminal state: n=16 Ising tile flashed + gatemate_bitstream_flashed=True
  KV260:
    - .hwh generated (exp2514, vivado v2025.2.1)
    - PYNQ SD card prep documented (exp2526)
    - Physical flash pending operator action
    - Terminal state: board-level latency transcript + kv260_synthesis_succeeded=True
  PolarFire SoC: TERMINAL (exp2501 — energy_sanity_check_passed=True)

arXiv status:
  gate_1 (Phase 1 ship): True
  gate_2 (paper integrity audit): True
  gate_3 (Phase 4 validated): blocked_precondition (IsingVerifier stub)
  gate_4 (AUROC adversarially verified): True (0.9750 group-conditional)
  latex_compile_success: False (docs/arxiv-paper/main.tex — abstract 522 words)
  submission_package_ready: False
```

---

## Five Phases of Milestone .244

### Phase 0: Archive and Activate
**exp2530** — Archive .243 and activate .244. Standard milestone transition.

### Phase 1: IsingVerifier Implementation + Phase 4 ARM-EBM v4
**exp2531** — Implement IsingVerifier.energy(step_text: str) -> float in
`python/carnot/verify/semantic_energy.py`. The class is currently a stub. The implementation:
extract arithmetic/numerical claims from text using regex/NLP heuristics, verify each claim
for consistency, return normalized energy in [0, 1]. Threshold: at least a working energy()
method that returns non-trivial values (not always 0.0 or 1.0) on test text.

**exp2532** — Phase 4 ARM-EBM v4 NO FALLBACK, gated on exp2531. Uses IsingVerifier.energy(step_text)
directly (no SemanticEnergy fallback). retire_if_same_verdict=true vs exp2519 — this is the final
structured attempt at Phase 4 with a real IsingVerifier. If energy_proxy_used≠'ising_verifier_direct'
OR step_granularity_achieved=False again, Phase 4 is permanently retired and paper §4 is revised.

### Phase 2: Ensemble v7b Regression Fix + Adaptive Conformal
**exp2533** — Ensemble v7b: assign Tier 0r to Group D (proof-path class), run group-conditional
calibration with 10 verifiers across 4 groups (A: logprob, B: semantic, C: logic, D: proof-path),
5 seeds. Gate: ensemble_v7b_auroc >= 0.975 to restore cite-safe headline.

**exp2534** — Adaptive Conformal v2 + ACSE (arXiv:2604.13991 + arXiv:2605.04295), gated on
exp2533.ensemble_v7b_auroc >= 0.975. Prompt-adaptive calibration on top of ensemble v7b.
This experiment has been blocked since .242 by ensemble regression; .244 finally unblocks it.

### Phase 3: New Verifier + arXiv LaTeX Fix
**exp2535** — Tier 0u Logical Consistency verifier prototype (arXiv:2605.03971, May 2026).
The paper models self-consistency constraints between a response and the LLM's own self-check
verification as a label constraint graph. Carnot's implementation: generate a brief self-check
prompt for each candidate response, compare self-check verdict with response claims, compute
consistency score. Gate: tier0u_auroc > 0.70.

**exp2536** — LaTeX compile fix + arXiv submission package v2. Diagnose pdflatex failures on
`docs/arxiv-paper/main.tex`, fix abstract (522→≤250 words), fix LaTeX errors. Incorporate
Phase 4 final outcome from exp2532 into §4 (positive or negative per operator rule). Gate:
latex_compile_success==True AND submission_package_ready==True.

### Phase 4: Hardware Continuity
**exp2537** — GateMate A1 LUT mapping workaround. The yosys 0.64 ↔ nextpnr-himbaechel 0.10
mismatch (CC_LUT3/CC_LUT2/CC_LUT1 emitted vs CC_LUT4 accepted) is the last GateMate blocker.
Try `synth_gatemate -abc9` which enables ABC9 optimization and typically reduces to CC_LUT4.
Alternatively: post-process the netlist JSON to upgrade lower-order LUTs. Gate: gatemate_lut_mapping_resolved==True OR gatemate_blocker_documented==True.

**exp2538** — KV260 SD card physical flash attempt. SD card prep was fully documented in exp2526.
Attempt automated flash: download PYNQ SD image, flash if /dev/sd* is present, document results.
If no SD card device: write operator action checklist with exact commands. Gate: kv260_flash_attempted OR kv260_flash_documentation_complete.

### Phase 5: Continuous Self-Learning + Paper + Synthesis
**exp2539** — FR-11 Tier 3 JEPA → VerifyRepairPipeline integration (continuous_self_learning_task:true).
JEPA violation AUC=0.8889 is now strong enough to deploy as a "predict first" fast-path: if JEPA
predicts low violation probability, skip full Ising verification. Gate: jepa_pipeline_integrated==True.

**exp2540** — Paper-v6 Phase 4 final outcome update + missing citations. Incorporates exp2532 outcome
into §4, adds 4/δ Bound citation (arXiv:2512.02080), Fast-Slow Training citation (arXiv:2605.12484),
and new papers from this sweep.

**exp2541** — Capstone v244 (claude+opus, requires_claude, NO HARD GATE). Multi-artifact synthesis
across 11 deliverables (exp2530-exp2540). Produces operator_recommendation for arXiv submission.

**exp2542** — Retro v244 (codex).

---

## Dependency Graph

```
exp2530 (archive) → always first
exp2531 (IsingVerifier impl) → ungated
exp2532 (Phase 4 v4) → gated on exp2531.ising_verifier_text_energy_implemented==true
exp2533 (ensemble v7b) → ungated (fixes Group C regression independently)
exp2534 (adaptive conformal) → gated on exp2533.ensemble_v7b_auroc>=0.975
exp2535 (Tier 0u verifier) → ungated
exp2536 (LaTeX fix) → ungated (reads exp2532 result if available)
exp2537 (GateMate LUT) → ungated (hardware, independent)
exp2538 (KV260 flash) → ungated (hardware, independent)
exp2539 (JEPA pipeline) → ungated (builds on exp2525 baseline)
exp2540 (paper update) → reads exp2532 outcome for §4
exp2541 (capstone) → reads exp2530-exp2540
exp2542 (retro) → reads exp2541
```

Critical path: exp2531 → exp2532 (Phase 4 terminal determination) → exp2536 (LaTeX + §4) → exp2541 (capstone submission recommendation)

---

## Hardware Requirements

| Board | Current State | .244 Target | Blocker |
|---|---|---|---|
| GateMate A1-EVB-2M | DirtyJTAG verified, netlist synthesized, LUT mismatch blocking P&R | LUT mapping resolved → end-to-end bitstream | synth_gatemate -abc9 or techmap workaround |
| KV260 | .hwh generated, SD card prep documented | Physical SD card flash (automated or manual) | SD card device accessible? |
| PolarFire SoC | TERMINAL | No new tasks (graduated) | — |

---

## Decentralization Compliance

- Rule 1 (local-first open models): IsingVerifier.energy() works offline without any LLM call
- Rule 2 (closed models optional): exp2535 Tier 0u self-check may use local model; closed = opt-in
- Rule 3 (IPFS mirroring): no new weights being published in this milestone
- Rule 4 (multiple surfaces): IsingVerifier available via Python API, CLI, and MCP
- Rule 5 (hardware portability): GateMate + KV260 hardware tracks maintained
- Rule 6 (data minimization): all new verifiers compute locally
- Rule 7 (no vendor abstractions in core): IsingVerifier implementation stays in `python/carnot/verify/`

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml` for retired patterns. No experiment in this milestone
matches any retired scope:
- GRPO/VPRM lineage: no GRPO tasks
- WOPR puzzle cartridges: no puzzle tasks
- HardNet++/DSP repair stack: no repair stack tasks
- HalluSAEGeometricProbe: exp2535 uses a different approach (logical consistency, not SAE geometry)
- THRML scaling sweep: no THRML tasks
- SpecAnn Phase 3 sampler: no SpecAnn tasks
- iCE40 PIMI: GateMate tasks use openFPGALoader path, not PIMI sparse adjacency

---

## Failed-Experiment Rerun Compliance

| Task | Prior failure | Addressed by |
|---|---|---|
| exp2532 Phase 4 v4 | exp2519 (blocked_precondition: IsingVerifier stub) | exp2531 implements IsingVerifier.energy(text). retire_if_same_verdict=true |
| exp2533 ensemble v7b | exp2521 (regression 0.9607: Group C dragged down) | Tier 0r moved to new Group D with independent calibration |
| exp2534 adaptive conformal | exp2524 (not run: exp2521 failed gate) | exp2533 fixes ensemble gate; adaptive conformal finally unblocked |
| exp2536 LaTeX fix | exp2527 (latex_compile_success=False) | Diagnose and fix pdflatex errors + abstract word count |
| exp2538 KV260 flash | exp2526 (documentation done, flash not attempted) | Attempts automated flash; operator checklist if SD not accessible |
| exp2539 JEPA pipeline | exp2525 (JEPA trained, not integrated) | Different scope: pipeline integration vs training |

---

## Agent Routing

| Task | Agent | Justification |
|---|---|---|
| exp2530 archive | codex | Mechanical YAML/text operation |
| exp2531 IsingVerifier impl | codex | Single-file code implementation following existing patterns |
| exp2532 Phase 4 v4 | codex | Single-script analysis with structured gates |
| exp2533 ensemble v7b | codex | Calibration rerun with modified group assignment |
| exp2534 adaptive conformal | codex | Calibration computation following known algorithm |
| exp2535 Tier 0u verifier | codex | Single-file verifier implementation following BaseVerifier pattern |
| exp2536 LaTeX fix | codex | File edit + pdflatex diagnosis |
| exp2537 GateMate LUT | codex | Toolchain command execution with known fix options |
| exp2538 KV260 flash | codex | Hardware prep script execution |
| exp2539 JEPA pipeline | codex | Single-file integration following VerifyRepairPipeline pattern |
| exp2540 paper update | codex | Text editing + citation addition |
| exp2541 capstone | claude+opus | Multi-artifact synthesis (11 files), cross-phase judgment under ambiguity, arXiv operator recommendation. Meets all 3 positive-criterion conditions per CLAUDE.md |
| exp2542 retro | codex | Templated summary from capstone |

Agent distribution: 12 codex/gpt-5.5 (92.3%), 1 claude+opus (7.7%)
