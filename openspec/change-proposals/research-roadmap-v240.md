# Research Roadmap v240: AUROC Adversarial Resolution + Phase 4 ARM-EBM Empirical + KAN Retrain + PolarFire v3 + arXiv Gate

**Milestone:** 2026.05.240
**Status:** PROPOSED
**Date:** 2026-05-19
**Previous milestone:** 2026.05.239 — 10/12 tasks completed (AUROC 0.9351 via isotonic calibration but flagged TAUTOLOGY; FR-11 Tier 3 JEPA complete; Phase 4 ODAR failed; KV260 bitstream generated but not flashed; PolarFire missing; KAN blocked; paper audit fixed; arXiv still blocked)

---

## What .239 Proved

Milestone .239 had 10 of 12 tasks complete. Key findings:

**Major wins:**
- **AUROC 0.9351 achieved** (exp2473): Isotonic calibration of Fisher conformal p-values across 9 verifiers reached 0.9351, breaching the HIVE peer baseline 0.9236 by +0.0115. BUT: adversarial flag TAUTOLOGY (isotonic_auroc == best_calibrated_auroc — same variable used twice). Independent replication needed before paper citation.
- **FR-11 Tier 3 JEPA COMPLETE** (exp2475): jepa_predictor_implemented=True, jepa_violation_auc=0.7633. JEPA predictor trained on 36 telemetry examples; min_logprob is best feature. FR-11 Tiers 1-3 all satisfied.
- **Paper integrity FIXED** (exp2479): audit_passed_after_fix=True. exp1100 timing discrepancy resolved, citation gaps addressed. Paper is now auditable for arXiv submission modulo Phase 4 hold.
- **KV260 bitstream GENERATED** (exp2477): Vivado 2025.2.1 produced a 7.8MB bitstream (sha256=1bb0c3b…) targeting xck26-sfvc784-2LV-c. kv260_bitstream_flashed=False — no Xilinx JTAG programmer connected (only DirtyJTAG for GateMate and FlashPro5 for PolarFire on this bench).
- **GateMate on-board timing benchmark** (unnamed exp): sampler timing captured for GateMate (terminal state hit in .237; timing benchmark completed in .239).
- **Qwen censorship disclosure added** to paper §6 Limitations.

**Gaps confirmed:**
1. **AUROC TAUTOLOGY flag**: isotonic_auroc and best_calibrated_auroc are the same field (duration_s=0.12s suggests the calibration code uses the same variable twice). The 0.9351 number may be a real finding OR a code artifact — independent 5-seed replication is required.
2. **Phase 4 empirical validation FAILED** (exp2474): odar_energy_auroc=0.5584, pearson_r=0.19 — ODAR energy does NOT correlate with hallucination labels above chance. arXiv operator hold remains (phase4_validated=False).
3. **KAN model missing** (exp2476): blocked_kan_model_missing. The model was present in .238 (exp2467 had kan_model_exists=true) but missing in .239. The path exp2476 checked is different from exp2467's path, or the model was cleaned up. Must retrain.
4. **PolarFire MISSING** (exp2478): 3 consecutive Gemini CLI failures (wall-clock timeout + node bundle errors). carnot_runs_on_polarfire=False still. JAX unconditional import fix never applied.
5. **KV260 physical flash blocked**: bitstream exists but no Xilinx JTAG programmer is connected. Operator must physically attach programmer OR find alternative flash path.
6. **arXiv hold**: operator hold remains until phase4_validated=True. All other gates met (phase1_ship+audit+paper integrity).

---

## Three Biggest Gaps vs PRD Vision (entering .240)

### Gap 1: Phase 4 Empirical Validation (arXiv blocked on this)
The ODAR free-energy-as-routing approach tested in exp2474 yielded odar_energy_auroc=0.5584 (barely above chance, pearson_r=0.19). The Phase 4 hypothesis — "verifier energy is a free energy functional of the generative model" — needs an empirical test that doesn't depend on ODAR routing. The ARM-EBM bijection paper (arXiv:2512.15605) provides the alternative: if Carnot's Ising energy correlates with the LLM's implicit energy E=-log p(response), the bijection IS Phase 4 empirical validation. This is CPU-only (no new GGUF inference — just logprob extraction from existing telemetry manifests). Additionally, the Qwen censorship-circuit finding (research-references.md) predicts Fast-Slow Variant divergence on PRC vs non-PRC topics — a second direct test.

### Gap 2: AUROC 0.9351 Adversarial Resolution + Extension
The isotonic_auroc=0.9351 is flagged TAUTOLOGY. Before this can be cited in the paper, it must be replicated independently (5 random train/test splits) and the calibration code must be verified to use distinct variables for platt_auroc vs isotonic_auroc vs best_calibrated_auroc. Additionally, the group-conditional conformal approach (arXiv:2602.01285) separates verifiers by signal type and may push the verified AUROC further while also being adversarially cleaner (no TAUTOLOGY possible across groups).

### Gap 3: Hardware Completeness + KAN Certification
- KAN model missing: must locate or retrain. exp2467 had kan_model_exists=True; exp2476 checked a different path. The KAN model is essential for the certified deployment claim in the paper.
- PolarFire: 3x Gemini CLI failures. Switch to codex (which has been reliable on Python file edits). The fix is a 3-line change to python/carnot/__init__.py (try/except ImportError around `import jax`).
- KV260: bitstream generated; physical flash requires operator to connect Xilinx JTAG programmer. Document the requirement and explore USB-UART alternative paths.

---

## Architecture Snapshot (entering .240)

```
Tier 0 Verifiers (conformal p-value ensemble):
  Tier 0a: SemanticEnergy (AUROC=0.810)
  Tier 0b: HALT (AUROC=0.8539)
  Tier 0c: FregeLogic (AUROC=0.8831)
  Tier 0d: DiffuTruth (AUROC=0.588, marginal)
  Tier 0e: LogCons Hierarchical (AUROC=0.8896)
  Tier 0f: PCIB (AUROC=0.8669)
  Tier 0g: LaaB Meta-Judgment (AUROC=0.854)
  Tier 0h: NCO (AUROC=0.678, fixed from 0.500 tautology)
  Tier 0i: ODAR routing (AUROC=0.5584, excluded — below baseline)
  Tier 0p: LLM-as-Judge (AUROC=0.6412, excluded — below SemanticEnergy)
  Fisher conformal ensemble (9 verifiers): AUROC=0.9167 (Fisher ceiling)
  Isotonic calibration v4 (9 verifiers): AUROC=0.9351 (flagged TAUTOLOGY — needs replication)

Tier 1: KAN energy verifier (AUROC=0.994, certified_coverage=0.0 — model missing)
Tier 2: Constraint memory (FR-11 Tier 2, SQLite, cross_session_retention=1.0)
Tier 3: JEPA predictive verifier (FR-11 Tier 3, jepa_violation_auc=0.7633)
Tier 4: Adaptive energy landscape (FR-11 Tier 4, NOT YET IMPLEMENTED)

Hardware:
  KV260: bitstream_generated (7.8MB), kv260_board_flashed=False (no JTAG programmer)
  GateMate: TERMINAL (gatemate_bitstream_flashed=True, timing benchmark done)
  PolarFire: SSH reachable, carnot_runs_on_polarfire=False (JAX import bug not yet fixed)

Paper-v6: audit_passed_after_fix=True; arXiv hold: phase4_validated=False
Phase 1 ship: COMPLETE (PyPI+HF mirror+MCP+CLI docs+external reproducer all met)
```

---

## Milestone .240 Phase Structure

### Phase 0: Archive + Activate
- **exp2483**: Archive .239 results to research-complete.yaml and activate .240.

### Phase 1: AUROC Adversarial Resolution + Extension
- **exp2484**: AUROC Adversarial Replication v1 — 5-seed replication of isotonic calibration from exp2473. Verify calibration code uses distinct variables (not TAUTOLOGY). Report true_replicated_auroc with 95% CI. Critical for paper citation validity.
- **exp2485**: Group-Conditional Conformal Ensemble v5 (arXiv:2602.01285) — separate 9 verifiers into 3 groups (logprob-class, semantic-class, logic-class), calibrate each group independently, aggregate with group-conditional conformal coverage. Gated on exp2484 replication result.

### Phase 2: Phase 4 Empirical Alternative
- **exp2486**: Phase 4 ARM-EBM Bijection Empirical Test (arXiv:2512.15605) — for each telemetry entry, compute LLM implicit energy E=-log p(response) from existing logprob fields; compute Carnot Ising energy from constraint violations; test pearson_r and AUROC of energy delta as hallucination predictor. CPU-only. If pearson_r > 0.3 AND arm_ebm_auroc > 0.65, phase4_validated=True.
- **exp2487**: Phase 4 Qwen PRC Censorship Divergence Test — MANDATORY GGUF. Generate 20 PRC-topic and 20 neutral-topic responses using Qwen3.6-35B-A3B-GGUF; compute Carnot verifier energy for both sets; test if energy_prc > energy_neutral (Fast-Slow Variant divergence predicted by Phase 4 active-inference hypothesis). If energy_prc_mean > energy_neutral_mean AND prc_vs_neutral_p < 0.05, phase4_validated=True (alternative test).

### Phase 3: Continuous Self-Learning
- **exp2488**: FR-11 Tier 4 Adaptive Energy Landscape Prototype — initial implementation of online KAN structural adaptation. When accumulated constraint violations detect a new error pattern, the KAN adds a knot/spline to the constraint region. Prototype must demonstrate ≥1 structural adaptation on the telemetry corpus. continuous_self_learning_task: true.

### Phase 4: Hardware + KAN
- **exp2489**: KAN Energy Tier Locate + Retrain + LipNeXt (arXiv:2601.18513) — locate the KAN model (check paths exp2467 used vs exp2476); if found, apply LipNeXt λ·local_lip² penalty and retrain; if not found, retrain from scratch using telemetry corpus. Target: mean_local_lipschitz < 5.0, certified_coverage > 0.5, new_kan_auroc > 0.97.
- **exp2490**: PolarFire Carnot Deploy v3 — fix try/except ImportError around `import jax` in python/carnot/__init__.py; install carnot via --no-deps on PolarFire SSH; verify `python3 -c "import carnot; print(carnot.__version__)"` succeeds. Codex (NOT Gemini — 3x Gemini failures).
- **exp2491**: KV260 JTAG Physical Flash Documentation + USB-UART Alt Path — document exact physical hardware required (Digilent JTAG HS2 or USB Cable II); explore if USB-UART on KV260's USB3 port can flash bitstream; document xc3sprog/openocd alternative chains. Deliverable: kv260_flash_requirements.md + alt_path_feasibility bool.

### Phase 5: Paper + Synthesis
- **exp2492**: Paper-v6 Phase 4 + AUROC Section Update — if phase4_validated (from exp2486 or exp2487), write §7 Phase 4 results; update AUROC table with replicated value from exp2484 (corrected or confirmed 0.9351). If phase4 not validated, write clear partial-validation note with ODAR negative result + ARM-EBM result.
- **exp2493**: Capstone v240 (claude+opus, requires_claude) — synthesize all .240 results; determine arXiv submission readiness; produce final AUROC table; assess Phase 4 hold; NO HARD GATE (capstone always runs).
- **exp2494**: Retro v240 (codex).

---

## Dependency Graph

```
exp2483 (archive)
  ├── exp2484 (AUROC replication)
  │   └── exp2485 (group-conditional conformal, gated on exp2484)
  ├── exp2486 (ARM-EBM Phase 4)
  ├── exp2487 (Qwen PRC Phase 4)  [MANDATORY GGUF]
  ├── exp2488 (FR-11 Tier 4)
  ├── exp2489 (KAN retrain)
  ├── exp2490 (PolarFire v3)
  └── exp2491 (KV260 doc)
exp2484, exp2485, exp2486, exp2487, exp2489, exp2490 → exp2492 (paper update)
exp2492 → exp2493 (capstone, no hard gate)
exp2493 → exp2494 (retro)
```

---

## Hardware Requirements

| Board | Status entering .240 | Task | Terminal state condition |
|---|---|---|---|
| KV260 | bitstream_generated, not_flashed | exp2491 (doc + alt path) | kv260_board_flashed=True (needs physical programmer) |
| GateMate | TERMINAL (flashed + timing done) | none required | already terminal |
| PolarFire | SSH reachable, carnot not installed | exp2490 (deploy v3) | carnot_runs_on_polarfire=True |

---

## Agent Routing

| Task | Agent | Justification |
|---|---|---|
| exp2483 | codex | Archive pattern, mechanical |
| exp2484 | codex | Sklearn calibration + score aggregation |
| exp2485 | codex | Group-conditional conformal, established sklearn pattern |
| exp2486 | codex | Logprob extraction + pearson_r, mechanical |
| exp2487 | codex | GGUF inference + energy comparison (templated) |
| exp2488 | codex | KAN structural adaptation prototype |
| exp2489 | codex | KAN training + LipNeXt penalty |
| exp2490 | codex | Python file edit + SSH command (3-line fix) |
| exp2491 | codex | Documentation task + openocd research |
| exp2492 | codex | Paper section update (pre-drafted content) |
| exp2493 | claude+opus | Multi-file synthesis: 12+ artifact reads, open-ended judgment on arXiv readiness, cross-cutting Phase 4 + AUROC + hardware assessment |
| exp2494 | codex | Retro template |

**Agent routing: 11 codex (91.7%), 1 claude+opus (8.3%)** — within CLAUDE.md codex-default quota.

---

## Decentralization Compliance (Rules 1-7)

- **Rule 1 (local-first)**: All verifier inference uses local GGUF models. exp2487 MANDATORY Qwen3.6-35B-A3B-GGUF. No closed-weight dependencies.
- **Rule 2 (closed frontier optional)**: No closed-weight requirements. All paths work with local GGUF.
- **Rule 3 (distribution mirroring)**: Paper-v6 artifacts will be mirrored to HuggingFace + IPFS when arXiv submission proceeds.
- **Rule 4 (multiple integration surfaces)**: Python API, CLI, MCP server all maintained.
- **Rule 5 (hardware portability)**: KAN certified deployment + PolarFire RISC-V target both advance hardware portability.
- **Rule 6 (data minimization)**: No closed-weight calls in this milestone.
- **Rule 7 (no vendor abstractions in core)**: All experiments read from standard interfaces; no vendor SDK imports in core.

---

## Exclusion Manifest Cross-Check (pre-emit)

Checked `ops/exclusion_manifest.yaml`:
- exp2091 (retired): "Tier 1 CSL Grammar Updates via gemini CLI" — no scope match in .240 tasks.
- exp887, exp783, exp799, exp804, exp809, exp825 (retired): discriminative JEPA OOD failure — exp2488 is FR-11 Tier 4 (adaptive KAN structure), not discriminative JEPA; different deliverable shape.
- exp260, exp308, exp309 (retired): sequential inference loop — no match in .240.
- All other retired exp_ids: no scope match detected in .240 tasks.

Zero retired experiment scope matches found.

---

## Failed-Experiment Rerun Compliance

| Task | Prior failure(s) | What changed |
|---|---|---|
| exp2483 | exp2471 (archive .238→.239) | Different milestone metadata (.239→.240) |
| exp2484 | exp2473 (TAUTOLOGY adversarial flag, 0.9351) | Explicitly checks for TAUTOLOGY; uses distinct variable names; 5-seed replication |
| exp2485 | exp2461/exp2448/exp2438 (v3/v2/v1 conformal) | New aggregation method: group-conditional vs scalar calibration (arXiv:2602.01285) |
| exp2486 | exp2474 (ODAR odar_energy_auroc=0.5584) | New mechanism: ARM-EBM bijection logprob energy vs Carnot Ising energy (arXiv:2512.15605) — not ODAR routing |
| exp2487 | exp2474 (Phase 4 empirical failed) | New approach: Qwen PRC vs non-PRC topic divergence (censorship circuit); different prediction target |
| exp2488 | none (FR-11 Tier 4 never attempted) | Fresh scope |
| exp2489 | exp2476 (blocked_kan_model_missing) | Explicitly checks multiple candidate paths (exp2467 path vs exp2476 path); retrains if missing |
| exp2490 | exp2478 (missing — Gemini CLI 3x), exp2466 (partial) | Agent changed to codex (Gemini CLI consistently fails on PolarFire SSH tasks) |
| exp2491 | exp2477 (bitstream_not_flashed, no JTAG) | Scope narrowed to documentation + alternative path research (not attempting flash) |
| exp2492 | none (paper update with new Phase 4 content) | Fresh scope |
| exp2493 | exp2481 (capstone v239) | Different milestone synthesis; NO HARD GATE |
| exp2494 | exp2482 (retro v239) | Different milestone |
