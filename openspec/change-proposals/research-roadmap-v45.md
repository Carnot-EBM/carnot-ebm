# Research Roadmap v45 — Milestone 2026.04.45

**Status:** Proposed
**Milestone:** 2026.04.45
**Title:** Live-Calibrated CoACE and DSVD — Closing the Offline/Live Distribution Gap
**Planned Experiments:** 589–600 (12 experiments)
**Planned Date:** 2026-04-20 onwards

---

## What Milestone 2026.04.44 Proved

Milestone .44 delivered two breakthroughs and surfaced a critical new failure mode:

1. **JEPA v11 AUC=1.0 on contrastive pairs (RETRO-063 nominally resolved, Exp 580).**
   Explicit CPMI contrastive pair construction (correct chain vs incorrect chain for the same
   question) fixed the three-consecutive-retrain anti-correlation problem. HOWEVER: Exp 577
   produced only 9 synthetic pairs (real pair count insufficient), so AUC=1.0 is almost
   certainly overfitting. The CPMI architecture is sound; the corpus is too small for a
   reliable estimate. Milestone .45 must retrain on Exp 578's 100 live pairs.

2. **DSVD mid-generation detection viable at AUC=0.976 (Exp 587).**
   DSVDAdapter (arXiv 2503.03149) implementing a lightweight hidden-state verification head
   on Qwen3.5-0.8B achieved AUC=0.976 vs CoACE v1 AUC=0.824 on the same corpus.
   This means mid-generation detection outperforms post-hoc arithmetic extraction by 0.15+
   AUC points. Mid-generation detection + rollback is a fundamentally different repair
   architecture that may succeed where post-hoc extraction has failed.

3. **CoACE offline/live distribution gap discovered (RETRO-066, Exp 581).**
   Exp 576 showed CoACEExtractorV2 achieving 86.7% recall on Exp 565's 25 test responses.
   Exp 581 ran the SAME extractor on 25 live production responses from the FOVER corpus.
   Result: v2_recall=5.9% (unchanged from v1). The offline test corpus (Exp 565) was drawn
   from simple GSM8K responses with clean `A op B = C` patterns — not representative of
   actual IT-model arithmetic prose. The 86.7% was a measurement artifact.
   Root cause: IT models write narrative mathematics ("Adding 47 to 28 gives us 76") and
   multi-step derivations ("so the running total is now 150"), not simple equations. CoACE
   v2's prose patterns weren't calibrated against live model output distributions.

4. **ExclusionManifest built but conductor not wired (RETRO-067, Exp 575).**
   The manifest JSON and check script were created; conductor_consulted=False. The same 5
   experiments (308, 260, 309, 425, 410) appeared in the slowest-5 for the eighth
   consecutive milestone. Cumulative waste: ~2,870 minutes (47.8 hours) since .37.

5. **Live 50q A collected (Exp 578), corpus expanded (Exp 579).**
   100 live pairs (GSM8K 0-49) collected with inference_mode='live_gpu'. FOVER corpus now
   has real Qwen3.5-0.8B and Gemma4-E4B-it responses to calibrate CoACE v3 against.

6. **FPGA still blocked on Vivado (Exps 584, 585).**
   KV260 board is present and physically ready. Vivado not installed on the host machine.
   The bitfile synthesis is a manual human action: install Vivado 2023.2, run the TCL script.

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: CoACE Offline/Live Distribution Gap — Extraction Calibration Needed (RETRO-066, FR-12)

**Root cause confirmed (Exp 581):** CoACEExtractorV2 trained on synthetic offline corpus achieves
86.7% recall on curated test responses but only 5.9% recall on live FOVER production responses.
The 16 false negatives from Exp 581 are live IT-model outputs with arithmetic errors expressed as:

- Natural language arithmetic: "Adding 47 to 28 gives us 76 dollars" — no `=` sign
- Cumulative variable chains: "The subtotal is 150... therefore the final amount (150 + 25 = 176)"
  where the error is in the step between `150 + 25` and `176` (correct: 175)
- Percentage word problems: "20 percent of 300 people, or 60 people, attended"
  (pattern: `N percent of M... or P people` — P should equal N/100*M)
- Division in disguise: "split equally among 4 groups of 12 gives 48 total" — 4*12=48 but
  the stated claim may differ
- Unit conversion errors: "converting 2.5 hours to minutes gives 145 minutes" (should be 150)

**The fix:** Calibrate CoACEExtractorV3 on Exp 578's 100 LIVE pairs. For each incorrect live
response (where is_correct=False), manually extract the arithmetic violation type, then
build pattern matchers for the actual error distributions in real IT-model outputs.

This is not a regex change — it requires analyzing the actual error patterns in the 100 live
pairs from Exp 578 before writing any new extractor code.

**Why PRD cares:** FR-12 (Verifiable Reasoning) — at 5.9% live recall, Carnot identifies
5.9% of real violations. This is not a product. With DSVD at 0.976 AUC, the detection gap
exists at the extraction layer, not the energy layer.

### Gap 2: JEPA v11 on 9 Synthetic Pairs — Live Validation Required (RETRO-063 partial)

**Root cause (Exp 580):** JEPACPMIPairBuilder (Exp 577) produced `honest_verdict=pairs_built_insufficient`
— only 9 synthetic pairs available when the CPMI retrain ran. AUC=1.0 on 9 pairs is
statistically meaningless (a 2-parameter model can memorize 9 examples). The CPMI
architecture is correct, but the validation is unreliable.

**The fix:** Retrain JEPA v12 on the 100 live pairs from Exp 578. Exp 578 collected
`n_pairs_collected >= 40` pairs per batch (actual: ~100 pairs total). For each question,
Qwen3.5-0.8B and Gemma4-E4B-it both generated responses — this gives cross-model
contrastive pairs where the same question has at least one correct and one incorrect
answer. CPMI pair builder should now have sufficient real data.

**Why PRD cares:** FR-11 (Autonomous Self-Learning Loop) — JEPA Tier 3 predictive
verification is the core of the self-learning architecture. A predictor validated on 9
synthetic pairs cannot be wired into the production cascade.

### Gap 3: RETRO-033 #13 — Live Verify-Repair Still Undemonstrated (13 Attempts)

**Current state:** RETRO-033 has never been resolved in 12 consecutive attempts. The root
cause has always been insufficient extraction recall — the pipeline cannot improve what it
cannot detect. Exp 582 was blocked by Exp 581's gate (v2_recall=5.9%).

**The fix requires:**
1. CoACE v3 with live_recall >= 30% (from Gap 1 fix)
2. OR DSVD-guided detection + repair (DSVD AUC=0.976, already validated)

DSVD is now the primary path to RETRO-033 because it does not depend on arithmetic
pattern extraction — it detects violations from hidden-state geometry. If DSVD detects
a violation, the repair can target the specific generation boundary where the hidden
state diverged, rather than trying to repair extracted equations post-hoc.

---

## Architecture Diagram (Verification Cascade — .45 State)

```
LLM Response (IT model: Gemma4-E4B-it, Qwen3.5-0.8B)
    │
    ▼
[Tier 0a] CarnotThinkProbe (3-step CoT verdict, ThinkPRM arXiv 2504.16828)
    │  verdict='incorrect' → fast-path violation
    ▼
[Tier 0b] SpilledEnergyDetector (per-token logit discrepancy, arXiv 2602.18671)
    │  high_spill_fraction → pass through
    ▼
[Tier 0c] NUP Probe v4 (contrastive bigram energy, AUC=1.0, Exp 523)
    │  score <= threshold → skip downstream
    ▼
[Tier 0d] HallucinationBasinDetector (latent basin depth, arXiv 2604.04743)
    │  basin_risk <= threshold → skip downstream
    ▼
[Tier 0e] HalluField (thermodynamic energy-path, arXiv 2509.10753) [added .43]
    │  advisory: is_unstable recorded in certificate
    ▼
[Tier 1]  SinkProbe (attention sink concentration, arXiv 2604.10697)
    │  mean_sink >= threshold → skip downstream
    ▼
[Tier 2]  EORM (energy reward model, 55M params, GRPO-retrained on real data)
    │  energy < threshold → verified
    ▼
[Tier 2.5] DSVDAdapter [NEW .45 WIRE-IN if live AUC >= 0.80]
    │  Hidden-state verification head; detects violation boundary during/after generation
    │  dsvd_auc=0.976 (Exp 587 synthetic); needs live validation (Exp 592)
    │  If violation_detected: targeted rollback to violation boundary
    ▼
[Tier 3]  CoACEExtractorV3 [NEW .45, live-calibrated on Exp 578 pairs]
    │      + VerifyRepairPipeline
    │      parse arithmetic graph (live-calibrated patterns) → execute → compare
    ▼
    Verified / Violated + Repair Suggestion

[Tier 3 fast-path] JEPA v12 [NEW .45, trained on 100+ live pairs]
    Partial CoT (first 50%) → JEPA v12 → energy_score
    → high energy: run full Tier 3 verification
    → low energy: skip Tier 3 (fast-path approval)

[Symbolic-KAN Energy Tier] [added .44, arXiv 2603.23854]
    KAN energy function with interpretable symbolic rules
```

---

## Phase Descriptions

### Phase 0: Process Infrastructure (Exps 589-590, CPU-only, MUST BE FIRST)

**Exp 589 (CPU-only):** ExclusionManifest Conductor Wire-In + NPU Unblock v7 (RETRO-067).

Exp 575 built `scripts/conductor_exclusion_manifest.json` listing experiments 308, 260, 309,
425, 410 as excluded. `conductor_consulted=False` because research_conductor.py was not modified.
This experiment creates a `scripts/conductor_session_wrapper.py` that:
1. Calls `check_exclusion_manifest.py` for each pending experiment before the conductor spawns it
2. Logs which experiments were skipped and why
3. Provides clear human instructions for using it

NPU unblock: attempt `pip install mlir-aie` (IRON alternative path per Exp 435) and
`sudo pacman -S ninja openblas` detection. Even if still blocked, document the exact
current state clearly so a human can unblock in one command.

**Exp 590 (CPU-only):** CARNOT_FORCE_LIVE Import-Time Assertion Module.

Create `python/carnot/pipeline/live_assertion.py` with `assert_live_gpu_available()`:
- If `torch.cuda.is_available()` returns True AND `CARNOT_FORCE_LIVE != '1'` → raise
  `RuntimeError("CARNOT_FORCE_LIVE must be '1'. Run: source scripts/session_startup.sh")`
- This makes RETRO-062 impossible to recur: any experiment using this assertion will
  immediately fail with a clear error message rather than silently falling back to synthetic.
- Wire into all future live GPU experiment templates.

### Phase 1: Live-Calibrated Extraction (Exps 591-592, CPU-only)

**Exp 591 (CPU-only):** CoACEExtractorV3 — Live-Corpus Calibration (RETRO-066 CRITICAL PATH).

The ONLY way to fix RETRO-066 is to analyze what the live corpus actually contains.
Algorithm:
1. Load Exp 578's live pairs (`results/live_pairs_578.json`).
2. For each incorrect response (is_correct=False): manually analyze what arithmetic
   operations appear that CoACE v2 misses. Build a catalog of real error patterns.
3. Implement `CoACEExtractorV3` with calibrated patterns for the live distribution:
   - Natural language quantity chains: "P plus Q equals R" where R ≠ P+Q
   - Cumulative running totals: "bringing the total to X" where X diverges from sum
   - Percentage-of patterns: any "N% of M" → verify eval(N/100 * M) ≈ stated result
   - Division/split patterns: "divided equally among N groups" → verify stated per-group
   - Unit conversion verification: hours/minutes, km/m, etc.
4. Gate: `live_recall >= 0.30` opens Exp 594 (Live VR CoACE v3).

**Exp 592 (CPU-only):** DSVD Live Validation — Validate AUC=0.976 on Real FOVER Corpus.

Exp 587 achieved `dsvd_auc=0.976` with `honest_verdict=tier_2_5_viable`. The experiment
likely ran on the FOVER corpus which mixes synthetic and real pairs. This experiment:
1. Loads the FOVER corpus and isolates ONLY the live pairs (from Exps 578-579 with
   `inference_mode='live_gpu'`).
2. Runs DSVDAdapter on these live-only pairs.
3. Reports `live_auc`, `live_f1`, `live_precision`, `live_recall`.
4. Gate: `live_auc >= 0.80` opens Exp 595 (Live VR DSVD).
5. If validated, wire DSVDAdapter as Tier 2.5 in ThreeTierPipeline.

### Phase 2: JEPA v12 Live Corpus Retrain (Exp 593, CPU-only, FR-11 Mandatory)

**Exp 593 (CPU-only):** JEPA v12 Live Corpus Retrain — Validate CPMI on 100 Real Pairs.

JEPA v11 was trained on 9 synthetic pairs and achieved AUC=1.0 (likely overfitting).
This experiment:
1. Loads `results/live_pairs_578.json` (100 live pairs, GSM8K 0-49, 2 models).
2. Uses `JEPACPMIPairBuilder.build_pairs()` on the live corpus.
3. Trains JEPA v12 with `CPMIContrastiveLoss` on real contrastive pairs.
4. Evaluates with 80/20 train/val split, reports `val_auc`.
5. Saves to `results/jepa_predictor_v12.safetensors` if val_auc >= 0.70.
6. `retro_063_validated=True` if val_auc >= 0.70 (first validated live AUC).

### Phase 3: Live Verify-Repair Attempts (Exps 594-596, GPU required)

**Exp 594 (GPU-required, GATED on Exp 591):** Live VR CoACE v3 — RETRO-033 Attempt #13.

Gate: reads `results/experiment_591_coace_v3_live.json`. If `live_recall < 0.30`: write
blocked artifact and exit. 50 GSM8K questions (indices 300-349). Full verify-repair with
`CoACEExtractorV3`. `retro_033_resolved=True` if `signed_improvement > 0 AND inference_mode='live_gpu'`.

**Exp 595 (GPU-required, GATED on Exp 592):** Live VR DSVD Detection — RETRO-033 Attempt #13 Alt.

Gate: reads `results/experiment_592_dsvd_live_val.json`. If `live_auc < 0.80`: write
blocked artifact and exit. 50 GSM8K questions (indices 350-399). DSVD detection mode:
DSVDAdapter identifies the violation boundary token, then targeted repair regenerates
from that boundary only (not full response). This is a fundamentally different repair
architecture from CoACE post-hoc extraction + full regeneration.
`retro_033_resolved=True` if `signed_improvement > 0 AND inference_mode='live_gpu'`.

**Exp 596 (GPU-required, GATED on Exp 594 or 595):** Live 200q Wilson CI — RETRO-038 Attempt #9.

Gate: reads Exp 594 or 595; if either has `signed_improvement > 0`, proceed. 200 GSM8K
questions (indices 300-499) with the winning extractor (CoACE v3 or DSVD). Reports Wilson
confidence intervals. `retro_038_resolved=True` if `lower_CI > 0` (improvement above zero
with statistical confidence). This is the publishable credibility result.

### Phase 4: Self-Learning + New Research (Exps 597-599)

**Exp 597 (CPU-only):** FR-11 Real Violations v4 — Tier 1 Self-Learning with Live Violations.

If Exp 594 or 595 produced violations, use them to run `ConstraintAdditionFromMemory`.
Track which constraint patterns are added, measure FP rate before/after.
`fr11_real_violations_confirmed=True` if any new constraint is added from live violations.
This is the first time Tier 1 self-learning can operate on real constraint violation data.

Also: test `MISE dense reward calibration` (arXiv 2604.11611) on the (original, repair,
verdict) triples from Exp 594/595. MISE backward inference asks "what constraint was
this response trying to satisfy?" — score repair quality without per-step annotation.
Combined experiment: FR-11 relay + MISE calibration in one script.

**Exp 598 (CPU-only):** HISR Hindsight Segmental Process Rewards + D-Wave Cloud.

Two sub-experiments combined to save one experiment slot:

1. **HISR (arXiv 2603.18683):** Apply hindsight importance scores to
   `ConstraintAdditionFromMemory`. Instead of counting all violations equally, weight
   each violation by how much it predicted the final incorrect outcome. Compare weighted
   vs unweighted FP reduction.

2. **D-Wave Quantum Annealing Cloud:** Install `dwave-ocean-sdk` (pip, free tier).
   Run 100-spin Ising sampling via D-Wave's `neal` simulated annealer (local, no API
   key needed). Compare latency vs `ParallelIsingSampler`. If `dwave-cloud-client` is
   available, run one small QUBO on the real D-Wave Leap QPU free tier. Report:
   `neal_latency_ms`, `cpu_latency_ms`, `speedup_ratio`.

**Exp 599 (CPU-only):** Vivado Install Gate + GRPO Contrastive NUP Probe Retrain.

Two sub-experiments:

1. **Vivado Gate:** Check if Vivado 2023.2 is installed. If installed, run
   `vivado -mode batch -source hardware/kv260/synth_ising.tcl` to synthesize the
   bitfile. If not installed: generate exact CachyOS install commands (Arch pacman +
   Vivado installer path) in the result artifact for human action.

2. **GRPO Contrastive NUP Probe Retrain (arXiv 2503.06639):** Use the live benchmark
   pairs accumulated across Exps 578-595 as GRPO-style contrastive training data —
   (question, correct_response, incorrect_response) triples from live GPU inference.
   Retrain NUP Probe with this data. Target: AUC >= 0.750 on live FOVER corpus.

### Phase 5: Retrospective (Exp 600)

**Exp 600 (CPU-only):** Milestone 2026.04.45 Operational Retrospective.

---

## Dependency Graph

```
Exp 589 ──────────────────────────────── RETRO-067 partial closure
Exp 590 ──────────────────────────────── RETRO-062 prevention module
Exp 591 (CoACE v3) ──────────────────── gates Exp 594
Exp 592 (DSVD live) ─────────────────── gates Exp 595
Exp 593 (JEPA v12) ──────────────────── FR-11 live validation
Exp 594 (VR CoACE v3) ───────────────── RETRO-033 #13 (CoACE path)  ──┐
Exp 595 (VR DSVD) ───────────────────── RETRO-033 #13 (DSVD path)  ───┤
                                                                        └─→ Exp 596
Exp 596 (200q Wilson CI) ────────────── RETRO-038 #9 (publishable)
Exp 597 (FR-11 + MISE) ──────────────── depends on Exp 594/595 violations
Exp 598 (HISR + D-Wave) ─────────────── independent new research
Exp 599 (Vivado + GRPO NUP) ─────────── independent (FPGA + probe retrain)
Exp 600 (Retro) ──────────────────────── always last
```

---

## Success Criteria

| Criterion | Experiment | Target | Status |
|-----------|-----------|--------|--------|
| RETRO-066 resolved (live recall >= 30%) | Exp 591 | `live_recall >= 0.30` | Proposed |
| DSVD validated on live data | Exp 592 | `live_auc >= 0.80` | Proposed |
| JEPA v12 live validation | Exp 593 | `val_auc >= 0.70` | Proposed |
| RETRO-033 resolved (live improvement) | Exp 594 OR 595 | `signed_improvement > 0` | Proposed |
| RETRO-038 resolved (Wilson CI) | Exp 596 | `lower_CI > 0` | Proposed |
| FR-11 real violations confirmed | Exp 597 | `fr11_real_violations_confirmed=True` | Proposed |
| D-Wave sampling validated | Exp 598 | `neal_latency_ms < cpu_latency_ms` | Proposed |
| GRPO NUP Probe AUC improved | Exp 599 | `live_auc >= 0.750` | Proposed |

---

## Hardware Requirements

| Experiment | GPU Required | Model | VRAM Est. |
|-----------|-------------|-------|-----------|
| Exps 589-593 | No | CPU-only | 0 |
| Exp 594 | Yes (cuda:0+1) | Gemma4-E4B-it + Qwen3.5-0.8B | 12 + 2 GB |
| Exp 595 | Yes (cuda:0+1) | Gemma4-E4B-it + Qwen3.5-0.8B | 12 + 2 GB |
| Exp 596 | Yes (cuda:0+1) | Gemma4-E4B-it + Qwen3.5-0.8B | 12 + 2 GB |
| Exps 597-600 | No | CPU-only | 0 |

---

## Open RETROs Addressed

| RETRO | Priority | Milestone .45 Action |
|-------|----------|---------------------|
| RETRO-033 | CRITICAL (13 carries) | Attempt #13 via CoACE v3 AND DSVD parallel paths |
| RETRO-038 | HIGH (9 carries) | Attempt #9 once RETRO-033 shows improvement |
| RETRO-056 | HIGH | Exp 589: Conductor wrapper script (manifest wired) |
| RETRO-060 | CLOSED .44 | Monitor: JEPA v12 retrain validates |
| RETRO-063 | PARTIAL | Exp 593: Validate AUC=1.0 on 100 live pairs |
| RETRO-064 | CRITICAL | Exp 591: CoACE v3 live-calibrated extraction |
| RETRO-066 | NEW | Exp 591: Fix offline/live distribution gap |
| RETRO-067 | NEW | Exp 589: Wire manifest into conductor session |

---

## New Papers Incorporated

All papers below were found after milestone .44 planning and are incorporated into .45 experiments:

- **arXiv 2604.11611 MISE** — Dense reward calibration via backward inference; Exp 597 EORM calibration
- **arXiv 2603.18683 HISR** — Hindsight segmental process rewards; Exp 598 ConstraintAdditionFromMemory
- **arXiv 2503.06639 GRPO Verifiable Rewards** — Contrastive pairing from binary rewards; Exp 599 NUP Probe retrain
- **D-Wave Ocean SDK** — Quantum annealing cloud via free tier; Exp 598 QPU sampling
