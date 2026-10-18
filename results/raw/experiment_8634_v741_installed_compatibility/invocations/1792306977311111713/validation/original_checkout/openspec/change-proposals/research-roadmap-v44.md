# Research Roadmap v44 — Milestone 2026.04.44

**Status:** Proposed
**Milestone:** 2026.04.44
**Title:** Recall Surgery and Contrastive JEPA — First Verified Improvement on Live Models
**Planned Experiments:** 575–588 (14 experiments)
**Planned Date:** 2026-04-20 onwards

---

## What Milestone 2026.04.43 Proved

Milestone .43 confirmed four important facts and opened three new critical RETROs:

1. **CoACEExtractor achieves TP > 0 (RETRO-061 RESOLVED).** Exp 565 confirmed that
   CoACEExtractor (code-execution arithmetic checking) achieves coace_tp_rate > 0 on
   25 known-incorrect IT-model responses. gate_open=True. This is the first extraction
   method that fires on instruction-tuned model outputs.

2. **CoACE recall is only 5.9% — pipeline accuracy unchanged (RETRO-064 CRITICAL).**
   Exp 569 ran full live verify-repair with CoACEExtractor on 50 GSM8K questions.
   CoACE found 7 violations, applied 7 repairs, but signed_improvement=0.0 (26%→26%).
   Root cause: coace_recall=0.059 — only 1 of 17 incorrect responses was flagged.
   At 5.9% recall, the pipeline's expected accuracy lift is near-zero over 50 questions.
   CoACE recall must exceed ~30% before improvement is detectable.

3. **PURE objective insufficient for JEPA (RETRO-063 CRITICAL).** Exp 567 retrained
   JEPA v10 with the PURE min-form loss (arXiv 2504.15275). Result: v10_auc=0.4444,
   still below the 0.5 random baseline. Three consecutive retrains (v8: 0.444, v9: 0.4286,
   v10: 0.4444) all produce anti-correlated predictions. The PURE loss still allows hedging
   because step-score computation shares parameters with the energy function that makes
   all steps equally uncertain. The fix: explicit positive/negative pair construction where
   each training example pairs a correct and incorrect response to the same question.

4. **FR-11 real violations confirmed (MAJOR POSITIVE).** Exp 570 found 12 violations
   across 25 live GSM8K questions using CoACEExtractor. fr11_real_violations_confirmed=True.
   First time the self-learning relay operates on real constraint violations.

5. **Live 50q A still unrun (RETRO-062, 3rd consecutive miss).** Exp 563 was blocked
   because CARNOT_FORCE_LIVE was not set at session start. Questions 0-49 remain absent
   from the FOVER corpus for the third consecutive milestone.

6. **KV260 FPGA board arrived but bitfile not synthesized.** Exp 568 confirmed the board
   is physically present; honest_verdict=synthesis_required. Vivado synthesis command is
   documented; bitfile generation is a manual step requiring Vivado installation.

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: CoACE Recall at 5.9% — Pipeline is Statistically Invisible (RETRO-064, FR-12)

**Root cause confirmed:** CoACEExtractor currently matches simple `A op B = C` equations
in a single pass. It misses:
- Multi-step chains: "first X = A + B, then Y = X × C, finally total = Y + D = E"
- Percentage arithmetic: "20% of 150 is 30" (pattern: N% of M is P)
- Bracket expressions: "(47 + 28) / 3 = 25.0" 
- Approximate equality: "approximately 150" (near violations)
- Cumulative tracking: carried values reused in later steps (chain of 3+ operations)

At 5.9% recall, even perfect repair would yield +0.06pp accuracy improvement — statistically
invisible over 50 questions and indistinguishable from noise. The pipeline cannot demonstrate
its value until recall exceeds ~30%.

**The fix:** Expand CoACEExtractor pattern coverage in three directions:
1. Multi-step chain tracking: parse the entire CoT as a sequence of equations, track
   intermediate results, verify chain consistency via execution of the full computation graph.
2. Prose arithmetic patterns: "N% of M" → eval "N/100 * M"; "P divided by Q" → eval "P/Q";
   "X times Y" → eval "X * Y".
3. Approximate equality tracking: build a numeric context dictionary mapping variable
   names to computed values, detect when a stated approximation contradicts the exact value.

**Why PRD cares:** FR-12 (Verifiable Reasoning) — the system must identify constraint
violations. At 5.9% recall, Carnot identifies 5.9% of violations. This is not a product.

### Gap 2: JEPA Anti-Correlation Persists — Three Retrains, Same Failure Mode (RETRO-063)

**Root cause confirmed:** All three retrains (v8, v9, v10) used variants of a cross-entropy
loss applied to step-level binary labels. The model finds a local minimum at P=0.5 everywhere
because step-level labels are noisy (many intermediate steps in a correct chain briefly
look "wrong" before the final correct answer emerges). The BCE loss cannot distinguish
"step is intermediate and looks wrong" from "step is actually wrong."

**The fix (explicit contrastive pair construction):** Instead of labeling individual steps,
pair WHOLE CHAINS by question:
- Positive chain: Qwen3.5-0.8B's correct response to question Q
- Negative chain: Qwen3.5-0.8B's incorrect response to question Q (or Gemma4's incorrect response)
- Loss: `hinge(E(negative_chain) - E(positive_chain) - margin)` — force incorrect chains
  to have higher energy than correct chains for the SAME question.

This eliminates step-level noise: we don't need to know which STEP is wrong, only that
the incorrect chain as a whole should score lower. The (Q, correct, incorrect) triple
is constructable from the FOVER corpus where both models answered the same question.

**Why PRD cares:** FR-11 (Autonomous Self-Learning Loop) — JEPA predictive verification
is Tier 3 of the self-learning architecture. A predictor with AUC < 0.5 inverts the
signal — it would direct inference TOWARD violations rather than away from them.

### Gap 3: Live 50q A Missing — FOVER Corpus Systematically Incomplete (RETRO-062)

**The problem:** GSM8K questions 0-49 have been absent from the FOVER corpus across
THREE consecutive milestones (.42, .43, .44 if we don't fix it). The corpus now has 132
pairs but with a systematic gap in question coverage.

**The fix:** Hard preflight abort. Run Live 50q A as the FIRST experiment in the conductor
session. Add a Python assertion that raises before any model load if CARNOT_FORCE_LIVE
is not '1'. The conductor must be forced to prioritize this or exit.

---

## Architecture Diagram (Verification Cascade — .44 State)

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
    │  low partition variance → skip downstream
    ▼
[Tier 1]  SinkProbe (attention sink concentration, arXiv 2604.10697)
    │  mean_sink >= threshold → skip downstream
    ▼
[Tier 2]  EORM (energy reward model, 55M params, GRPO-retrained on real data)
    │  energy < threshold → verified
    ▼
[Tier 3]  CoACEExtractor v2 [NEW .44, recall >30%] + VerifyRepairPipeline
    │      (Code-Assisted Constraint Extraction with chain tracking + prose patterns)
    │      parse arithmetic graph → execute → compare → violation or verified
    ▼
    Verified / Violated + Repair Suggestion

[Tier 3 alt] PRA EBM Beam Search [added .43, arXiv 2604.09482]
    │  K=3 candidates per step, EORM selects minimum energy
    └→ proactive constraint satisfaction during generation

[Tier 3 fast-path] JEPA v11 [NEW .44, contrastive pairs]
    Partial CoT (first 50%) → JEPA v11 → energy_score
    → high energy: run full Tier 3 verification
    → low energy: skip Tier 3 (fast-path approval)

[Symbolic-KAN Energy Tier] [NEW .44, arXiv 2603.23854]
    KAN energy function with interpretable symbolic rules
    → "fires when a + b ≠ stated_result" (human-readable constraint)
```

---

## Phase Descriptions

### Phase 0: Process Infrastructure (Exp 575, CPU-only, MUST BE FIRST)

**Exp 575 (CPU-only):** Build the conductor exclusion manifest — `scripts/conductor_exclusion_manifest.json`.

RETRO-056 has carried 7 consecutive milestones (.37 through .43) without implementation.
This is the **top process priority** before any research work. The manifest lists experiments
that are fully complete and should never be re-entered. Without it, the conductor continues
to spend ~385 minutes/milestone re-running the same 5 slowest legacy experiments.

The exclusion manifest is a simple JSON: `{"excluded": [308, 260, 309, 425, 410]}` plus a
loader in `scripts/check_exclusion_manifest.py` that the conductor can call at session start.
Cumulative waste from 7-milestone carry: ~2,485 minutes (41.4 hours).

### Phase 1: Recall Surgery — RETRO-064 Critical Path (Exps 576-577, CPU-only)

**Exp 576 (CPU-only):** CoACEExtractor v2 — expand recall from 5.9% to >30%.

Three new extraction capabilities:
1. **Multi-step chain tracking:** Parse the full CoT as a computation graph. Assign symbolic
   variable names to intermediate results (step1_result = eval(expr1), step2_result = step1_result + B, etc.).
   Detect when a stated value diverges from the computed value at any point in the chain.
2. **Prose arithmetic patterns:** Handle "N% of M", "X times Y", "P divided by Q",
   "difference between P and Q", "ratio of P to Q". Translate to Python expressions before eval().
3. **Numeric context tracking:** Build a dictionary of named quantities as the CoT progresses.
   When "the answer" is stated, verify it against the accumulated computation result.

Target: recall > 30% on Exp 565's 25 known-incorrect live responses.
Deliverable: `python/carnot/extraction/coace_extractor_v2.py`.

**Exp 577 (CPU-only):** JEPA CPMI Contrastive Pair Builder — redesign training data construction using hard-negative mining (arXiv 2604.10660).

CPMI (Contrastive Pointwise Mutual Information) constructs contrastive training pairs by finding
the hardest negatives — incorrect reasoning steps that maximally increase MI with the wrong answer.
For Carnot:
1. Group FOVER corpus entries by question_id (same question, multiple model responses).
2. For each group with both correct and incorrect responses: yield (question, correct_chain, incorrect_chain) triples.
3. Build `JEPACPMIPair` dataclass: question_id, correct_embeddings (list of step embeddings), incorrect_embeddings, hard_negative_step_idx (index of most misleading step).
4. Hard-negative selection: for each incorrect chain, identify the step with highest energy agreement with the wrong final answer (most "convincing" wrong step). This is the hard negative.
5. Contrastive loss: `hinge(E(incorrect_chain) - E(correct_chain) - margin)` where
   `chain_energy = mean(step_energy)` — not min-form, not BCE. Hard negatives provide stronger training signal than random negatives.
6. Validation: if FOVER corpus has fewer than 10 cross-model pairs, generate synthetic pairs where one response has an injected arithmetic error at a random step.

Deliverable: `python/carnot/inference/jepa_cpmi_pairs.py`.

### Phase 2: Live Data Sprint (Exps 578-579, GPU required, SECOND priority after Phase 0)

**Exp 578 (GPU-required):** Live 50q A v3 — RETRO-062 closure, GSM8K questions 0-49.

HARD PREFLIGHT: raise `RuntimeError("CARNOT_FORCE_LIVE must be '1' — run: source scripts/session_startup.sh")` 
as the FIRST line before any import of transformers or torch if `os.environ.get('CARNOT_FORCE_LIVE') != '1'`.
Write blocked artifact and `sys.exit(1)` if gate fails. This is the third attempt; gate must
be enforced at import-time, not just before model load.

Collect GSM8K 0-49, FOVER-annotated, same structure as Exp 552 (successful batch B).
Target: n_pairs_collected >= 40.

**Exp 579 (GPU-required):** Live 50q C — GSM8K questions 200-249.

New data batch to expand the FOVER corpus from ~132 pairs to ~182+ pairs. Same structure
as Exp 552 and 578. Target: n_pairs_collected >= 40.

### Phase 3: JEPA v11 Contrastive Retrain (Exp 580, CPU-only, FR-11 Mandatory)

**Exp 580 (CPU-only):** JEPA v11 Contrastive Retrain with explicit pair construction.

Uses `JEPAContrastivePairs` from Exp 577. Trains on FOVER corpus v2 (132 pairs) plus
any new pairs from Exps 578-579 if available. Architecture unchanged (embed_dim=128, n_layers=2).
Loss: explicit contrastive margin — NOT BCE, NOT PURE, NOT min-form.
Save best checkpoint (by val AUC) to `results/jepa_predictor_v11.safetensors`.
Target: AUC >= 0.600 (first time above random across 4 retrain attempts).

### Phase 4: Integration Validation (Exps 581-583, CPU/GPU, gated on Phase 1)

**Exp 581 (CPU-only):** CoACE Recall Diagnostic v2 — validate v2 extractor.

Run CoACEExtractor v2 on Exp 565's 25 known-incorrect live responses.
Compare: v1 recall (5.9%) vs v2 recall (target >30%).
**GATE:** if v2_recall < 0.20, write blocked artifact for Exps 582-583 (still too low to detect improvement).
This gate prevents wasting GPU time on a pipeline that cannot demonstrate improvement.

**Exp 582 (GPU-required, GATED on Exp 581):** Live VR CoACE v2 — RETRO-033 attempt #12.

Run full live verify-repair with CoACEExtractor v2 on 50 GSM8K questions (indices 250-299).
Target: signed_improvement > 0 AND inference_mode='live_gpu' → RETRO-033 resolved.
**Do not run if Exp 581 v2_recall < 0.20.**

**Exp 583 (GPU-required, GATED on Exp 581):** FR-11 Real Violations v3 — relay with improved CoACE.

Run Tier 1 self-learning relay with CoACEExtractor v2.
Track violations found per batch, constraint addition rate, FP trend.
Deliverable: fr11_real_violations_v3_confirmed field with expanded recall.

### Phase 5: FPGA Hardware Acceleration (Exps 584-585)

**Exp 584:** KV260 Vivado Synthesis — generate bitfile.

Run `vivado -mode batch -source hardware/kv260/synth_ising.tcl`. This is a SYNCHRONOUS
long-running command (~30-60 minutes for synthesis). The experiment script should:
1. Check if Vivado is installed (`which vivado` or `vivado -version`).
2. If installed: run synthesis, capture output, detect success via `[SUCCESS]` in log.
3. If NOT installed: print detailed installation instructions for Xilinx Vivado 2023.2
   and write artifact with honest_verdict='vivado_not_installed'.
4. If synthesis succeeds: verify output bitfile exists at `output/carnot_ising_synth/carnot_ising.bit`.

Deliverable: `results/experiment_584_kv260_synthesis.json` with `bitfile_built` and `bitfile_path`.

**Exp 585 (GATED on Exp 584 bitfile_built=True):** KV260 Live Benchmark v3.

Set `CARNOT_KV260_BITFILE` to the synthesized bitfile path.
Run 100-spin Ising sampler benchmark (1000 trials) against CPU baseline (~290ms/call).
Target: hardware_latency_us < 100 (vs CPU baseline ~290,000 μs → 2900x speedup).
Report speedup_ratio, energy_per_sample, and whether FPGA path is viable for production.

### Phase 6: New Research (Exps 586-587, CPU-only)

**Exp 586:** Symbolic-KAN Energy Interpretability (arXiv 2603.23854).

Replace KAEMEnergy's continuous spline activations with discrete symbolic equations.
The energy function becomes human-readable: "constraint fires when a + b ≠ c where a,b,c
are parsed from CoT". This directly addresses the pipeline's opacity — users cannot
currently understand why CoACE fires on a given input.

Implementation: `SymbolicKANEnergy` class — fit symbolic regression on energy function inputs
vs outputs; select best symbolic form per activation (linear, polynomial, exponential, trig);
export human-readable formula string.

Deliverable: `python/carnot/models/symbolic_kan_energy.py` + interpretability report.

**Exp 587:** DSVD — Dynamic Self-Verify Decoding Adapter (arXiv 2503.03149, EMNLP 2025).

DSVD detects hallucinations in real-time via parallel hidden-state analysis during generation,
then applies dynamic rollback to correct the specific problematic tokens rather than
regenerating the full response. Carnot's current post-hoc pipeline (generate → verify → repair)
could be improved by mid-generation interception.

Implementation (CPU prototype, no live GPU needed):
1. `DSVDVerificationHead` — lightweight linear probe on Qwen3.5-0.8B hidden states (layer -4):
   - Input: hidden state tensor at step boundary (every 32 tokens)
   - Output: arithmetic_violation_probability (scalar 0-1)
2. `DSVDAdapter` — wraps generation loop:
   - At each 32-token boundary, call DSVDVerificationHead
   - If violation_probability > threshold: record "mid-generation violation detected" 
   - Compare detection time vs CoACEExtractor (post-hoc) on same CoT steps
3. Benchmark on FOVER corpus v2 (synthetic hidden states from saved embeddings):
   - AUC: mid-generation DSVD vs post-hoc CoACE
   - Latency: per-step probe vs full extraction

Connection to Carnot pipeline: If DSVD achieves AUC > 0.60, it can be inserted as a
Tier 2.5 between EORM and CoACE — high-probability violations get CoACE confirmation,
low-probability get DSVD fast-path skip.

Deliverable: `python/carnot/pipeline/dsvd_adapter.py` + AUC comparison chart.

### Phase 7: Operational Retrospective (Exp 588)

**Exp 588:** Milestone 2026.04.44 Operational Retrospective.

Standard milestone retrospective. Key questions:
1. Did CoACE v2 achieve recall > 30% (RETRO-064 partially resolved)?
2. Did JEPA v11 achieve AUC >= 0.600 (RETRO-063 resolved)?
3. Did Live 50q A collect >= 40 pairs (RETRO-062 resolved)?
4. Did live verify-repair with CoACE v2 show signed_improvement > 0 (RETRO-033 resolved)?
5. Did KV260 bitfile synthesis complete (hardware progress)?

---

## Dependency Graph

```
Exp 575 (ExclusionManifest) — no dependencies, must be first

Exp 576 (CoACE v2)    ─────────────────────────┐
Exp 577 (JEPA Pairs)  ──────────┐              │
                                 │              │
Exp 578 (50q A v3)   ─────────────────────────────────── Exp 579 (50q C)
                                 │              │
                       Exp 580 (JEPA v11) ◄─────┘ (depends on 577, uses 578+579 data if available)
                       
Exp 581 (CoACE Recall Diag v2) ←── depends on 576 (CoACE v2 + 565 known-incorrect responses)
Exp 582 (Live VR v2) ←── GATED on 581 recall > 0.20 + GPU
Exp 583 (FR-11 v3)   ←── GATED on 581 recall > 0.20 + GPU

Exp 584 (Synthesis)   ─────┐
Exp 585 (Live FPGA) ◄──────┘ (GATED on 584 bitfile_built=True)

Exp 586 (Symbolic-KAN) — independent
Exp 587 (Mythos)       — independent

Exp 588 (Retro) ←── depends on all previous
```

---

## Success Criteria

| Criterion | Experiment | Success Definition |
|-----------|------------|-------------------|
| retro_064_partial | Exp 581 | coace_v2_recall >= 0.20 |
| retro_064_resolved | Exp 581 | coace_v2_recall >= 0.30 |
| retro_063_resolved | Exp 580 | jepa_v11_auc >= 0.600 |
| retro_062_resolved | Exp 578 | n_pairs_collected >= 40 |
| retro_033_resolved | Exp 582 | signed_improvement > 0 AND inference_mode='live_gpu' |
| retro_056_resolved | Exp 575 | exclusion_manifest_built=True AND conductor_consulted=True |
| fr11_relay_improved | Exp 583 | fr11_violations_found > 0 AND constraints_added > 0 |
| fpga_progress | Exp 584 | bitfile_built=True OR vivado_installed=True |
| symbolic_viable | Exp 586 | formula_interpretable=True AND energy_preserved=True |
| phase3_validated | Exp 587 | accuracy_improves_with_T=True AND energy_monotone=True |

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|----------|-------|
| Exp 575 | CPU | ExclusionManifest — no GPU needed |
| Exp 576 | CPU | CoACE v2 — pure Python arithmetic parsing |
| Exp 577 | CPU | JEPA pair builder — small embedding operations |
| Exp 578 | GPU (cuda:0, ~10GB + cuda:1, ~1.5GB) | CARNOT_FORCE_LIVE=1 mandatory |
| Exp 579 | GPU | Same as Exp 578 |
| Exp 580 | CPU | JEPA training on 132-182 pairs — small model |
| Exp 581 | CPU (GPU fallback) | CoACE recall diagnostic — uses saved responses |
| Exp 582 | GPU (CARNOT_FORCE_LIVE=1) | GATED on Exp 581 recall |
| Exp 583 | GPU (CARNOT_FORCE_LIVE=1) | GATED on Exp 581 recall |
| Exp 584 | CPU + Vivado install | FPGA synthesis — requires Xilinx Vivado 2023.2 |
| Exp 585 | KV260 FPGA board + CARNOT_KV260_BITFILE | GATED on Exp 584 |
| Exp 586 | CPU | Symbolic-KAN — pure JAX operations |
| Exp 587 | CPU | Phase 3 prototype — small MLP + energy descent |
| Exp 588 | CPU | Retrospective |

---

## Open RETROs Being Addressed

| RETRO | Description | Addressed By | Expected Resolution |
|-------|-------------|--------------|---------------------|
| RETRO-031 | KAEM speedup only 1.29x | Not in .44 (low priority, deferred to .45) | — |
| RETRO-033 | Live VR 0% improvement (11 attempts) | Exp 582 | Depends on Exp 581 recall gate |
| RETRO-038 | VeriCoT+VPRM live 200q benchmark | Not in .44 (superseded by CoACE) | — |
| RETRO-056 | Exclusion manifest not built (7 milestones) | Exp 575 | CRITICAL — must resolve |
| RETRO-062 | Live 50q A not collected (3 misses) | Exp 578 | Hard preflight gate |
| RETRO-063 | JEPA anti-correlated (3 retrains) | Exps 577+580 | Contrastive pair redesign |
| RETRO-064 | CoACE recall only 5.9% | Exps 576+581 | Expanded pattern coverage |
| RETRO-065 | RAPL unavailable | Not addressed in .44 (AMD machine, low priority) | — |

---

## New RETROs Expected

If JEPA v11 still shows AUC < 0.5 after contrastive pair redesign:
- RETRO-066: JEPA architecture fundamentally broken — must redesign predictor model
  (larger encoder, attention-based step embedding, contrastive pre-training)

If CoACE v2 recall still < 20%:
- RETRO-067: Arithmetic pattern coverage insufficient — needs LLM-based extraction
  (use a second LLM call to extract equations from prose CoT, then execute them)

If FPGA synthesis fails due to Vivado not installed:
- RETRO-068: Vivado installation required before next FPGA attempt

---

## HuggingFace Publishing Progress

After this milestone (if RETRO-033 resolved):
- Publish first CoACE v2 extractor model weights to HuggingFace
- Update 16 existing EBM model READMEs to reference new extraction method

---

## Connection to Long-Term Vision

### Phase 3 (EBM Foundation Model)

Exp 587 (EBM Mythos Prototype) directly validates the Phase 3 architecture:
- Prelude → IsingRepairLoop → Coda is the discrete-to-continuous bridge
- If accuracy-vs-T shows improvement, this validates the iterative energy descent
  approach for non-autoregressive reasoning
- The hardware path is clear: IsingRepairLoop maps directly to FPGA Ising sampler (Exp 584-585)

### Self-Learning Tiers

| Tier | Status After .43 | Target in .44 |
|------|-----------------|---------------|
| 1: Online weights | Working (fr11_violations=True, 12 found) | Wire v2 extractor for higher violation density |
| 2: Constraint memory | Working (1 constraint added) | Higher recall → more patterns accumulated |
| 3: JEPA predictive | Anti-correlated (AUC=0.44) | Contrastive pairs → first above-random AUC |
| 4: Adaptive structure | Not started | Exp 587 Phase 3 prototype |
