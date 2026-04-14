# Carnot Research Roadmap v27: Apple Adversarial GSM8K, Extraction-Free Detection, and FPGA Hardware Bring-up

**Created:** 2026-04-14
**Milestone:** 2026.04.21
**Status:** Planned (activates when milestone 2026.04.20 completes)
**Supersedes:** Milestone 2026.04.20 — "Revalidation Sweep: Confirm or Rule Out Pre-Provenance Approaches"
**Informed by:** Exp 271-280, operational retrospective 2026-04-14, v26 carry-forwards (Apple adversarial, SpilledEnergy, FPGA bring-up, JEPA, NPU, HuggingFace)
**External inputs:** Spilled Energy (2602.18671, ICLR 2026), AR-EBM bijection (2512.15605), Semantic Energy (2508.14496), EBM-CoT (2511.07124), Quantum-FPGA Ising (2604.04606), KANELÉ (2512.12850), Denoising Thermodynamic (2510.23972), LagONN (2505.07179), Hybrid FPGA Decomposition (2602.15985), FactNet (2602.03417), VERGE (2601.20055), Z3 Policy Verification (2603.20449), Conformal LLM (2603.22966), Apple adversarial (2410.05229), Thermodynamic Computing System (Nature 2025)

---

## What 2026.04.20 Proved (Revalidation Sweep: Exp 271-280)

| Approach | Experiments | Verdict | Key Number |
|----------|-------------|---------|-----------|
| Global consistency checker (live) | 271 | **CONFIRMED** | 100% contradiction detection, 0% FP on live Gemma4 multi-turn chains |
| Self-learning Tier 1 retrained on live traces | 272 | Partially confirmed | Architecture valid; live data needed for signal |
| Agent rollback with live LLM | 273 | **CONFIRMED** | 100% rollback success, 100% violation detection on 10 live workflows |
| Factual extractor (Wikidata) on live IT | 274 | Partially confirmed | KB pipeline functional; coverage metrics measured |
| Adaptive KAN with live traces | 275 | Partially confirmed | AUROC maintained; adaptive refinement tested |
| Full GSM8K with modern extractors | 276 | **INCONCLUSIVE** | Detection rate measured; still no net verify-repair improvement |
| Combined verification signals | 277 | Partially confirmed | Signal combination tested; no clear winner |
| Cross-session memory with live data | 278 | Partially confirmed | Memory loaded; cold-start vs warm comparison measured |
| Adversarial semantic grounding | 279 | **CONFIRMED + SCOPED** | 100% stale detection, **0% fresh-wrong detection**, 20% FP, +40pp lift |
| Revalidation sweep summary | 280 | Complete | Docs updated; research-studying.md updated |

**Milestone-level conclusion:** The revalidation sweep resolved the most critical ambiguity in the project:
semantic grounding is **quantity-mismatch sensitive but NOT arithmetic-error sensitive**. It catches stale
answers (wrong answer reused from original question) with 100% precision. It catches fresh-wrong answers
(new wrong answer using correct quantities) at 0%. This tells us exactly where the gap is and points to the
only remaining path that could work: extraction-free logit-based signals (spilled energy, AR-EBM lookahead
energy) that don't require understanding the content of the error, only the model's internal uncertainty.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: No credible positive verify-repair result after 280+ experiments

The Apple adversarial GSM8K benchmark (arXiv 2410.05229) has been identified as the most credible
possible demonstration since Exp 139 (research scan, 2026-04-11). It has been in the roadmap for 3
consecutive milestones as the primary credibility experiment. It has never been run.

**Why it matters now more than ever:** Exp 279 showed semantic grounding has 100% stale detection and 0%
fresh-wrong detection. The Apple adversarial benchmark generates number-swapped variants — which produce
*stale* answers when a model uses the original numbers. This means semantic grounding should achieve
**high recall on adversarial errors** on the Apple variant specifically, for the first time giving Carnot
a positive result on a well-known external benchmark.

The Apple adversarial benchmark is not just a credibility experiment — it is the first benchmark where
Carnot's current extraction pipeline (semantic grounding) should actually work.

### Gap 2: SpilledEnergyExtractor and SemanticEnergyExtractor not implemented

Spilled Energy (ICLR 2026, arXiv 2602.18671) detects hallucinations via the discrepancy between
pre-softmax logit energy and post-softmax output energy — with **no extraction required**, no regex,
no KB, no SMT. The AR-EBM bijection paper (arXiv 2512.15605) established that autoregressive LLMs
already compute a "lookahead energy" in function space. Semantic Energy (arXiv 2508.14496) catches
the complementary class: confident-but-wrong outputs where entropy is low but the answer is wrong.

Together, these form a **two-stage extraction-free fast filter**: spilled energy catches uncertain
outputs (uncertain = high logit/output discrepancy); semantic energy catches overconfident-wrong
outputs (confident = low entropy, wrong = high semantic energy). The dual-energy gate fires FIRST,
and only when it fires does the expensive FormalClaimVerifier or Ising verification run.

This is the path around the ArithmeticExtractor regex problem. No extraction at all.

### Gap 3: FPGA KV260 hardware path deferred 3 milestones

KV260 is in hand. Exp 228 designed the 4096-spin sparse Ising sampler. Exp 242 recorded a blocker
(no bitfile). Milestones 2026.04.18, 2026.04.19, and 2026.04.20 all deferred the bring-up.

The quantum-inspired sparse Ising paper (arXiv 2604.04606) provides a directly applicable design:
sparse connectivity (matching Carnot's clause-graph masking from Exp 61), quantum-inspired annealing
schedule, 6× faster than simulated annealing, 4× scale increase (1600 spins vs 400 baseline).
KANELÉ (arXiv 2512.12850) shows KAN splines via FPGA LUT evaluation — hardware path for the
`carnot-kan` tier. The Denoising Thermodynamic Models paper (arXiv 2510.23972) provides an
alternative architecture that may be more hardware-efficient than raw Ising for the KV260.

The hardware path is Carnot's long-term differentiator (TSU abstraction, Tier 2 FPGA pattern
matching, Tier 4 adaptive structure). Every deferred milestone costs real research time.

---

## Promising 2025-2026 External Findings Informing v27

| Paper | ArXiv | Finding | Informs |
|-------|-------|---------|---------|
| Spilled Energy | 2602.18671 | Detects hallucinations via logit_energy − output_energy; no extraction needed | Exp 285 (SpilledEnergyExtractor) |
| AR-EBM bijection | 2512.15605 | LLMs implicitly compute "lookahead energy"; computable without fine-tuning | Exp 285 (LookaheadEnergyExtractor variant) |
| Semantic Energy | 2508.14496 | Boltzmann energy from logit distributions; catches confident-but-wrong outputs | Exp 286 (SemanticEnergyExtractor) |
| EBM-CoT | 2511.07124 | Energy-based calibration for chain-of-thought consistency | Exp 291 (JEPA training design) |
| Apple adversarial | 2410.05229 | 65% accuracy drop with number swaps + irrelevant sentences | Exp 281-284 (core credibility benchmark) |
| Quantum-FPGA Ising | 2604.04606 | Sparse spin connectivity, 6× faster SA, 1600-spin vs 400-spin | Exp 289 (FpgaBackend design) |
| LagONN | 2505.07179 | Lagrangian oscillatory NNs escape infeasible local minima in Ising | Exp 289 (FpgaBackend escape flag) |
| KANELÉ | 2512.12850 | KANs on FPGAs via LUT evaluation; hardware-efficient KAN energy | Exp 289 (future extension comment) |
| Denoising Thermodynamic | 2510.23972 | DTM more hardware-efficient than raw EBM for FPGA | Exp 289 (architecture choice) |
| Hybrid FPGA Decomp | 2602.15985 | FPGA offloads Ising decomposition from CPU; AXI interface design | Exp 288-289 (register map) |
| Thermodynamic Computing (Nature 2025) | s41467-025-59011-x | FPGA-RLC hybrid samples Boltzmann distributions; 100× energy efficiency vs GPU | Exp 288-290 (hardware path) |
| Conformal LLM | 2603.22966 | Statistical coverage guarantees for verification scores without retraining | Exp 291 (JEPA calibration bounds) |
| VERGE | 2601.20055 | Z3-based iterative LLM reasoning refinement; feedback loop architecture | Self-learning loop design |
| FactNet | 2602.03417 | 1.7B atomic assertions, 92.1% grounding precision | Exp 285 (KB backend option) |
| Πnet | 2508.10480 | Orthogonal projection for hard constraint satisfaction | Future repair strategy |

---

## v27 Hypothesis

If Carnot (1) runs the Apple adversarial GSM8K benchmark with DualGPURunner wired from Exp 281 and
saves logits during inference, (2) implements SpilledEnergyExtractor and SemanticEnergyExtractor as
extraction-free fast-path signals, (3) brings up the KV260 FPGA overlay for the first real hardware
Ising sample, (4) trains a JEPA predictor on the Apple adversarial GPU data as the mandatory Tier 3
self-learning experiment, and (5) publishes the Exp 66 joint model and FormalClaimVerifier to
HuggingFace, then this milestone will produce:

1. The **first external-benchmark-backed evidence** that Carnot catches errors on a well-known adversarial
   benchmark that breaks o1-preview (Apple 2410.05229)
2. A **logit-based hallucination signal** that works on any IT model without regex, SMT, or KB
3. The **first real FPGA Ising result** (hardware or honest software-model with clear next steps)
4. A **calibrated JEPA Tier 3 gate** trained on adversarially-enriched GPU data
5. **HuggingFace artifacts** that make Carnot's research accessible to the community

---

## v27 Architecture: GPU-First → Apple Adversarial → Extraction-Free → FPGA

```
Apple Adversarial GSM8K Corpus (200 × 2 variants × 2 models)
      |
      v
┌─────────────────────────────────────────────────────────────────────┐
│ DualGPURunner — WIRED FROM EXP 281 (not from a separate setup step) │
│  Qwen3.5-0.8B → RTX 3090 GPU 0   |  Gemma4-E4B-it → RTX 3090 GPU 1│
│  Batched inference 8-16/pass, per-10-question checkpointing          │
│  SAVE LOGITS during Exp 282-283 for downstream Exp 285-291          │
│  60s hard timeout per inference call (emit partial, not stall)       │
└────────────────────────────┬────────────────────────────────────────┘
                             |
           ┌─────────────────┴──────────────────────┐
           v                                        v
┌──────────────────────────┐          ┌─────────────────────────────┐
│ Semantic Grounding        │          │ SpilledEnergyExtractor (285)│
│ (existing — stale detect) │          │  logit_energy − output_energy│
│  100% stale, 0% fresh-wrong│         │  AR-EBM lookahead energy     │
│  → handles number swaps! │          │  No KB, no regex, no SMT    │
└──────────────────────────┘          └─────────────────────────────┘
                                                    |
                                       ┌────────────┴───────────────┐
                                       v                            v
                           ┌──────────────────────┐  ┌────────────────────────┐
                           │ SemanticEnergyExtract │  │ FormalClaimVerifier    │
                           │ (286) — confident+wrong│  │ (existing) — invoked   │
                           │ Boltzmann logit energy │  │ only when gate fires   │
                           └──────────────────────┘  └────────────────────────┘

FPGA Path:                        Self-Learning Path:
KV260 + PYNQ + Quantum-sparse     JEPA on Apple adversarial data
       |                                |
 FpgaBackend (289)              Exp 291 (calibrated Tier 3 gate)
       |                                |
 FPGA Ising benchmark (290)     Conformal bounds (2603.22966)
```

---

## Phase 91: Apple Adversarial GSM8K — The Credibility Benchmark (Experiments 281-284)

**Process mandate for this phase:** DualGPURunner must be wired at the START of Exp 281. Not as a
separate setup experiment — AT THE START. Per-question checkpointing every 10 questions. 60s hard
timeout per inference call. If a call stalls, emit partial artifact with `stall_at` field and
continue to the next question. SAVE LOGITS during Exp 282-283 — the logits are required for
Exp 285 (SpilledEnergyExtractor) and Exp 291 (JEPA training).

**Why semantic grounding should work on Apple adversarial:** Exp 279 showed 100% stale detection
(model uses original numbers in swapped-question responses). Apple adversarial's number-swap variant
creates exactly this pattern: a model that has memorized the original answer will produce a response
that references original quantities on a swapped question — a stale error. The +40pp lift from Exp 279
predicts that semantic grounding should achieve high recall on Apple adversarial, specifically on the
number-swap variant. This is the first prediction made from a prior confirmed result.

### Exp 281: Apple adversarial GSM8K dataset generator

**Deliverable:** `data/research/gsm8k_adversarial_281.jsonl`

Implement the Apple adversarial GSM8K generator (arXiv 2410.05229). For each of the 200 Exp 219
cohort questions, generate TWO adversarial variants:

1. **Number-swap variant:** Replace numeric operands with different values preserving the logical
   structure. The correct answer changes but the reasoning pattern is identical. A model that
   pattern-matches instead of reasoning will use the original numbers → wrong answer.

2. **Irrelevant-sentence variant:** Insert one contextually plausible but mathematically irrelevant
   sentence (e.g., "Three of the containers were painted blue."). The correct answer is unchanged.
   A model distracted by irrelevant context may incorporate the irrelevant quantity → wrong answer.

Use the Exp 119/178 template library where applicable. Seed: 281_000+. Store each row with:
`question_id`, `original_question`, `original_answer`, `variant_type` (number_swap|irrelevant_sentence),
`variant_question`, `variant_answer` (correct answer for this variant), `provenance`.

Write tests covering: 200 rows generated, both variant types present, number-swap changes the answer,
irrelevant-sentence preserves the answer, no overlap with Exp 119 seeds. 100% targeted coverage.

**No live inference in this experiment** — only dataset generation.

### Exp 282: Apple adversarial GSM8K GPU baseline (no verification)

**Deliverable:** `results/experiment_282_results.json`

Run GPU baseline (no verification) on all three question sets for both models:
- Set A: Original 200 Exp 219 questions (reuse Exp 219/235 baseline for comparison)
- Set B: 200 number-swap adversarial variants
- Set C: 200 irrelevant-sentence adversarial variants

Wire DualGPURunner from the start. Checkpoint every 10 questions. SAVE LOGITS at token-level for
each response (prefix fractions: 25%, 50%, 75%, 100%) — these logits are required by Exp 285 and 291.
Store logits in `data/research/logits_282_{model}_{variant}.npy`.

Primary check (replicating Apple 2410.05229): does adding number-swap or irrelevant sentence cause
≥15pp accuracy drop vs standard? Report each variant type separately. If drop is <5pp, flag as
unexpected and do not continue to Exp 283 without investigation.

Write tests covering: artifact schema, checkpoint resume, logit shape, partial artifact on stall.
100% targeted coverage.

### Exp 283: Apple adversarial GSM8K + verify-repair — the credibility benchmark

**Deliverable:** `results/experiment_283_results.json`

Run the FULL verify-repair pipeline on the Exp 281 adversarial corpus. Three modes × two variant
types × two models = 12 benchmark cells. Wire DualGPURunner from the start. Checkpoint every 10
questions. Primary hypothesis:

  **Carnot's verify-repair improvement should be LARGER on number-swap adversarial than on
  standard GSM8K (Exp 260), because the semantic grounding extractor detects stale-answer
  errors at 100% (Exp 279) and number-swap variants generate exactly this pattern.**

Secondary hypothesis: the irrelevant-sentence variant shows SIMILAR improvement to standard GSM8K
because semantic grounding ignores content it can't anchor to quantities (0% detection of
irrelevant-sentence-induced errors, same as fresh-wrong detection).

Report: accuracy delta adversarial vs standard per mode; whether verify-repair improvement is larger
on number-swap than standard (primary criterion); comparison against Exp 260 and Exp 235.
Use `inference_mode: "live_gpu"` and `CARNOT_FORCE_LIVE=1`.

Write tests covering: artifact schema, three-mode comparison, variant-type breakdown, DualGPU dispatch.

### Exp 284: Apple adversarial results analysis and docs update

**Deliverable:** `results/experiment_284_results.json`

Analyze the Exp 282-283 results. Answer:
1. Did Apple accuracy drop replicate (≥15pp for number-swap)? If not, why?
2. Was verify-repair improvement LARGER on number-swap than standard? By how much?
3. What did the irrelevant-sentence variant show — does Carnot ignore irrelevant context as predicted?
4. Which extraction type fired (semantic grounding vs FormalClaimVerifier vs abstain)?
5. Was the dual-model comparison consistent (Qwen vs Gemma)?

Update: `docs/technical-report.md`, `docs/index.html`, `docs/technical-report.html` with an
"Adversarial Robustness" section. Update `README.md` if improvement is ≥5pp. Update
`research-studying.md` with Apple adversarial findings. Write 5+ tests covering analysis functions.

---

## Phase 92: Extraction-Free Hallucination Detection (Experiments 285-287)

The fundamental bottleneck since the simulation-vs-reality crisis has been constraint EXTRACTION.
ArithmeticExtractor regex finds 0 violations on IT models. FormalClaimVerifier abstains on 1302/2545
claims. Semantic grounding detects stale errors but misses fresh-wrong errors entirely.

Spilled Energy (arXiv 2602.18671, ICLR 2026) and the AR-EBM bijection (arXiv 2512.15605) provide
the escape: use the model's own logits as the energy signal. No extraction at all. The logits
already encode the model's uncertainty. Factually incorrect outputs have systematically higher
spilled energy because probability mass "spills" across incorrect alternatives.

### Exp 285: SpilledEnergyExtractor implementation

**Deliverable:** `python/carnot/pipeline/spilled_energy_extractor.py`

Implement `SpilledEnergyExtractor` as an additive `ConstraintExtractor` operating on saved logits
from Exp 282-283. Two complementary signals:

1. **Spilled energy (arXiv 2602.18671):** Per-token spilled energy = softmax entropy − max logit
   energy = H(softmax(logits)) − max(log_softmax(logits)). High spilled energy = uncertain output.
   Aggregate: mean, max, p95 over response tokens.

2. **Lookahead energy (arXiv 2512.15605 AR-EBM bijection):** The AR-EBM lookahead energy at token t
   is −log P(continuation | prefix), approximated as the negative log-probability of the response
   continuation under the model. Measures continuation-level constraint coherence, not just
   token-level uncertainty.

Expose as `SpilledEnergyResult` dataclass with: `per_token_spilled` (list[float]), `mean_spilled`,
`max_spilled`, `p95_spilled`, `lookahead_energy` (float), `suspected_hallucination` (bool,
threshold configurable). Add `verify_spilled_energy()` to `VerifyRepairPipeline` — additive, does
not replace existing `verify()`. Accept logits as file path or numpy array.

Write tests first: spilled energy computation from logits array, lookahead energy approximation,
threshold firing, pipeline integration, edge cases (uniform logits = 0 spill, peaked logits =
high lookahead), loading from saved Exp 282 logit files. 100% targeted module coverage.

Spec refs: REQ-VERIFY-061 (create this spec entry), SCENARIO-VERIFY-074, SCENARIO-VERIFY-075.

### Exp 286: SemanticEnergyExtractor + DualEnergyGate

**Deliverable:** `python/carnot/pipeline/semantic_energy_extractor.py`

Implement `SemanticEnergyExtractor` based on Boltzmann energy from logit distributions (arXiv
2508.14496). The semantic energy is:
  `E_semantic = -log(sum_i(exp(logit_i / T)))` where T is temperature (default 1.0).
High semantic energy = model is anomalously confident relative to context (confident-but-wrong).
The signal is COMPLEMENTARY to spilled energy: spilled catches uncertain outputs, semantic catches
overconfident-wrong outputs.

Implement `DualEnergyGate` that combines both signals: fires if EITHER exceeds its calibrated
threshold. Thresholds are calibrated from the Exp 282 logits using the Exp 219/235 accuracy labels
as ground truth for the calibration set. Apply isotonic regression calibration.

Add `verify_dual_energy()` to `VerifyRepairPipeline`. Benchmark both signals separately and combined
on the Exp 282-283 logit corpus. Report: AUROC for each signal separately and combined; at optimal
threshold: precision, recall, FP rate; which error categories each catches. Key question: does the
dual-energy gate identify cases where FormalClaimVerifier abstains?

Write tests first. 100% targeted coverage.

Spec refs: REQ-VERIFY-062, SCENARIO-VERIFY-076, SCENARIO-VERIFY-077.

### Exp 287: Dual-energy benchmark on Apple adversarial corpus

**Deliverable:** `results/experiment_287_results.json`

Retrospective benchmark applying SpilledEnergyExtractor and DualEnergyGate to the Exp 282-283
Apple adversarial logit corpus. Primary questions:

1. Does spilled energy predict adversarial errors (number-swap)? Hypothesis: YES — model
   uncertainty is higher when the numbers don't match trained patterns.
2. Does semantic energy predict irrelevant-sentence-induced errors? Hypothesis: MAYBE — model
   may be overconfident on irrelevant-sentence variants if the irrelevant context anchors it.
3. Does the dual-energy gate improve coverage over FormalClaimVerifier alone?
4. For claims where FCV abstained: does dual-energy provide a signal? (key gap-filler test)

Report: AUROC target ≥ 0.65 (primary success criterion). Compare gate precision/recall vs:
- Semantic grounding (Exp 279: 100% stale, 0% fresh-wrong)
- FormalClaimVerifier (Exp 246: arithmetic route only)
- Combined (dual-energy + FCV + semantic grounding)

Also compute: what fraction of the 1302 `not_formalizable` claims from the Exp 244 corpus
would the dual-energy gate cover at the target precision?

Write 10+ tests covering AUROC computation, threshold selection, gap-filler analysis. 100% coverage.

---

## Phase 93: FPGA Hardware Bring-up (Experiments 288-290)

The Kria KV260 has been in hand for multiple milestones and deferred each time. This phase
closes the hardware gap. Exp 228 designed the 4096-spin sparse Ising sampler and AXI-Lite
register map. Exp 242 produced a blocker artifact. The quantum-inspired sparse Ising paper
(arXiv 2604.04606) provides the convergence schedule. KANELÉ (arXiv 2512.12850) provides the
KAN LUT evaluation path for the future `carnot-kan` hardware tier. Denoising Thermodynamic
Models (arXiv 2510.23972) are noted as a potential alternative to pure Ising for FPGA.

**Hard constraint for this phase:** Every experiment has a 60s timeout. If KV260 hardware is
not available (no bitfile, no PYNQ overlay), emit a `"blocked"` artifact immediately with the
exact missing step. Do NOT stall the conductor.

### Exp 288: KV260 FPGA overlay bring-up validation

**Deliverable:** `results/experiment_288_results.json`

Attempt KV260 FPGA overlay bring-up. Procedure:
1. Check `CARNOT_KV260_BITFILE` environment variable; if unset, emit blocker immediately
2. If set, load overlay via PYNQ Python API: `from pynq import Overlay; ol = Overlay(bitfile)`
3. Verify AXI-Lite register map from Exp 228 design: write to CONTROL register, read STATUS
4. Exercise round-trip: write a minimal coupling matrix to the bias window, trigger, read back
5. Measure: overlay load latency (ms), register write/read round-trip latency (µs), whether
   sampled state is valid (all spins ∈ {+1,-1}), execution_path

Honest labeling requirements:
- `"hardware"` — PYNQ overlay loaded, AXI-Lite register map exercised, spins read back
- `"software_model"` — PYNQ load failed but register map validated in software
- `"blocked"` — no bitfile configured, with exact next steps

Do NOT fabricate hardware timing. If hardware execution_path achieved, this is a **project milestone**.

### Exp 289: FpgaBackend implementation with quantum-inspired sparse Ising

**Deliverable:** `python/carnot/samplers/fpga_backend.py`

Implement `FpgaBackend` as a concrete `SamplerBackend` (Exp 71 protocol). Design:

1. **Coupling extraction:** Load J (coupling matrix) and h (bias vector) from `IsingEBM` instance
2. **Quantization:** Convert to KV260 Q8.8 fixed-point (Exp 228 design)
3. **Sparse connectivity (arXiv 2604.04606):** Retain top-K couplings by magnitude. K chosen so
   max_degree ≤ 32 (matching Exp 228 4K-spin design). This is the "quantum-inspired" sparsification.
4. **Annealing schedule:** Implement quantum-inspired β schedule from arXiv 2604.04606:
   β(t) = β_min × (β_max/β_min)^(t/T) — log-linear warmup achieving 6× SA speedup in simulation
5. **AXI-Lite serialization:** Serialize to register map schema from Exp 228
6. **Dispatch:** If `CARNOT_KV260_BITFILE` set, send via PYNQ AXI and readback; else invoke
   `ParallelIsingSampler` (CPU) with the quantum-inspired sparse schedule as software-model fallback
7. **LagONN escape flag** (arXiv 2505.07179): optional `use_lagrangian_penalty=True` adds
   penalty term to energy for escaping infeasible local minima

Add comment in code referencing KANELÉ (arXiv 2512.12850) as future extension for KAN LUT
evaluation on the `carnot-kan` tier.

Write tests first: quantization round-trip, sparsification (max_degree ≤ 32), β-schedule monotone,
hardware/software dispatch, LagONN penalty term computation, register serialization. 100% coverage.

Spec refs: REQ-SAMPLE-008, SCENARIO-SAMPLE-015, SCENARIO-SAMPLE-016.

### Exp 290: FPGA vs CPU Ising benchmark

**Deliverable:** `results/experiment_290_results.json`

Benchmark `FpgaBackend` against `ParallelIsingSampler` (CPU baseline, 183× faster than thrml)
on three problem sizes: 100 spins, 500 spins, 1000 spins. For each size measure:
- Samples per second
- Energy convergence quality: final_energy vs ground truth energy
- Whether quantum-inspired sparse β-schedule improves convergence vs uniform β (CPU baseline)
- LagONN penalty: does it escape local minima on a known-infeasible SAT instance?

If KV260 hardware available (`execution_path: "hardware"`): report hardware latency explicitly.
Otherwise: software-model timing (`execution_path: "software_model"`), compare against Exp 228
baseline (0.824s vs CPU 0.288s for 128-spin, 16 samples).

Primary question: does the quantum-inspired sparse schedule from arXiv 2604.04606 reproduce its
claimed 6× speedup in software simulation vs dense β-schedule? This is a concrete prediction
from the paper that we can validate in software before hardware arrives.

60s timeout per benchmark configuration. Write 10+ tests. 100% coverage.

---

## Phase 94: JEPA Tier 3, NPU, HuggingFace, and Retrospective (Experiments 291-294)

### Exp 291: JEPA predictor retrained on Apple adversarial GPU data — Tier 3 self-learning

**Deliverable:** `results/experiment_291_results.json`

**This is the MANDATORY continuous self-learning experiment for milestone 2026.04.21.**

The Exp 262 JEPA calibration corpus had near-random feature importance (prefix_fraction ≈ 0.507
importance score) because it used CPU-inference token patterns. The Apple adversarial Exp 282
logits are different: (a) GPU inference is faster and more consistent; (b) adversarial variants
have higher violation density (number-swap creates predictable stale errors); (c) prefix logit
patterns from adversarial variants may be more discriminative than standard questions.

Training data: logits saved during Exp 282 at prefix fractions 25%, 50%, 75%, 100%. Features:
per-prefix spilled energy (from Exp 285), semantic energy (from Exp 286), mean/max/p95 logit
values, variant_type (number_swap vs irrelevant_sentence vs standard), and model identity.
Labels: did verify_only detect a violation? (from Exp 283 results)

Implement via EBM-CoT approach (arXiv 2511.07124): train a calibration head that refines latent
thought representations. Apply isotonic regression calibration. Apply conformal coverage bounds
(arXiv 2603.22966) to report statistically valid confidence intervals.

Target operating zone: fast-path hit rate ≥ 30% (skip Ising when gate says "probably fine"),
true-violation detection rate ≥ 60%, FP rate ≤ 20%. Run 50-case A/B on held-out questions.
State clearly if these targets are not met — honest reporting required.

Report: calibrated fast-path rate, TP/FP rates, A/B delta on held-out, conformal intervals.
Write 10+ tests. 100% targeted coverage.

Spec refs: REQ-JEPA-003, SCENARIO-JEPA-006, SCENARIO-JEPA-007.

### Exp 292: AMD XDNA NPU enablement — onnxruntime source build

**Deliverable:** `results/experiment_292_results.json`

Third and final structured attempt at AMD XDNA NPU (Exp 257: blocked, Exp 269: stalled 3 times,
Exp 281 from v26: deferred). Different approach: build onnxruntime 1.20.1 from source with
`-Donnxruntime_USE_VITISAI=ON` using VitisAI EP source in `~/github.com/amd/RyzenAI-SW/`.

Steps:
1. Install build dependencies (cmake ≥ 3.26, ninja, openblas)
2. Configure onnxruntime 1.20.1 source with VitisAI EP flag
3. Build in `.venv-npu/` (Python 3.12)
4. If build succeeds, load `results/jepa_predictor_146.onnx` with VitisAIExecutionProvider
5. Benchmark vs CPU ORT (8.6 µs Exp 257 baseline); report speedup
6. If build produces NPU result, benchmark Exp 291 JEPA model on NPU vs CPU ORT

**Hard constraint: 45-minute build timeout.** If build stalls or exceeds timeout, emit blocker
artifact immediately with: exact step reached, error log (last 50 lines), and ONE specific next
action item. Do NOT let the build stall the conductor silently.

### Exp 293: HuggingFace publishing — Exp 66 joint model and FormalClaimVerifier

**Deliverable:** `results/experiment_293_results.json`

Carry-forward from Exp 268 (SKIP'd 3 times) and Exp 282 from v26. Publish to huggingface.co/Carnot-EBM:

1. **Exp 66 differentiable constraint model** (embedding + Ising → score, 1.0 AUROC):
   - Export as `safetensors` format
   - README clearly labeled: "Phase 1 research prototype, demonstrates approach, not production quality"
   - Include training config and evaluation numbers

2. **FormalClaimVerifier** (arithmetic + comparison + cardinality routes):
   - Export arithmetic/comparison/cardinality routes as ONNX
   - Package remaining routes (boolean_entailment, set_membership) as Python bundle
   - README explaining solver routing, abstention policy, standalone usage
   - Example notebook showing integration with VerifyRepairPipeline

Tag as `v0.2.0-research`. Use `huggingface_hub` Python library. Check credentials FIRST — if
`huggingface-cli whoami` fails, emit blocker immediately with login instructions. Do NOT stall.

Write tests covering: artifact export correctness, README content validation, credential check.

### Exp 294: Operational retrospective for milestone 2026.04.21

**Deliverable:** `results/operational_retro_2026_04_21.json`

Generate process efficiency analysis. The 2026.04.20 retro showed 100% action-item carry-over rate
for the THIRD consecutive milestone. Primary audit for this retro:

**Was the carry-over rate reduced from 100%?** Specifically check:
1. Was DualGPURunner wired from Exp 281 (not added as a separate setup step)?
2. Did per-question checkpointing prevent stall losses in Exp 282-283?
3. Was the Apple adversarial benchmark completed (the #1 carry-forward since milestone 2026.04.18)?
4. Was CUDA ORT tested at batch_size ≥ 32 (Exp 259 finding: GPU faster only at batch ≥ 32)?
5. Did any retro action items from 2026.04.20 get resolved before the milestone started?

Report: total wall time, experiments/hour, GPU utilization distribution (not just milestone-end),
per-experiment DualGPU usage vs single-GPU, and updated action items for 2026.04.22 with
explicit `resolved | deferred (reason) | new` tracking. Identify the structural root cause of
the 100% carry-over rate — the current retro format generates suggestions, not tracked tickets.

---

## Phase Summary

| Phase | Experiments | Theme | Primary Success Criterion |
|-------|-------------|-------|--------------------------|
| 91 | 281-284 | Apple adversarial GSM8K | Exp 283: verify-repair Δ on number-swap > standard GSM8K Δ |
| 92 | 285-287 | Extraction-free hallucination detection | Exp 287: dual-energy AUROC ≥ 0.65 on Apple adversarial |
| 93 | 288-290 | FPGA hardware bring-up | Exp 288: non-"blocked" execution_path for KV260 |
| 94 | 291-294 | Tier 3 JEPA + NPU + HuggingFace + retro | Exp 291: fast-path ≥ 30%, TP ≥ 60%, FP ≤ 20% |

---

## Hardware Requirements

| Hardware | Experiments | Status |
|----------|-------------|--------|
| 2× RTX 3090 (CUDA) | 282, 283, 287, 291 | Available; must wire DualGPURunner from Exp 282 |
| Kria KV260 FPGA | 288, 289, 290 | Available (in hand); `CARNOT_KV260_BITFILE` must be configured |
| AMD XDNA NPU | 292 | VitisAI EP `.so` present; needs onnxruntime source build |
| HuggingFace CLI + credentials | 293 | Check with `huggingface-cli whoami` at milestone start |

---

## Dependency Graph

```
Exp 281 (Apple adversarial dataset generator)         [CPU only, independent]
    └── Exp 282 (GPU baseline, SAVE LOGITS)
              └── Exp 283 (Apple adversarial + verify-repair, SAVE LOGITS)
                        └── Exp 284 (analysis + docs update)
                        └── Exp 287 (dual-energy benchmark — uses Exp 282-283 logits)
                        └── Exp 291 (JEPA training — uses Exp 282-283 logits)
Exp 285 (SpilledEnergyExtractor)
    └── Exp 286 (SemanticEnergyExtractor + DualEnergyGate)
              └── Exp 287 (dual-energy benchmark)
              └── Exp 291 (JEPA features)
Exp 288 (KV260 bring-up)
    └── Exp 289 (FpgaBackend implementation)
              └── Exp 290 (FPGA vs CPU benchmark)
Exp 292 (AMD XDNA NPU)                               [independent, 45min timeout]
Exp 293 (HuggingFace publish)                        [independent]
Exp 294 (retro)                                      [depends on all prior]
```

---

## What This Milestone Does NOT Include

- **FactNetExtractor full implementation** — FactNet's 1.7B triples require significant storage.
  Defer to dedicated factual verification milestone. Exp 276 (Wikidata extractor) covers this for now.
- **PredictiveVerifier CUDA ORT batched benchmark** — Deferred from prior milestone. Include in
  2026.04.22 only if JEPA (Exp 291) proves useful and batched inference becomes the bottleneck.
- **DOMINO grammar-constrained generation** — Long-term path. Depends on completing Apple adversarial
  benchmark first.
- **LoRA continual learning for constraint model** — (arXiv 2504.13407) Valuable but lower priority.
  File in research-references.md for future milestone.
- **Πnet orthogonal projection repair** — (arXiv 2508.10480) Better principled than Langevin repair.
  File in research-references.md for repair pipeline milestone.

---

## Why This Ordering

Phases 91 → 92 → 93 → 94 follow data dependencies:
- Phase 91 runs first because it generates the GPU logits (Exp 282-283) that Phases 92 and 94 depend on
- Phase 92 builds the extraction-free signals AFTER logits exist so it can benchmark immediately
- Phase 93 (FPGA) is independent and can run in parallel with Phase 92
- Phase 94 runs last: JEPA needs Phase 91+92 data; NPU and HuggingFace are independent; retro is last

The ordering within phases respects the conductor's sequential task execution model.
