# Research Studying — Ranked Ideas for Future Experiments

**Purpose:** Claude (outer loop) continuously researches novel ideas from
online sources, ranks them by potential impact on Carnot's current state,
and queues the most promising into the next roadmap milestone. Codex (inner
loop) executes the current experiments.

**Updated:** 2026-04-12
**Current Focus:** Precision ceiling on larger models (Exp 184: -2% on 3B) + constraint extraction on instruction-tuned models (Exp 210)

## How This Works

1. Claude searches arxiv, OpenReview, GitHub, Extropic, Semantic Scholar, HN
2. Each finding is ranked by: relevance × novelty × feasibility × urgency
3. Top ideas are promoted to `research-roadmap-next.yaml` when a slot opens
4. Lower-ranked ideas stay here for future consideration
5. Ideas that prove irrelevant are moved to "Archived"

## Ranking Criteria

- **Relevance (1-5):** How directly does this apply to Carnot's current gaps?
- **Novelty (1-5):** Is this a new approach we haven't tried?
- **Feasibility (1-5):** Can we implement this in 1-2 experiments?
- **Urgency (1-5):** Does our current research depend on this?
- **Score = R × N × F × U** (max 625)

## Active Research Queue (Ranked)

### URGENT Rank 0: Live vs Simulated Inference Validation
- **Score:** 5×5×5×5 = **625** (MAXIMUM)
- **Source:** Internal finding — ALL positive results were simulated inference
- **Crisis:** Exp 184 is the FIRST live GPU experiment and shows -2% standard,
  -12% adversarial on 3B model. But ALL previous positive results (Exp 91,
  120, 121, 161, 162) used SIMULATED inference. We cannot distinguish whether
  the negative result is model-size (precision ceiling) or inference-mode
  (simulation was unrealistically favorable).
- **MUST DO IMMEDIATELY:** Run 0.8B Qwen3.5 with LIVE GPU inference on the
  SAME GSM8K questions. If 0.8B live shows +10-14%, precision ceiling is real
  and we fix it with Z3/confidence. If 0.8B live shows ~0%, our ENTIRE results
  narrative is based on simulation artifacts and we have a fundamental problem.
- **Status:** INVESTIGATED — result is CONFIRMED ARTIFACT
- **Finding:** Live 0.8B inference on GSM8K produces identical wrong answers
  as the checkpoint (Q0=182, Q1=3, Q2=120000). The model scores ~25% on
  GSM8K — the simulated inference assumed ~65-70% (instruction-tuned level).
  ALL positive improvement numbers were measured against fake baselines.
- **Root cause:** Simulated inference was calibrated to published benchmarks
  for instruction-tuned models, but we loaded the BASE model (Qwen3.5-0.8B,
  not an instruct variant). The base model's actual GSM8K score is ~25%.
- **Impact:** The core +10-28% improvement claim is based on simulation
  artifacts. Real live inference shows 0% improvement at both 0.8B and 3B.
- **Path forward:** Either (a) use instruction-tuned models, (b) improve
  prompt engineering for base models, or (c) acknowledge constraint
  verification helps simulated/ideal scenarios but not raw base model outputs.
- **Why #0:** This is the most important finding of the entire project.

### Rank 1: Confidence-Calibrated Constraint Verification
- **Score:** 5×4×5×5 = 500
- **Source:** Internal finding (Exp 184: 3B model -2% regression)
- **Idea:** Weight constraint violations by confidence level. High-confidence
  violations (exact arithmetic mismatch) get repaired; low-confidence
  (approximate values, intermediate steps) get logged but not repaired.
  This directly addresses the precision ceiling where FP > TP on larger models.
- **Status:** Already in roadmap as Exp 202. Highest priority.
- **Why #1:** Without this, Carnot's value proposition shrinks as models improve.

### Rank 2: Semantic Constraint Verification via Chain-of-Thought Decomposition
- **Score:** 5×5×3×5 = 375
- **Source:** Exp 184 error analysis — larger models make semantic errors, not arithmetic
- **Idea:** Decompose chain-of-thought into logical steps, verify each step's
  LOGIC (not just arithmetic). "If A then B, A is true, therefore B" can be
  checked structurally. Apply the global consistency checker (Exp 172, 100%
  detection) to single-response multi-step reasoning.
- **Status:** Noted in research-program.md, not yet in roadmap
- **Why #2:** Addresses the 67% of errors that are currently uncatchable

### Rank 3: Speculative Decoding with Constraint Pre-Filtering
- **Score:** 4×5×3×4 = 240
- **Source:** Speculative decoding literature + our guided decoding (0.006ms)
- **Idea:** Use a small draft model to generate candidate tokens, then
  verify each candidate's constraint energy BEFORE the large model commits.
  Like speculative decoding but with constraint energy as the accept/reject
  criterion instead of probability matching.
- **Status:** Not in roadmap. Needs research.
- **Why #3:** Combines two proven techniques (spec decoding + constraint energy)

### Rank 4: Contrastive Constraint Learning from Model Errors
- **Score:** 4×4×4×4 = 256
- **Source:** Exp 184 data — we now have (correct, incorrect) pairs from a 3B model
- **Idea:** Train constraint extractors on the SPECIFIC error patterns of each
  model size. Instead of one-size-fits-all ArithmeticExtractor, learn what
  the 3B model gets wrong vs right and build model-specific constraints.
  The self-learning tracker (Exp 132) already accumulates this data.
- **Status:** Partially addressed by Exp 201 (precision curve)
- **Why #4:** Makes the constraint system model-adaptive

### Rank 5: FPGA Ising Sampler with Real-Time Coupling Updates
- **Score:** 3×5×3×3 = 135
- **Source:** Kria KV260 arriving in 4 days + research-hardware-wishlist.md
- **Idea:** Implement a 4K p-bit Ising sampler in Verilog with AXI-Lite
  interface for real-time coupling updates. The coupling matrix is
  reprogrammed for each constraint verification, not fixed at synthesis.
  This enables dynamic constraint checking at hardware speed.
- **Status:** Hardware ordered. Needs Verilog implementation.
- **Why #5:** Validates the TSU hardware path

### Rank 6: Energy-Aware Beam Search
- **Score:** 4×4×3×3 = 144
- **Source:** Guided decoding (Exp 110) + beam search literature
- **Idea:** Modify beam search to include constraint energy in the beam score.
  Standard beam search: score = log_prob. Energy beam search:
  score = log_prob - alpha * constraint_energy. This naturally steers
  generation toward constraint-satisfying sequences without post-hoc repair.
- **Status:** Not in roadmap
- **Why #6:** Principled integration of energy into generation

### Rank 7: Hierarchical Constraint Composition for Complex Reasoning
- **Score:** 3×4×3×4 = 144
- **Source:** Exp 63 (hierarchical Ising) + Exp 172 (global consistency)
- **Idea:** Compose constraints hierarchically: word-level (arithmetic),
  sentence-level (logic), paragraph-level (consistency), document-level
  (factual). Each level feeds violations to the next. This mirrors how
  human reasoning catches errors at multiple scales.
- **Status:** Partially explored (Exp 63, 172, 176)
- **Why #7:** Framework for scaling verification to complex documents

### Rank 8: Differentiable Constraint Compilation to Hardware
- **Score:** 3×5×2×3 = 90
- **Source:** Exp 66 (differentiable constraints) + FPGA path
- **Idea:** Compile differentiable KAN constraints directly to FPGA lookup
  tables. The spline knots become LUT entries. Training updates the LUT
  contents without FPGA resynthesis. This is the bridge between Tier 4
  adaptive structure and hardware acceleration.
- **Status:** Long-term, needs FPGA first
- **Why #8:** The eventual production architecture

## New Findings from Study Run (2026-04-11)

### NSVIF: Neuro-Symbolic Verification via First-Order Logic (HIGH RELEVANCE)
- **Source:** [arxiv 2601.17789](https://arxiv.org/html/2601.17789v1)
- **What:** Formalizes instruction verification as a CSP — extracts constraints
  from instructions, converts to first-order logic, solves with Z3 SMT solver.
- **Relevance to precision ceiling:** This is EXACTLY what we need for larger
  models. Instead of pattern-matching arithmetic (ArithmeticExtractor), formalize
  the constraints as FOL and use an SMT solver. FOL constraints have NO false
  positives — they're either satisfied or not. This could eliminate the FP
  problem on 3B+ models entirely.
- **Score:** 5×5×4×5 = **500** — ties with Rank 1
- **Action:** Promote to roadmap. Replace ArithmeticExtractor's regex with
  Z3 SMT solving for arithmetic constraints. Keep regex as fast path, Z3
  as verification backend.

### ConstraintLLM: Neuro-Symbolic for Industrial Scheduling
- **Source:** [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.809.pdf)
- **What:** Neuro-symbolic framework combining LLMs with constraint solvers
  for industrial scheduling. LLM generates constraint specifications, solver
  verifies feasibility.
- **Relevance:** Directly applicable to our scheduling domain (Exp 44, LagONN).
  Could improve scheduling constraint extraction.
- **Score:** 4×4×4×3 = 192

### FPGA P-Bit Cluster: 6400 Spins, 64 Billion Flips/Second
- **Source:** [arxiv 2512.24558](https://arxiv.org/html/2512.24558) + 
  [Nature Electronics](https://www.nature.com/articles/s41928-024-01182-4)
- **What:** Multi-FPGA cluster implementing sparse Boltzmann machines with
  p-bits. Achieved 6400 spins (80×80 Ising) on FPGA, 50-64 billion
  probabilistic flips/second. CD training with up to n=10M sweeps.
- **Relevance:** Our KV260 (arriving in 4 days) has 256K LUTs — enough for
  ~4K p-bits. This paper provides the implementation reference: sparse
  connectivity, local parallel updates, low-precision arithmetic.
  Key detail: they use CD-n with n=10M sweeps per update, far more than
  our CD-1 or CD-5. Worth testing higher-n CD on our learned Ising models.
- **Score:** 4×4×5×3 = 240 — promotes above energy-aware beam search
- **Action:** Use as implementation reference for KV260 Ising sampler.
  Add high-n CD experiment to roadmap.

### Speculative Speculative Decoding (ICLR 2026)
- **Source:** [ICLR 2026](https://openreview.net/pdf?id=aL1Wnml9Ef)
- **What:** Meta-speculation — speculate the NEXT round during current
  verification. Amortizes verification cost across rounds.
- **Relevance:** If we combine with constraint energy, the draft model
  generates candidates, constraint energy pre-filters, and the target
  model verifies. Three-level pipeline. But complex to implement.
- **Score:** 3×5×2×3 = 90

### KAN Computing-in-Memory (Nature Communications 2026)
- **Source:** [Nature Comms](https://www.nature.com/articles/s41467-026-69592-w)
- **What:** Hardware implementation of KAN using tunable Gaussian-like
  memory cells. Spline activations implemented as analog memory lookups.
- **Relevance:** Validates our Tier 4 vision (KAN → hardware). Not directly
  actionable until we have the right hardware, but confirms the path.
- **Score:** 3×5×2×2 = 60

### Agentic Confidence Calibration (2026)
- **Source:** [arxiv 2601.15778](https://arxiv.org/html/2601.15778v1)
- **What:** Holistic Trajectory Calibration — extracts process-level features
  across an agent's entire trajectory to calibrate confidence.
- **Relevance:** Directly applicable to our multi-turn agentic verification.
  Instead of per-step constraint checking, calibrate confidence across the
  whole reasoning trajectory. Could improve the global consistency checker.
- **Score:** 4×4×3×4 = 192

## Updated Rankings After Study Run

| Rank | Idea | Score | Status |
|------|------|-------|--------|
| 1 | NSVIF: FOL + Z3 SMT constraint verification | **500** | NEW — promote to roadmap |
| 1 | Confidence-calibrated constraints | 500 | In roadmap (Exp 202) |
| 3 | Semantic constraint via CoT decomposition | 375 | Noted |
| 4 | Contrastive constraint learning | 256 | Partially in Exp 201 |
| 5 | FPGA p-bit cluster (implementation ref) | **240** | NEW — use for KV260 |
| 6 | Speculative decoding with constraints | 240 | Needs research |
| 7 | ConstraintLLM industrial scheduling | **192** | NEW |
| 7 | Agentic confidence calibration | **192** | NEW |
| 9 | Energy-aware beam search | 144 | Noted |
| 9 | Hierarchical constraint composition | 144 | Partially explored |
| 11 | FPGA Ising real-time updates | 135 | KV260 arriving |
| 12 | Speculative speculative decoding | **90** | NEW — complex |
| 12 | Differentiable constraint compilation | 90 | Long-term |
| 14 | KAN computing-in-memory | **60** | NEW — validates path |

### Kona 1.0 Architecture Details (STRATEGIC INTELLIGENCE)
- **Source:** [logicalintelligence.com](https://logicalintelligence.com/kona-ebms-energy-based-models),
  [BusinessWire Jan 2026](https://www.businesswire.com/news/home/20260120751310)
- **What:** Kona 1.0 is now in pilot programs. Key architectural details:
  - **Non-autoregressive at trace level** — generates complete reasoning traces
    simultaneously (not token-by-token)
  - **Continuous latent space** — outputs dense vector tokens, not discrete
  - **Self-correcting** — learns by recognizing and correcting own mistakes
  - **96.2% Sudoku** in 313ms (vs LLMs at 2%)
  - Yann LeCun added to leadership (validates EBM direction)
  - Pilot sectors: energy, manufacturing, semiconductors
- **Relevance:** This is our North Star competitor. Key differences from Carnot:
  - Kona generates reasoning; Carnot verifies LLM reasoning
  - Kona is non-autoregressive; Carnot works with autoregressive LLMs
  - Kona operates in continuous latent space; we're bridging to it (Exp 64-66)
  - The self-correcting aspect is what our verify-repair loop does externally
- **Implications for our precision ceiling:** Kona's continuous latent space
  may not have the FP problem because it doesn't use discrete constraint
  matching. Our Z3 SMT approach (NSVIF) is the bridge.
- **Score:** Strategic intelligence, not directly actionable. Monitor.

### Extropic Z1 Timeline Update
- **Source:** [extropic.ai/hardware](https://extropic.ai/hardware)
- **What:** Z1 chip (hundreds of thousands of p-bits per chip, millions per
  card) scheduled for early access 2026. XTR-0 testing platform was Q3 2025.
  Mass-manufacturable using standard CMOS.
- **Relevance:** Our KV260 FPGA (arriving in 4 days) is the bridge. If Z1
  early access opens, we have the SamplerBackend abstraction (Exp 71) ready
  to plug in. Our FPGA work validates the architecture before Z1 ships.
- **Score:** 3×3×2×3 = 54 — monitor, hardware path validated

### "Hallucination is Inevitable" (HuggingFace trending)
- **Source:** [huggingface.co/papers/2401.11817](https://huggingface.co/papers/2401.11817)
- **What:** Formal proof that LLMs inherently hallucinate — cannot learn all
  computable functions. Hallucination is a mathematical inevitability.
- **Relevance:** VALIDATES our entire approach. If hallucination can't be
  eliminated from INSIDE the model, external verification (Carnot) is the
  only path. This is the theoretical justification for our product.
- **Score:** 5×1×1×5 = 25 — not actionable but validates our thesis

## Libraries of Reference (Consulted During Study Runs)

Study runs check ALL of these sources:
1. **arxiv.org** — primary research papers
2. **OpenReview.net** — NeurIPS/ICML/ICLR submissions
3. **extropic.ai/writing** — TSU hardware updates
4. **Semantic Scholar** — citation tracking for key papers
5. **HuggingFace papers** (huggingface.co/papers) — daily ML paper feed
6. **GitHub trending** — new repos (ising-model, energy-based-model topics)
7. **logicalintelligence.com** — Kona architecture updates
8. **FPGA conferences** (FCCM, FPL, DAC) — Ising machine implementations
9. **AMD developer forums** — NPU/XDNA updates
10. **Nature Electronics/Communications** — hardware implementations
11. **ACL Anthology** — NLP constraint/verification papers

## Needs Investigation (Unranked)

- LagONN + guided decoding combination (oscillatory escape + energy steering)
- Multi-agent constraint verification (one agent generates, another verifies)
- Retrieval-augmented constraints (look up facts before verifying)
- Constraint transfer learning (train on one domain, apply to another)
- Grammar-constrained decoding as constraint substitute (ACL 2025 finding)
- Block verification for speculative decoding (5-8% speedup, OpenReview)
- Physics-informed KAN with augmented Lagrangian (Nature 2025)

## Archived (Investigated, Not Promising)

- LNN adaptive couplings within chains: -90% vs static Ising (Exp 116)
- Precision-based constraint reweighting: 0% improvement (Exp 134)
- Activation-based EBMs: detect confidence not correctness (14 principles)

<!-- EXP210_STUDYING_START -->
## Study Run 2026-04-12 - Constraint Extraction for Instruction-Tuned Models

### Ranking update
| Rank | Idea | Score | Why it matters |
|------|------|-------|----------------|
| 1 | Prompt-to-constraint intermediate representation with solver fallback | 625 | NSVIF, DeCRIM, and ConstraintLLM all point to the same fix: extract atomic constraints from the instruction before verifying the answer. |
| 2 | Benchmark-first extraction workbench | 500 | FollowBench, CFBench, RealInstruct, and VIFBench provide the missing datasets needed to measure extraction recall and false positives directly. |
| 3 | Dual-path verification: prompt-answer first, CoT second | 500 | CoT verification is promising, but monitorability papers say Carnot should never depend on raw CoT alone. |
| 4 | Typed step-graph verification for arithmetic and logic traces | 375 | VeriCoT, PCRLLM, Deductive Verification, and Typed CoT all support moving from free-form traces to explicit premises and rules. |
| 5 | Constraint-programming route for scheduling and resource tasks | 240 | ConstraintLLM plus IndusCP is the best external path for Carnot's scheduling extractor gap. |
| 6 | CoT monitorability score and fallback policy | 240 | Recent monitorability work implies Carnot needs a gate deciding when CoT evidence is safe to trust. |

### Key takeaways
- The strongest direct fit is prompt-side instruction verification: convert instructions into atomic constraints first, then verify the answer against them.
- Step-level CoT verification is now technically credible, but only when reasoning traces are reformatted into explicit premises, rules, and typed steps.
- Benchmark coverage for fine-grained instruction constraints is finally good enough to evaluate extraction quality directly instead of using answer accuracy as a proxy.
- Recent monitorability papers make raw chain-of-thought an unsafe sole source of truth; Carnot needs a fallback path that does not trust CoT by default.

### Proposed experiments for 2026-04-15
- **EXP-211 - Instruction-to-Constraint IR Benchmark**
  Goal: Build a gold benchmark of atomic prompt constraints from FollowBench, RealInstruct, CFBench, and VIFBench, then measure extraction recall and false positives on instruction-tuned models.
  Hypothesis: Prompt-side decomposition will reduce false positives more than answer-only regex extraction because the verifier will know exactly which constraints matter before inspecting the response.
  Success criteria: Atomic constraint recall >= 0.85 on the curated benchmark, satisfied-constraint false-positive rate <= 0.05, and measurable improvement over the current regex plus Z3 promptless path.
- **EXP-212 - Dual-Path CoT Verifier with Typed Step Graphs**
  Goal: Implement a step-level verifier for arithmetic and logic traces using premise-rule-conclusion records inspired by VeriCoT, PCRLLM, Deductive Verification, and Typed CoT.
  Hypothesis: A typed step graph will catch errors that answer-only checking misses, but only when combined with prompt-derived constraints and a fallback to answer-level verification.
  Success criteria: On a live instruction-tuned cohort, catch >= 25% of wrong answers missed by prompt-only verification while adding < 2% extra false positives on correct answers.
- **EXP-213 - CoT Monitorability Audit and Fallback Policy**
  Goal: Measure whether Qwen and Gemma instruction-tuned models expose enough faithful reasoning to justify CoT-based extraction, using recent faithfulness and pathology metrics.
  Hypothesis: Monitorability differs by model family and task, so Carnot should gate CoT extraction behind a measured trust score rather than assuming traces are faithful.
  Success criteria: Produce a per-model monitorability score, a pathology breakdown, and a simple policy that predicts when to trust CoT extraction versus prompt-answer-only verification.
<!-- EXP210_STUDYING_END -->

## Study Run 2026-04-12 — Post-Milestone 2026.04.14 + Early 2026.04.15

**Updated:** 2026-04-12
**Current Focus:** Semantic grounding gap (0/9 wrong answers detected on live GSM8K)

### New Findings

#### Property-Generated Solver (HIGH IMPACT — code verification)
- **Source:** [arxiv 2506.18315](https://arxiv.org/abs/2506.18315)
- **What:** Uses property-based testing to validate LLM-generated code. Properties
  are simpler to define than exhaustive test oracles. **23-37% pass@1 improvement.**
- **Relevance:** Directly applicable to Exp 217 (property code verifier) and our
  HumanEval pipeline. Could multiply the +3.3pp we got in Exp 208.
- **Score:** 5×5×5×5 = **625** — MAXIMUM. Implement immediately.
- **Action:** Integrate PBT into CodeExtractor for Exp 217/220.

#### Eidoku: Neuro-Symbolic Verification Gate
- **Source:** [arxiv 2512.20664](https://arxiv.org/pdf/2512.20664)
- **What:** Deterministic rejection gate for LLM reasoning hallucinations.
  Neuro-symbolic sanity check that gates generative output.
- **Relevance:** Exactly what our verify-repair pipeline does. Validate our
  architecture against their design patterns.
- **Score:** 5×4×4×4 = 320

#### Neuro-Symbolic Compliance (LLM + SMT for Finance)
- **Source:** [arxiv 2601.06181](https://arxiv.org/html/2601.06181v1)
- **What:** LLM interprets regulations → generates SMT constraints → solver
  enforces consistency. 86.2% SMT code gen accuracy, 100x reasoning speedup.
- **Relevance:** Same pattern as our Z3 extractor but for legal/financial domain.
  Validates LLM-as-SMT-generator approach.
- **Score:** 4×4×4×3 = 192

#### SCoRe: Multi-Turn RL Self-Correction (ICLR 2025)
- **Source:** ICLR 2025 SuperCorrect
- **What:** Multi-turn RL teaches LLMs to self-correct. +15.6% MATH, +9.1% HumanEval.
- **Relevance:** Our verify-repair loop is external self-correction. SCoRe shows
  internal self-correction can complement it. Could inform repair prompting.
- **Score:** 4×4×3×4 = 192

#### Learning to Self-Verify (CRITICAL INSIGHT)
- **Source:** [arxiv 2602.07594](https://arxiv.org/html/2602.07594v1)
- **What:** Self-verification doesn't improve with model scale. Needs explicit
  training. Generation and verification are asymmetric capabilities.
- **Relevance:** Validates Carnot's external verification approach. LLMs can't
  self-verify — they need us.
- **Score:** 5×3×1×5 = 75 — not actionable but validates thesis

#### Thought Anchors (NeurIPS 2025 Workshop)
- **Source:** [OpenReview](https://openreview.net/forum?id=VnSlfeRCaU)
- **What:** Identifies which CoT reasoning steps have outsized impact on final
  answers. Some steps are "anchors" that determine the trajectory.
- **Relevance:** Could improve our CoT monitorability audit (Exp 213) — focus
  verification on anchor steps, not all steps.
- **Score:** 4×5×3×4 = 240

#### Scientific Knowledge-Driven Decoding Constraints
- **Source:** [arxiv 2604.06603](https://arxiv.org/html/2604.06603)
- **What:** Hard constraints combined with LLM distributions during decoding
  without interfering with normal reasoning.
- **Relevance:** Directly applicable to our guided decoding (Exp 110). Better
  constraint integration method.
- **Score:** 4×4×3×3 = 144

### Updated Rankings After 2026-04-12 Study Run

| Rank | Idea | Score | Status |
|------|------|-------|--------|
| 1 | **Property-Based Testing for code verification** | **625** | NEW — integrate into Exp 217/220 |
| 1 | Prompt-to-constraint IR with solver fallback | 625 | In progress (Exp 211-212) |
| 3 | Confidence-calibrated constraints | 500 | Deferred |
| 4 | Semantic constraint via CoT decomposition | 375 | In progress (Exp 215-216) |
| 5 | Eidoku verification gate pattern | **320** | NEW — architecture validation |
| 6 | Contrastive constraint learning | 256 | Partially explored |
| 7 | Thought Anchors for CoT focus | **240** | NEW — improve Exp 213 |
| 8 | FPGA p-bit cluster | 240 | KV260 arriving soon |
| 9 | Neuro-Symbolic Compliance (SMT) | **192** | NEW — validates Z3 approach |
| 9 | SCoRe self-correction | **192** | NEW — inform repair prompting |
| 11 | ConstraintLLM scheduling | 192 | Noted |
| 12 | Energy-aware beam search | 144 | Noted |
| 12 | Scientific decoding constraints | **144** | NEW — guided decoding |
| 14 | FPGA Ising real-time updates | 135 | KV260 arriving |

### Implications for Milestone 2026.04.16

The Property-Generated Solver finding is transformative for code verification.
Our HumanEval result (+3.3pp) used only basic execution testing. PBT showed
23-37% improvement on similar benchmarks — we should expect a much larger delta
if we integrate property-based testing into our CodeExtractor + repair loop.

**Proposed milestone 2026.04.16 theme: "Scale What Works"**
1. Scale code verification with PBT (our strongest live result)
2. FPGA Ising prototype (KV260 should have arrived)
3. Full 164-problem HumanEval with PBT + repair (publishable result)
4. Multi-model code verification (Qwen + Gemma + larger models)
5. Self-learning from code verification traces (Tier 1-2)
6. Bridge to production: package the code verification pipeline
