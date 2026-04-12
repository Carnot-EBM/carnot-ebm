# Research Studying — Ranked Ideas for Future Experiments

**Purpose:** Claude (outer loop) continuously researches novel ideas from
online sources, ranks them by potential impact on Carnot's current state,
and queues the most promising into the next roadmap milestone. Codex (inner
loop) executes the current experiments.

**Updated:** 2026-04-11
**Current Focus:** Precision ceiling on larger models (Exp 184: -2% on 3B)

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
- **Status:** NOT IN ROADMAP — must add immediately
- **Why #0:** This determines whether Carnot's core claim is real or an artifact.

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
