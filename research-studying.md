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

## Needs Investigation (Unranked)

- LagONN + guided decoding combination (oscillatory escape + energy steering)
- Multi-agent constraint verification (one agent generates, another verifies)
- Retrieval-augmented constraints (look up facts before verifying)
- Constraint transfer learning (train on one domain, apply to another)

## Archived (Investigated, Not Promising)

- LNN adaptive couplings within chains: -90% vs static Ising (Exp 116)
- Precision-based constraint reweighting: 0% improvement (Exp 134)
- Activation-based EBMs: detect confidence not correctness (14 principles)
