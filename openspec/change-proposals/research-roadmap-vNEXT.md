# Carnot Research Roadmap: Milestone 2026.05.215

**Theme:** Live GPU Verification Pivot, FPGA Rescope, and Tier 3 Predictive Self-Learning

## 1. Status at End of 2026.05.214

Milestone .214 proved the feasibility of Thermodynamically Constrained Neural Generation, integrated substrate-aware KANs in simulation, and established early dynamic resolution for Continuous Self-Learning (CSL). However, as mandated by the `research-program.md` April 2026 findings:
1. Regex-based constraint extraction is fundamentally broken for instruction-tuned models. We need NSVIF/Z3 or LLM-as-extractor techniques.
2. The architectural derivation proved that $k=15$ is infeasible on a single-chain FPGA. We must pivot to $k_{max}=5$ with Manifold Substitution.
3. CSL must advance to Tier 3 (JEPA Predictive Verification) coupled with recent findings on Adaptive Data Harvesting for universal constraints (arxiv:2605.09707).

## 2. Goals for 2026.05.215

1. **Rebuild Constraint Extraction:** Implement a Z3/NSVIF extractor and LLM-as-extractor for reliable verification on live instruction-tuned output.
2. **Hardware Realism ($k_{max}=5$):** Refactor the FPGA topology toward parallel PT-SB chains capped at $k=5$, satisfying the KV260 deployment constraints.
3. **Continuous Self-Learning (Tier 3):** Implement predictive verification using JEPA and adaptive data harvesting to forecast violations before autoregressive generation finishes.
4. **Live E2E Benchmarks:** Execute full GSM8K/HumanEval validation on live GPU utilizing the newly mandated unsloth SOTA GGUF models.

## 3. Architecture

```mermaid
graph TD
    A[SOTA GGUF (Live GPU)] -->|CoT Response| B[LLM Extractor / Z3 Formalizer]
    B -->|Verified Constraints| C[Energy Function (k_max=5 KAN)]
    C -->|Violations| D[Tier 3 JEPA Predictor]
    D -->|Continuous Training| E[Adaptive Data Harvester]
    C -->|Feedback| A
```

## 4. Execution Phases

### Phase 0: Housekeeping
Archive the .214 roadmap and initialize .215 execution tracking.

### Phase 1: Robust Extraction
Retire the regex-based ArithmeticExtractor. Build an LLM-guided extractor that outputs formal logic (Z3/NSVIF) representations of CoT steps for reliable constraint detection.

### Phase 2: FPGA Rescope
Pivot the KV260 simulator and architecture to the $k_{max}=5$ parallel PT-SB chain configuration, abandoning the monolithic $k=15$ approach. 

### Phase 3: Tier 3 Self-Learning
Integrate arxiv:2605.09707 (Adaptive Data Harvesting) to construct a JEPA predictive verification model. It will train on live GPU logs to forecast energy spikes early in the LLM generation window.

### Phase 4: Live GPU Capstone
Run a complete End-to-End evaluation of the pipeline (Z3 extraction + JEPA prediction) on `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF`. Conclude with the milestone retrospective.

## 5. Hardware Requirements
- **Development/Training:** Local dual-RTX 3090 system (48GB VRAM) for live GGUF inference.
- **Hardware Integration:** Target KV260 FPGA boards using mock simulation until physical boards are deployed.
