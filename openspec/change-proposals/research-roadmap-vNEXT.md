# Milestone 2026.05.177: Neuro-Symbolic Extraction, Compositional Energy, and KAN Hardware

## 1. Context and Outcomes of Milestone 2026.05.176
Milestone .176 resolved the legacy audit gaps and integrated Phase 4 carry-forwards, setting the stage for broader structural alignment and robust extraction. 
The current biggest gaps between our state and the PRD vision are:
1. **Extraction Gap:** We lack a reliable constraint-aware retrieval module (CARM) to accurately bridge unstructured instructions to our deterministic executable validators.
2. **Compositional Reasoning:** Complex multi-constraint optimization still degrades. We need Compositional Energy Minimization (CEM) to decompose and aggregate energy landscapes.
3. **Continuous Self-Learning:** Retained policy updates require rigorous non-forgetting checks (zero soundness mistakes) during continuous feedback loops.

## 2. Phase Architecture

### Phase 1: Constraint-Aware Retrieval & Extraction (CARE)
We will adapt the ConstraintLLM principles to extract verifier constraints directly from complex prompts.
- **Goal:** Improve parser yield and constraint recall over raw LLM drafting.
- **Method:** Integrate a Constraint-Aware Retrieval Module that fetches known verifiable properties before schema generation.

### Phase 2: Compositional Energy Minimization (CEM)
Instead of forcing the LLM to resolve all constraints simultaneously, decompose constraints into local energy landscapes.
- **Goal:** Ensure stable multi-constraint resolution.
- **Method:** Implement an iterative parallel energy minimization step over separated constraints.

### Phase 3: Continuous Self-Learning Pipeline
Deploy an online learning workflow that evaluates skill promotion using zero soundness mistakes as an absolute gate.
- **Goal:** Retain capabilities without regression.
- **Method:** Policy replay verification under rigorous asymmetric-cost utility scoring.

### Phase 4: KAN Hardware Substrate Blueprinting
Prepare the KAN energy tiers for FPGA deployment by moving to LUT-friendly representations.
- **Goal:** Validate KAN scalability for KV260 execution without synthesis.
- **Method:** Implement KANELÉ-style Look-Up Table evaluations and measure BOPs/NABS.

## 3. Dependency Graph
- Phase 1 (Extraction) → Phase 2 (Compositional)
- Phase 2 (Compositional) → Phase 3 (Continuous Learning)
- Phase 4 (Hardware) runs orthogonally.

## 4. Hardware Requirements
- **Local:** Dual RTX 3090 (Mandated SOTA GGUF inference)
- **Substrate:** CPU/Python simulation for KAN evaluation. No direct board claims.