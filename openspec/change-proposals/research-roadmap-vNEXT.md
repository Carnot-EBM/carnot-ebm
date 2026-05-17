# Carnot Research Roadmap: Milestone 2026.05.216

**Theme:** FPGA Synthesis, KAN Continual Learning (KAN-CL), and Energy-Guided Decoding

## 1. Status at End of 2026.05.215

Milestone .215 effectively pivoted the KV260 hardware path towards $k_{max}=5$ (Manifold Substitution), brought up Z3/NSVIF as our primary semantic constraint extractor on instruction-tuned models, and instantiated a JEPA-based predictor for Tier 3 Continuous Self-Learning (CSL). However:
1. The FPGA $k_{max}=5$ models remain in simulation. We need Verilog RTL synthesis via OSS-CAD-Suite to prove hardware feasibility.
2. The JEPA predictor suffers from Catastrophic Forgetting when learning new constraint paradigms. We must adopt KAN-CL (per-knot importance regularization) to fix this.
3. The Verify-and-Repair loop treats all tokens equally. We must fix the "Action Bottleneck" (arxiv:2605.14558) via energy-weighted gradient redistribution (ActFocus) so our predictor accurately assigns blame.
4. Energy-Guided Decoding needs to be fully instantiated on the GGUF runtime to mitigate hallucination dynamically.

## 2. Goals for 2026.05.216

1. **KAN Continual Learning (KAN-CL):** Implement per-knot importance regularization in our KAN energy tiers to completely eliminate catastrophic forgetting during continuous self-learning.
2. **Resolve Action Bottleneck:** Implement ActFocus token reweighting. Ensure that constraint violation feedback strongly penalizes the specific 'action' tokens that violated the constraints, rather than distributing loss evenly over reasoning tokens.
3. **KV260 RTL Synthesis:** Finalize the $k_{max}=5$ Verilog design and run it through `yosys` via OSS-CAD-Suite to emit a real hardware bitstream, completing the FPGA bring-up.
4. **Energy-Guided Decoding:** Apply Energy-Guided Decoding strategies directly on live GGUF `unsloth` models.

## 3. Architecture

```mermaid
graph TD
    A[SOTA GGUF (Live GPU)] -->|Energy-Guided Decoding| B[LLM Extractor / Z3 Formalizer]
    B -->|Constraints| C[KAN Energy Function + KAN-CL]
    C -->|ActFocus Weighting| D[Tier 3 JEPA Predictor]
    D -->|Continuous Training| E[Adaptive Data Harvester]
    C -.->|Verilog RTL Synthesis| F[KV260 FPGA (k_max=5)]
```

## 4. Execution Phases

### Phase 0: Housekeeping
Archive .215 and initialize .216. 

### Phase 1: KAN-CL and ActFocus
Integrate the per-knot regularization for KAN energy tiers to prevent forgetting (arxiv:2605.12306). Then, apply the ActFocus loss modulation (arxiv:2605.14558) so that training gradients properly target the action tokens that break constraints.

### Phase 2: Energy-Guided Decoding
Wire up the Energy-Guided Decoding loop (arxiv:2507.07731) to force the local LLMs to generate text that minimizes the learned KAN energy landscape. 

### Phase 3: Hardware Synthesis
Take the $k_{max}=5$ Manifold Substitution architecture and build `hardware/kv260/ising_sampler_v5.v`. Synthesize it using OSS-CAD-Suite to confirm LUT limits are not exceeded and to yield the `carnot_ising.bit` bitfile.

### Phase 4: Capstone Benchmark
Perform the final E2E test using `unsloth/gemma-4-31B-it-GGUF` to validate KAN-CL retention rates, ActFocus efficiency, and Energy-Guided Decoding correctness. Produce the retrospective.

## 5. Hardware Requirements
- **Development:** Dual-RTX 3090 (48GB VRAM) for live GGUF testing and training the JEPA predictor.
- **Hardware Integration:** KV260 FPGA with OSS-CAD-Suite (yosys/nextpnr) installed locally for synthesis operations.