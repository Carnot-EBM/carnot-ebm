# Hardware Portfolio Narrowing

**Run date:** 20260507
**Experiment:** 1460
**Spec:** REQ-HW-049, SCENARIO-HW-049

## Decision

Carnot keeps three active hardware tracks and defers the rest:

1. Dual RTX 3090 CUDA local SOTA runtime repair.
2. KV260/FPGA Discrete SB RTL lint and simulation.
3. THRML/Extropic TSU compatibility simulation.

This is a scope decision, not a new performance claim. No KV260 board
execution, Extropic hardware access, NPU acceleration, photonic execution,
or live SOTA inference is claimed by this decision.

## Rationale

The active tracks were selected for immediate research value and current
readiness:

| Active track | Evidence | Research value | Current boundary |
|---|---|---|---|
| Dual RTX 3090 CUDA runtime repair | Exp 1442 saw two RTX 3090s and cached Qwen/Gemma GGUF models, but llama.cpp failed to load `libcudart.so.12`. | Unblocks headline-eligible live repair and full-pipeline benchmark evidence. | No live SOTA inference claim until a smoke run records `usable_response=true`. |
| KV260/FPGA Discrete SB RTL lint and simulation | Exp 1451 completed Verilator lint and Icarus simulation for `hardware/kv260/discrete_sb_256.v`; Vivado was absent. | Keeps the FPGA sampler path alive with local HDL evidence. | No KV260 board, bitfile, or latency claim until Vivado synthesis, bitfile flashing, and board commands are captured. |
| THRML/Extropic TSU compatibility simulation | Research references identify THRML as the near-term public software surface while Z1/XTR-0 hardware access remains unavailable. | Preserves TSU-facing sampler portability without blocking on external hardware. | No Extropic hardware access, Z1/XTR-0 execution, or TSU latency claim until authenticated hardware execution is recorded. |

## Deferred Tracks And Reopen Conditions

| Deferred track | Why out of scope now | Reopen condition |
|---|---|---|
| KV260 board execution and latency claims | Vivado is absent, no bitfile exists, and no board commands ran. | Reopen when Vivado synthesis produces a bitfile, `CARNOT_KV260_BITFILE` points to it, and a KV260/PYNQ run records latency. |
| AMD Strix/XDNA NPU acceleration | VitisAI and IRON paths remain blocked; no NPU kernel produced acceleration evidence. | Reopen when `mlir-aie` or AMD VitisAI onnxruntime is installed and a local benchmark reports real NPU speedup. |
| Extropic Z1/XTR-0 hardware execution | No local Extropic hardware or authenticated execution transcript exists. | Reopen when early-access credentials or hardware allow a THRML/SDK run with model, device, latency, and sample-quality evidence. |
| Photonic or optical Ising-machine substrates | No local optical hardware, simulator-to-hardware API, or collaborator run exists. | Reopen when a concrete photonic provider/API/collaborator can run Carnot Ising cases. |
| D-Wave QPU cloud experiments | Cloud QPU work would add a branch while the immediate blockers are local runtime and RTL readiness. | Reopen when a specific Ising/QUBO benchmark needs QPU evaluation and Leap access plus budget are available. |
| Alveo/Agilex large production FPGA | Production FPGA purchases do not help until the KV260 flow closes. | Reopen after KV260 lint, synthesis, and board execution produce a measured sampler result that justifies larger fabric. |
| RX 7900 XTX Thunderbolt eGPU | The visible dual RTX 3090 CUDA path is more ready for immediate SOTA runtime repair. | Reopen if the RTX path is exhausted or the eGPU is connected and ROCm/JAX passes a real Carnot benchmark. |

## Honest Verdict

The portfolio is narrowed to three active tracks with no KV260 board,
Extropic, NPU, or photonic execution claim. The deferred tracks remain
documented future options, but they should not receive new milestone tasks
until their reopen conditions are met.
